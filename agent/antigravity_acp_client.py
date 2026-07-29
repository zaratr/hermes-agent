"""OpenAI-compatible shim that forwards Hermes requests to `agy-acp` over stdio.

This adapter lets Hermes treat the Antigravity ACP server (`agy-acp-windows-x64.exe`)
as a chat-style LLM backend ("The Brain") while Hermes acts natively as "The Body"
managing context and tools.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.copilot_acp_client import (
    CopilotACPClient,
    _resolve_home_dir,
    _extract_tool_calls_from_text,
    _completion_to_stream_chunks,
)
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "acp://antigravity"

DEFAULT_AGY_ACP_BIN = r"C:\Users\zarat\.gemini\antigravity\agy-acp\dist\agy-acp-windows-x64.exe"
DEFAULT_AGY_BIN = r"C:\Users\zarat\AppData\Local\agy\bin\agy.exe"


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_ANTIGRAVITY_ACP_COMMAND", "").strip()
        or DEFAULT_AGY_ACP_BIN
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_ANTIGRAVITY_ACP_ARGS", "").strip()
    if not raw:
        return ["--stdio"]
    return shlex.split(raw)


def _build_subprocess_env() -> dict[str, str]:
    env = hermes_subprocess_env(inherit_credentials=True)
    home = _resolve_home_dir()
    env["HOME"] = home
    try:
        from hermes_constants import apply_subprocess_home_env
        apply_subprocess_home_env(env)
    except Exception:
        pass
    env["AGY_BIN"] = os.getenv("AGY_BIN", DEFAULT_AGY_BIN)
    return env


def _build_prompt_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ACP-native prompt blocks from the last user message.

    agy manages its own conversation state (SQLite) across turns, so we must
    only send the *current* user turn — not the full conversation history that
    Hermes reconstructs. Sending the entire transcript as a single blob causes
    the model to see it as a long task instruction and respond with a generic
    greeting instead of answering the actual question.

    For vision: image_url content blocks are converted to ACP ``resource``
    blocks so they reach the model instead of being silently dropped.
    """
    # Walk backwards to find the last user message
    last_user: dict[str, Any] | None = None
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
            last_user = msg
            break

    if last_user is None:
        return [{"type": "text", "text": ""}]

    content = last_user.get("content")
    blocks: list[dict[str, Any]] = []

    if isinstance(content, str):
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "")
            if part_type == "text":
                text = str(part.get("text") or "").strip()
                if text:
                    blocks.append({"type": "text", "text": text})
            elif part_type == "image_url":
                # OpenAI image_url format: {"type": "image_url", "image_url": {"url": "..."}}
                image_url_obj = part.get("image_url") or {}
                url = str(image_url_obj.get("url") or "")
                if url.startswith("data:"):
                    # base64 data URI — embed inline as ACP resource
                    try:
                        header, b64data = url.split(",", 1)
                        mime = header.split(";")[0].replace("data:", "")
                        raw = base64.b64decode(b64data)
                        # ACP resource block with embedded binary as base64
                        blocks.append({
                            "type": "resource",
                            "resource": {
                                "uri": f"data:{mime};base64,{b64data}",
                                "mimeType": mime,
                                "blob": b64data,
                            },
                        })
                    except Exception:
                        pass
                elif url.startswith("file://"):
                    # Local file — read and embed
                    try:
                        file_path = Path(url.replace("file://", ""))
                        raw = file_path.read_bytes()
                        mime = _guess_mime(file_path.suffix.lower())
                        b64data = base64.b64encode(raw).decode("ascii")
                        blocks.append({
                            "type": "resource",
                            "resource": {
                                "uri": url,
                                "mimeType": mime,
                                "blob": b64data,
                            },
                        })
                    except Exception:
                        pass
                else:
                    # Remote URL — send as resource_link
                    blocks.append({
                        "type": "resource_link",
                        "uri": url,
                        "title": url,
                    })
    else:
        blocks.append({"type": "text", "text": str(content or "")})

    return blocks or [{"type": "text", "text": ""}]


def _guess_mime(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")


class AntigravityACPClient(CopilotACPClient):
    """OpenAI-client-compatible facade for Antigravity ACP (`agy-acp`)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key or "antigravity-acp",
            base_url=base_url or ACP_MARKER_BASE_URL,
            default_headers=default_headers,
            acp_command=acp_command or command or _resolve_command(),
            acp_args=list(acp_args or args or _resolve_args()),
            acp_cwd=acp_cwd,
            **kwargs,
        )
        self._model_id = model

    # ------------------------------------------------------------------
    # Override _create_chat_completion to extract only the last user
    # message instead of re-formatting the full transcript.  The parent
    # class (CopilotACPClient) flattens ALL messages into a single blob
    # which confuses agy (it already manages its own conversation state).
    # ------------------------------------------------------------------

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        from agent.copilot_acp_client import _DEFAULT_TIMEOUT_SECONDS

        prompt_blocks = _build_prompt_blocks(messages or [])

        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        response_text, reasoning_text = self._run_prompt_blocks(
            prompt_blocks,
            timeout_seconds=_effective_timeout,
        )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "antigravity-acp",
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float) -> tuple[str, str]:
        """Wrap plain text into a single ACP text block and delegate."""
        return self._run_prompt_blocks(
            [{"type": "text", "text": prompt_text}],
            timeout_seconds=timeout_seconds,
        )

    def _run_prompt_blocks(
        self,
        prompt_blocks: list[dict[str, Any]],
        *,
        timeout_seconds: float,
    ) -> tuple[str, str]:
        """Send prompt_blocks to agy-acp and return (text, reasoning)."""
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self._acp_cwd,
                env=_build_subprocess_env(),
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Antigravity ACP command '{self._acp_command}'."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Antigravity ACP process did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()

        next_id = 0

        def _request(
            method: str,
            params: dict[str, Any],
            *,
            text_parts: list[str] | None = None,
            reasoning_parts: list[str] | None = None,
        ) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._handle_server_message(
                    msg,
                    process=proc,
                    cwd=self._acp_cwd,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                ):
                    continue

                if msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise RuntimeError(
                        f"Antigravity ACP {method} failed: {err.get('message') or err}"
                    )
                return msg.get("result")

            stderr_text = "\n".join(stderr_tail).strip()
            if proc.poll() is not None and stderr_text:
                raise RuntimeError(f"Antigravity ACP process exited early: {stderr_text}")
            raise TimeoutError(f"Timed out waiting for Antigravity ACP response to {method}.")

        try:
            _request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {
                            "readTextFile": True,
                            "writeTextFile": True,
                        }
                    },
                    "clientInfo": {
                        "name": "hermes-agent",
                        "title": "Hermes Agent",
                        "version": "0.19.0",
                    },
                },
            )
            session = _request(
                "session/new",
                {
                    "cwd": self._acp_cwd,
                    "mcpServers": [],
                    "permissionMode": "bypassPermissions",
                },
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError("Antigravity ACP did not return a sessionId.")

            if self._model_id and any(
                str(self._model_id).lower().startswith(p)
                for p in ("gemini-", "claude-", "gpt-")
            ):
                try:
                    _request(
                        "session/set_config_option",
                        {
                            "sessionId": session_id,
                            "configId": "model",
                            "value": str(self._model_id),
                        },
                    )
                except Exception:
                    pass

            try:
                _request(
                    "session/set_config_option",
                    {
                        "sessionId": session_id,
                        "configId": "mode",
                        "value": "bypassPermissions",
                    },
                )
            except Exception:
                pass

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            _request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": prompt_blocks,
                },
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
            )
            return "".join(text_parts), "".join(reasoning_parts)
        finally:
            self.close()
