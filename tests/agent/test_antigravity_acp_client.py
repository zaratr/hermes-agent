"""Tests for AntigravityACPClient."""

import base64
import io
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent.antigravity_acp_client import AntigravityACPClient, _build_prompt_blocks
from agent.copilot_acp_client import CopilotACPClient


def test_antigravity_acp_client_inheritance():
    """Verify AntigravityACPClient inherits from CopilotACPClient with proper defaults."""
    client = AntigravityACPClient(model="gemini-3.6-flash-low")
    assert isinstance(client, CopilotACPClient)
    # Model is stored as _model_id (CopilotACPClient base has no .model attr)
    assert client._model_id == "gemini-3.6-flash-low"
    # Command resolves to the agy-acp binary path
    assert "agy-acp" in client._acp_command or "agy" in client._acp_command.lower()


def test_build_prompt_blocks_text_only():
    """Only the last user message text reaches the prompt blocks."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "what model are you?"},
    ]
    blocks = _build_prompt_blocks(messages)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "what model are you?"
    # System and earlier messages must NOT be in the blocks
    assert "first question" not in blocks[0]["text"]
    assert "helpful assistant" not in blocks[0]["text"]


def test_build_prompt_blocks_image_url_data_uri():
    """Data-URI images are converted to ACP resource blocks."""
    raw = b"fake-png-bytes"
    b64 = base64.b64encode(raw).decode("ascii")
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]},
    ]
    blocks = _build_prompt_blocks(messages)
    types = {b["type"] for b in blocks}
    assert "text" in types
    assert "resource" in types
    resource_block = next(b for b in blocks if b["type"] == "resource")
    assert resource_block["resource"]["mimeType"] == "image/png"
    assert resource_block["resource"]["blob"] == b64


def test_build_prompt_blocks_remote_image_url():
    """Remote image URLs become resource_link blocks."""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]},
    ]
    blocks = _build_prompt_blocks(messages)
    link_blocks = [b for b in blocks if b["type"] == "resource_link"]
    assert len(link_blocks) == 1
    assert link_blocks[0]["uri"] == "https://example.com/img.png"


def _make_proc_mock(responses: list[str]) -> MagicMock:
    """Build a Popen mock whose stdout iterates over NDJSON response lines.

    The `_run_prompt` reader thread does ``for line in proc.stdout:`` which
    calls the iterator protocol, NOT ``readline()``.  We back stdout with a
    real io.StringIO so the iterator works correctly and the thread drains
    naturally when it reaches EOF.

    Note: do NOT use ``spec=subprocess.Popen`` here — at test runtime Popen
    is already replaced with a MagicMock by ``@patch``, and Python 3.11+
    raises InvalidSpecError when you try to spec from another Mock.
    """
    proc = MagicMock()  # plain mock; no spec needed
    proc.poll.return_value = None  # process appears alive
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.flush = MagicMock()
    # Use a real StringIO so ``for line in proc.stdout`` works
    proc.stdout = io.StringIO("".join(responses))
    proc.stderr = io.StringIO("")  # empty stderr
    return proc


@patch("agent.antigravity_acp_client._build_subprocess_env", return_value={})
@patch("hermes_cli._subprocess_compat.windows_hide_flags", return_value=0, create=True)
@patch("subprocess.Popen")
def test_antigravity_acp_client_prompt_flow(mock_popen, _mock_flags, _mock_env):
    """Verify the client completes the full JSON-RPC handshake and returns content."""
    responses = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "test-session-123"}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}) + "\n",
        # session/update notification — carries the model reply.
        # Wire format from agy-acp: update is a flat object with
        # sessionUpdate="agent_message_chunk" and content at top level.
        json.dumps({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "test-session-123",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "I am Gemini 3.6 Flash."},
                },
            },
        }) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 4, "result": {"stopReason": "end_turn"}}) + "\n",
    ]

    mock_popen.return_value = _make_proc_mock(responses)

    client = AntigravityACPClient(model="gemini-3.6-flash-low")
    response = client.chat.completions.create(
        model="gemini-3.6-flash-low",
        messages=[{"role": "user", "content": "what model are you?"}],
    )

    assert "Gemini" in response.choices[0].message.content
    assert mock_popen.called


@patch("agent.antigravity_acp_client._build_subprocess_env", return_value={})
@patch("hermes_cli._subprocess_compat.windows_hide_flags", return_value=0, create=True)
@patch("subprocess.Popen")
def test_antigravity_acp_client_reasoning_config(mock_popen, _mock_flags, _mock_env):
    """Verify that AntigravityACPClient passes model config option when model_id is set."""
    responses = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "test-session-456"}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}) + "\n",  # set_config_option
        json.dumps({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "test-session-456",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Reasoning enabled."},
                },
            },
        }) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 4, "result": {"stopReason": "end_turn"}}) + "\n",
    ]

    mock_popen.return_value = _make_proc_mock(responses)

    client = AntigravityACPClient(model="gemini-3.6-flash-low")
    response = client.chat.completions.create(
        model="gemini-3.6-flash-low",
        messages=[{"role": "user", "content": "think and respond"}],
    )

    assert "Reasoning enabled." in response.choices[0].message.content
    # Verify set_config_option was called (stdin.write should have been called 4 times:
    # initialize, session/new, set_config_option, session/prompt)
    assert mock_popen.called
    write_calls = mock_popen.return_value.stdin.write.call_args_list
    methods_written = [json.loads(c.args[0])["method"] for c in write_calls]
    assert "session/set_config_option" in methods_written
