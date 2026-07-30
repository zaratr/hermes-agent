"""Tests for AntigravityACPClient."""

import base64
import io
import json
from unittest.mock import MagicMock, patch

from agent import antigravity_acp_client as acp_module
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


def test_build_prompt_blocks_exposes_hermes_tools_to_acp_model():
    """Every Antigravity model receives the Hermes tool schemas it may call."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_terminal",
                "description": "Read the integrated desktop terminal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_line": {"type": "integer"},
                    },
                },
            },
        }
    ]

    blocks = _build_prompt_blocks(
        [{"role": "user", "content": "Read the terminal and diagnose the error."}],
        model="gemini-3.6-flash-medium",
        tools=tools,
    )

    prompt = next(block["text"] for block in blocks if block["type"] == "text")
    assert "read_terminal" in prompt
    assert "Read the integrated desktop terminal." in prompt
    assert "<tool_call>" in prompt


def test_build_prompt_blocks_includes_current_turn_tool_result():
    """A fresh ACP session receives the Hermes tool output it must interpret."""
    messages = [
        {"role": "user", "content": "An unrelated old request."},
        {"role": "assistant", "content": "An unrelated old response."},
        {"role": "user", "content": "Read the terminal and diagnose the error."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_terminal_1",
                    "type": "function",
                    "function": {"name": "read_terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_terminal_1",
            "name": "read_terminal",
            "content": "Traceback: pydantic.errors.PydanticImportError: BaseSettings moved.",
        },
    ]

    blocks = _build_prompt_blocks(messages, model="gemini-3.6-flash-medium")

    prompt = next(block["text"] for block in blocks if block["type"] == "text")
    assert "Read the terminal and diagnose the error." in prompt
    assert "read_terminal" in prompt
    assert "PydanticImportError" in prompt
    assert "An unrelated old request." not in prompt


def test_materialize_prompt_blocks_indirects_oversized_prompt(tmp_path):
    """Large Hermes tool catalogs reach agy through a file, not argv."""
    original_prompt = "tool schemas:\n" + ("x" * 50_000) + "\nLATEST REQUEST"

    with acp_module._materialize_prompt_blocks(
        [{"type": "text", "text": original_prompt}],
        workdir=tmp_path,
    ) as wire_blocks:
        wire_text = "\n".join(
            str(block.get("text") or "") for block in wire_blocks
        )
        prompt_files = list(tmp_path.rglob("prompt.txt"))

        assert len(wire_text) < 28_000
        assert original_prompt not in wire_text
        assert len(prompt_files) == 1
        assert prompt_files[0].read_text(encoding="utf-8") == original_prompt
        assert str(prompt_files[0]) in wire_text

    assert not list(tmp_path.glob(".hermes-antigravity-*"))


def test_materialize_prompt_blocks_decodes_image_for_native_vision(tmp_path):
    """Binary ACP resources become short local image references for agy."""
    raw = b"realistic-png-payload"
    b64 = base64.b64encode(raw).decode("ascii")
    blocks = [
        {"type": "text", "text": "Describe the attached image."},
        {
            "type": "resource",
            "resource": {
                "uri": f"data:image/png;base64,{b64}",
                "mimeType": "image/png",
                "blob": b64,
            },
        },
    ]

    with acp_module._materialize_prompt_blocks(
        blocks, workdir=tmp_path
    ) as wire_blocks:
        wire_json = json.dumps(wire_blocks)
        wire_text = str(wire_blocks[0].get("text") or "")
        image_files = list(tmp_path.rglob("image-1.png"))

        assert b64 not in wire_json
        assert len(image_files) == 1
        assert image_files[0].read_bytes() == raw
        assert str(image_files[0]) in wire_text
        assert "native multimodal vision" in wire_text

    assert not list(tmp_path.glob(".hermes-antigravity-*"))


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
    calls the iterator protocol, NOT ``readline()``. We back stdout with a real
    StringIO so the iterator works correctly and the thread drains naturally.
    """
    proc = MagicMock()
    proc.poll.return_value = None
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.flush = MagicMock()
    proc.stdout = io.StringIO("".join(responses))
    proc.stderr = io.StringIO("")
    return proc


@patch("agent.antigravity_acp_client._build_subprocess_env", return_value={})
@patch("hermes_cli._subprocess_compat.windows_hide_flags", return_value=0, create=True)
@patch("subprocess.Popen")
def test_antigravity_acp_client_prompt_flow(mock_popen, _mock_flags, _mock_env):
    """Verify the client completes the full JSON-RPC handshake and returns content."""
    responses = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "test-session-123"}}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}) + "\n",  # model option
        json.dumps({"jsonrpc": "2.0", "id": 4, "result": {}}) + "\n",  # mode option
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
        json.dumps({"jsonrpc": "2.0", "id": 5, "result": {"stopReason": "end_turn"}}) + "\n",
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
        json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}) + "\n",  # model option
        json.dumps({"jsonrpc": "2.0", "id": 4, "result": {}}) + "\n",  # mode option
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
        json.dumps({"jsonrpc": "2.0", "id": 5, "result": {"stopReason": "end_turn"}}) + "\n",
    ]

    mock_popen.return_value = _make_proc_mock(responses)

    client = AntigravityACPClient(model="gemini-3.6-flash-low")
    response = client.chat.completions.create(
        model="gemini-3.6-flash-low",
        messages=[{"role": "user", "content": "think and respond"}],
    )

    assert "Reasoning enabled." in response.choices[0].message.content
    write_calls = mock_popen.return_value.stdin.write.call_args_list
    methods_written = [json.loads(call.args[0])["method"] for call in write_calls]
    assert "session/set_config_option" in methods_written
    assert len(write_calls) == 5
