"""Consumer-visible failures found while independently verifying remediation."""
import os
import subprocess
from types import SimpleNamespace as NS

import pytest
from typer.testing import CliRunner

from nexus_agent_ai.agent import tools as T
from nexus_agent_ai.cli import app as cli
from nexus_agent_ai.providers.base import ToolCall
from nexus_agent_ai.providers.messages import (
    anthropic_messages, assistant_message, gemini_messages, normalize_messages, tool_result,
)
from tests.test_cli_outcomes import ScriptedProvider, prepare  # noqa: F401 (fixture)


@pytest.mark.parametrize("query", [
    "How do I add a dependency in poetry?",
    "Explain how to implement binary search.",
    "What changes do I need for type hints?",
    "Could you explain how to remove a list item?",
])
def test_informational_edit_keywords_do_not_require_file_changes(prepare, tmp_path, query):
    original = (tmp_path / "calc.py").read_bytes()
    result = prepare(ScriptedProvider(text="Here is an explanation of that programming operation.")).invoke(
        cli.app, ["chat", query, "--no-stream"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "calc.py").read_bytes() == original


def test_explanation_then_explicit_edit_still_requires_a_change(prepare):
    result = prepare(ScriptedProvider(text="The function currently adds its inputs.")).invoke(
        cli.app, ["chat", "Explain the error, then fix calc.py.", "--no-stream"]
    )
    assert result.exit_code == 1, result.output


def test_declined_first_run_cannot_restart_local_downloads(monkeypatch):
    from nexus_agent_ai.providers import local_provider
    monkeypatch.setenv("DEFAULT_PROVIDER", "local")
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "explicit-but-not-downloaded.gguf")
    monkeypatch.setattr("nexus_agent_ai.cli.onboarding.run_if_first_time", lambda: False)
    monkeypatch.setattr(local_provider.LocalProvider, "setup_model",
                        lambda *a, **kw: pytest.fail("startup ignored the download decline"))
    result = CliRunner().invoke(cli.app, ["chat", "hello", "--no-stream"])
    assert result.exit_code == 1, result.output
    assert "declined" in result.output.lower()


@pytest.mark.parametrize("arguments", ["", "null", None])
def test_empty_parameterless_native_arguments_translate_to_objects(arguments):
    history = [{"role": "assistant", "content": "", "tool_calls": [{
        "id": "parameterless", "type": "function",
        "function": {"name": "git_status", "arguments": arguments},
    }]}]
    assert anthropic_messages(history)[0]["content"][0]["input"] == {}
    assert gemini_messages(history)[0]["parts"][0]["function_call"]["args"] == {}


def test_invalid_native_arguments_are_not_silently_replaced():
    history = [{"role": "assistant", "content": "", "tool_calls": [{
        "id": "invalid", "type": "function",
        "function": {"name": "write_file", "arguments": '{"path":'},
    }]}]
    with pytest.raises(ValueError):
        normalize_messages(history)


def test_gemini_keeps_internal_text_separate_from_function_results():
    history = [
        assistant_message("", [ToolCall("read1", "read_file", {"path": "a.py"})]),
        tool_result("read1", "file text"),
        {"role": "user", "content": "Give the final answer.", "metadata": {"internal": True}},
    ]
    formatted = gemini_messages(history)
    assert formatted[1]["parts"] == [{
        "function_response": {"id": "read1", "name": "read_file", "response": {"result": "file text"}},
    }]
    assert formatted[2]["parts"] == [{"text": "Give the final answer."}]


@pytest.mark.parametrize("stream", [False, True])
def test_gemini_blocked_response_reports_failure_instead_of_empty_success(stream):
    from nexus_agent_ai.providers.gemini_provider import GeminiProvider
    types = pytest.importorskip("google.genai.types")
    response = NS(candidates=None, prompt_feedback=NS(block_reason="SAFETY"))
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.model, provider._types = "offline-test", types
    provider.client = NS(models=NS(
        generate_content=lambda **kw: response,
        generate_content_stream=lambda **kw: iter([response]),
    ))
    with pytest.raises(RuntimeError, match="Gemini.*SAFETY"):
        if stream:
            list(provider.stream([], [], ""))
        else:
            provider.complete([], [], "")


@pytest.mark.skipif(os.name != "nt", reason="Windows DACLs")
def test_atomic_replacement_and_backup_keep_private_windows_dacl(tmp_path):
    from nexus_agent_ai.utils.private_files import restrict_permissions
    work = tmp_path / "shared-parent"
    work.mkdir()
    subprocess.run(["icacls", str(work), "/grant", "*S-1-1-0:(OI)(CI)R"],
                   capture_output=True, check=True, text=True)
    target = work / "notes.txt"
    target.write_text("private original", encoding="utf-8")
    restrict_permissions(target, 0o600)

    def access_entries(path):
        output = subprocess.run(["icacls", str(path)], capture_output=True,
                                check=True, text=True).stdout
        return [line.replace(str(path), "", 1).strip()
                for line in output.splitlines() if ":(" in line]

    original_acl = access_entries(target)
    assert original_acl
    T.execute_read_file(str(target))
    assert T.execute_write_file(str(target), "private replacement").startswith("Successfully")
    assert target.read_text(encoding="utf-8") == "private replacement"
    assert target.with_name("notes.txt.bak").read_text(encoding="utf-8") == "private original"
    assert access_entries(target) == original_acl
    assert access_entries(target.with_name("notes.txt.bak")) == original_acl
