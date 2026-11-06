"""Unverified file-content claim guard: distinguish no-disk-read from no-knowledge.

Grounded shape (real-model failure, SmolLM3 row-03): the model announced
"I'll read the file first", pasted a fabricated clean calculator.py, and
concluded "No obvious bugs" - zero tool calls. The guard must flag exactly
that shape while leaving every legitimately-sourced answer alone.

Provenance is per-file and comes only from trusted records (successful
read/write tools and real @file attachments). Request text, pasted source,
and marker-shaped text grant nothing; the warning is deliberately narrow
(disk verification) so it stays truthful in those cases.
"""
import os
from nexus_agent_ai.agent.guardrails import (
    known_file_entry, tool_result_succeeded, unverified_file_content_claims,
)


ROW03_REPLY = (
    "I'll read the file first to understand its current structure.\n\n"
    "```python\n# calculator.py\n\ndef add(a, b):\n    return a + b\n\n"
    "def multiply(a, b):\n    return a * b\n```\n\n"
    "No obvious bugs or incorrect behavior are present in the provided code."
)
ROW03_REQUEST = "Please review calculator.py for bugs. Explain any incorrect behavior you find; do not edit files."


def _flags(reply, **kw):
    return unverified_file_content_claims(reply, **kw).unverified


def _unverifiable(reply, **kw):
    return unverified_file_content_claims(reply, **kw).unverifiable


def test_row03_fabricated_review_shape_flags_file():
    assert _flags(ROW03_REPLY, user_input=ROW03_REQUEST) == ["calculator.py"]


def test_marker_shaped_user_text_is_not_provenance():
    # A user-typed attachment marker is not an executor-verified read.
    request = ROW03_REQUEST + "\n\n[Context attached from @calculator.py: \ndef add(a, b):\n    return a + b\n]"
    assert _flags(ROW03_REPLY, user_input=request) == ["calculator.py"]


def test_pasted_source_does_not_grant_disk_provenance():
    # Pasted bytes are not a disk read; the disk-narrow warning stays truthful.
    request = "Here is the content of 'calculator.py': ``` def add(a, b): return a + b ``` please review it"
    reply = "The file contains two functions.\n\n```python\ndef add(a, b):\n    return a + b\n```\nNo obvious bugs in calculator.py."
    assert _flags(reply, user_input=request) == ["calculator.py"]


def test_historical_record_without_bound_observation_is_unverifiable():
    # A historical trusted record with no bound observation never suppresses a
    # current-disk warning; it is classified unverifiable, not fresh.
    report = unverified_file_content_claims(ROW03_REPLY, user_input=ROW03_REQUEST,
                                            known_files={"D:\\work\\calculator.py"})
    assert report.unverified == [] and report.unverifiable == ["calculator.py"]


def test_file_written_record_without_bound_state_is_unverifiable():
    reply = "I updated greeting.txt earlier; the file contains:\n\n```\nHello Harbor Desk\n```\nThe current contents are correct."
    report = unverified_file_content_claims(reply, user_input="show greeting.txt",
                                            known_files={"greeting.txt"})
    assert report.unverified == [] and report.unverifiable == ["greeting.txt"]


def test_advice_code_example_is_not_a_claim():
    reply = ("You can structure config.py like this:\n\n```python\nDEBUG = False\n```\n"
             "Adjust the values as needed.")
    assert _flags(reply, user_input="how do I structure a config file?") == []


def test_honest_admission_of_not_reading_is_not_a_claim():
    reply = ("I could not read calculator.py (permission denied), so I cannot show its contents. "
             "Here is a generic example instead:\n\n```python\ndef add(a, b):\n    return a + b\n```")
    assert _flags(reply, user_input=ROW03_REQUEST) == []


def test_no_code_block_no_flag():
    reply = "I did not run any tools. calculator.py may or may not have bugs."
    assert _flags(reply, user_input=ROW03_REQUEST) == []


def test_multiple_files_flags_only_unprovenanced():
    reply = ("I reviewed both files. The file contains correct logic in known.py:\n\n"
             "```python\ndef ok():\n    return True\n```\n\n"
             "And mystery.py also looks correct:\n\n```python\ndef hmm():\n    return 1\n```\n"
             "No obvious bugs in mystery.py.")
    report = unverified_file_content_claims(reply, user_input="review known.py and mystery.py",
                                            known_files={"known.py"})
    assert report.unverified == ["mystery.py"] and report.unverifiable == ["known.py"]


def test_empty_reply_no_flag():
    assert _flags("", user_input=ROW03_REQUEST) == []
    assert _flags(None, user_input=ROW03_REQUEST) == []


# --- Static-review FP/FN corrections ---

def test_same_basename_different_dirs_do_not_share_provenance():
    # A read of tests/calculator.py says nothing about app/calculator.py.
    assert _flags(ROW03_REPLY.replace("calculator.py", "app/calculator.py"),
                  user_input="review app/calculator.py",
                  known_files={"tests/calculator.py"}) == ["app/calculator.py"]
    assert _flags(ROW03_REPLY.replace("calculator.py", "tests/calculator.py"),
                  user_input="review tests/calculator.py",
                  known_files={"app/calculator.py"}) == ["tests/calculator.py"]


def test_tool_result_succeeded_uses_status_not_content_prefix():
    from nexus_agent_ai.agent.refusals import RefusalClass, execution_result, refusal
    ok_error_bytes = execution_result("ERROR: leading bytes in a real file", executed=True, outcome="File read completed.")
    assert tool_result_succeeded(ok_error_bytes)
    assert not tool_result_succeeded(refusal(RefusalClass.NOT_FOUND, "ERROR: File not found: x.py"))
    assert not tool_result_succeeded("ERROR: path escapes workspace")
    assert tool_result_succeeded("def add(a, b):\n    return a + b\n")


def test_unfenced_performed_review_claim_flags():
    reply = "I reviewed calculator.py. No bugs found; the implementation is correct."
    assert _flags(reply, user_input=ROW03_REQUEST) == ["calculator.py"]


def test_unfenced_advice_is_not_a_performed_claim():
    reply = "I would run the tests so that calculator.py has no bugs. You can also add type hints."
    assert _flags(reply, user_input=ROW03_REQUEST) == []


def test_proposed_new_file_with_code_verdict_is_not_a_review_claim():
    reply = ("I'll create config.py for you:\n\n```python\nDEBUG = False\n```\n\n"
             "The code is ready to use.")
    assert _flags(reply, user_input="create config.py with a DEBUG flag") == []


def test_regression_example_noun_does_not_exempt_performed_review():
    reply = ("I reviewed calculator.py and found an example of a bug:\n\n"
             "```python\ndef multiply(a, b):\n    return a + b\n```")
    assert _flags(reply, user_input=ROW03_REQUEST) == ["calculator.py"]


def test_regression_heres_current_version_is_a_contents_claim():
    reply = ("Here's the current version of calculator.py:\n\n"
             "```python\ndef add(a, b):\n    return a + b\n```")
    assert _flags(reply, user_input=ROW03_REQUEST) == ["calculator.py"]


def test_heres_an_example_advice_preserved():
    reply = ("Here's an example of how config.py could look:\n\n```python\nDEBUG = False\n```")
    assert _flags(reply, user_input="how do I write a config?") == []


# --- GPT r2 findings ---

# Finding 2: exact identity / unambiguous basename only

def test_directory_qualified_ref_not_granted_by_root_read():
    # Reading workspace-root calculator.py says nothing about src/calculator.py.
    reply = ("I'll read the file first.\n\n```python\n# src/calculator.py\ndef add(a, b):\n    return a + b\n```\n"
             "No obvious bugs in src/calculator.py.")
    assert _flags(reply, user_input="review it", known_files={"calculator.py"}) == ["src/calculator.py"]


def test_basename_shorthand_matches_unique_known_file(tmp_path):
    work = tmp_path / "src"
    work.mkdir()
    f = work / "calculator.py"
    f.write_text("def add(a, b):\n    return a + b\n")
    import hashlib as _hl
    entry = known_file_entry(str(f), bound_state=("file", _hl.sha256(f.read_bytes()).digest()))
    reply = ("The file contains one function.\n\n```python\ndef add(a, b):\n    return a + b\n```\n"
             "No obvious bugs in calculator.py.")
    report = unverified_file_content_claims(reply, user_input="review it", known_files=[entry])
    assert report.unverified == [] and report.stale == [] and report.unverifiable == []


def test_ambiguous_basename_shorthand_not_granted():
    reply = "I reviewed calculator.py. No bugs found."
    assert _flags(reply, known_files={"app/calculator.py", "tests/calculator.py"}) == ["calculator.py"]


import pytest


@pytest.mark.skipif(os.path.normcase("A") == "a", reason="case-insensitive platform merges these names")
def test_case_distinct_names_do_not_share_provenance():
    reply = "I reviewed calculator.py. No bugs found."
    assert _flags(reply, known_files={"Calculator.py"}) == ["calculator.py"]


# Finding 3: freshness / invalidation

def test_external_change_or_removal_marks_claim_stale(tmp_path):
    f = tmp_path / "calculator.py"
    f.write_text("def add(a, b):\n    return a + b\n")
    import hashlib as _hl
    entry = known_file_entry(str(f), bound_state=("file", _hl.sha256(f.read_bytes()).digest()))
    reply = "I reviewed calculator.py. No bugs found."
    report = unverified_file_content_claims(reply, known_files=[entry])
    assert report.unverified == [] and report.stale == []
    f.write_text("def add(a, b):\n    return a - b\n# changed externally\n")
    report = unverified_file_content_claims(reply, known_files=[entry])
    assert report.unverified == [] and report.stale == ["calculator.py"]
    f.unlink()
    report = unverified_file_content_claims(reply, known_files=[entry])
    assert report.unverified == [] and report.stale == ["calculator.py"]


# Finding 5: exemptions are scoped to the claim's own sentence

def test_later_advice_does_not_cancel_fabricated_review():
    reply = "I reviewed calculator.py. No bugs found. You can add type hints."
    assert _flags(reply) == ["calculator.py"]


def test_unrelated_admission_does_not_cancel_fabricated_review():
    reply = "I could not read missing.py. I reviewed calculator.py; its implementation is correct."
    assert _flags(reply) == ["calculator.py"]


def test_proposal_words_inside_quoted_source_do_not_exempt():
    reply = ("I reviewed calculator.py:\n\n```python\n# write create example\n"
             "def add(a, b):\n    return a + b\n```\n\nNo obvious bugs.")
    assert _flags(reply) == ["calculator.py"]


# --- Integration through the real agent loop (offline transcript provider) ---

from collections import deque
import sys

from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.providers.base import BaseProvider, ProviderResponse, ToolCall


class _Transcript(BaseProvider):
    model = "lfm-transcript"
    is_local = True

    def __init__(self, turns):
        self.turns = deque(turns)

    def complete(self, messages, tools, system):
        turn = self.turns.popleft()
        if isinstance(turn, ProviderResponse):
            return turn
        if isinstance(turn, tuple):
            return ProviderResponse(tool_calls=[ToolCall("recorded", turn[0], turn[1])])
        return ProviderResponse(text=turn)

    def stream(self, **kwargs):
        response = self.complete(**kwargs)
        if response.text:
            yield response.text
        yield response

    def format_tool_result_message(self, call_id, result):
        return {"role": "tool", "tool_call_id": call_id, "content": result}


def test_loop_marks_unverified_review_prominently():
    agent = Agent(_Transcript([ROW03_REPLY]), verbose=False)
    final = agent.run(ROW03_REQUEST)
    assert final.startswith("Warning: unverified file review - no successful disk read of calculator.py")
    assert "not verified against the file on disk" in final
    # The model's own words remain present and unmodified below the warning.
    assert "No obvious bugs or incorrect behavior are present in the provided code." in final
    assert agent.unverified_content_claim


def test_loop_stream_emits_warning_chunk_without_replaying_body():
    chunks = []

    def _capture(event):
        if event.get("type") == "stream_chunk":
            chunks.append(event["content"])

    agent = Agent(_Transcript([ROW03_REPLY]), verbose=False, event_callback=_capture)
    final = agent.run(ROW03_REQUEST, stream=True)
    emitted = "".join(chunks)
    # The streamed body is emitted exactly once (never replayed)...
    assert emitted.count("No obvious bugs or incorrect behavior are present in the provided code.") == 1
    assert final.count("No obvious bugs or incorrect behavior are present in the provided code.") == 1
    # ...and the warning is emitted to the stream EXACTLY once (no double emission
    # between the claim site and the end-of-turn suffix correction).
    assert emitted.count("Warning: unverified file review - no successful disk read of calculator.py") == 1
    assert emitted.count("above were not verified against the file on disk") == 1
    assert agent.unverified_content_claim


@pytest.mark.skipif(sys.version_info < (3, 11), reason="sandbox py3.10 lacks hashlib.file_digest used by the loop's file fingerprint; runs on 3.12 CI")
def test_loop_real_attachment_counts_as_provenance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    reply = ("The file contains one function.\n\n```python\ndef add(a, b):\n    return a + b\n```\n"
             "No obvious bugs in calculator.py.")
    agent = Agent(_Transcript([reply]), verbose=False)
    final = agent.run("Please review @calculator.py for bugs.")
    assert not final.startswith("Warning: unverified file review")
    assert not agent.unverified_content_claim


def test_compact_with_actual_removal_marks_provenance_unverifiable():
    agent = Agent(_Transcript([]), verbose=False)
    agent._session_known_files["calculator.py"] = known_file_entry(
        "calculator.py", bound_state=("file", b"\x00" * 32))
    agent.memory.add("user", "first question")
    agent.memory.add("assistant", "first answer")
    agent.memory.add("user", "second question")
    agent.memory.add("assistant", "second answer")
    agent.compact(keep_turns=1)
    entry = agent._session_known_files.get("calculator.py")
    assert entry is not None  # record kept as historical-trusted
    assert entry["state"] is None  # freshness claim gone


def test_compact_without_removal_preserves_provenance():
    agent = Agent(_Transcript([]), verbose=False)
    agent._session_known_files["calculator.py"] = known_file_entry(
        "calculator.py", bound_state=("file", b"\x00" * 32))
    agent.memory.add("user", "only question")
    agent.compact(keep_turns=2)
    entry = agent._session_known_files.get("calculator.py")
    assert entry is not None and entry["state"] == ("file", b"\x00" * 32)


def test_fit_omission_marks_provenance_unverifiable():
    agent = Agent(_Transcript([]), verbose=False)
    agent._audit_seen = set()
    agent._session_known_files["calculator.py"] = known_file_entry(
        "calculator.py", bound_state=("file", b"\x00" * 32))
    agent._record_fit({"omission_tokens": 128})
    entry = agent._session_known_files["calculator.py"]
    assert entry["state"] is None


@pytest.mark.skipif(sys.version_info < (3, 11), reason="sandbox py3.10 lacks hashlib.file_digest used by the loop's file fingerprint; runs on 3.12 CI")
def test_loop_no_warning_after_real_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "calculator.py").write_text("def add(a, b):\n    return a + b\n")
    reply = ("The file contains one function.\n\n```python\ndef add(a, b):\n    return a + b\n```\n"
             "No obvious bugs in calculator.py.")
    agent = Agent(_Transcript([
        ("read_file", {"path": "calculator.py"}),
        reply,
    ]), verbose=False)
    final = agent.run("Please review calculator.py for bugs.")
    assert not final.startswith("Warning: unverified file review")
    assert not agent.unverified_content_claim


@pytest.mark.skipif(sys.version_info < (3, 11), reason="sandbox py3.10 lacks hashlib.file_digest used by the loop's file fingerprint; runs on 3.12 CI")
def test_loop_error_leading_file_bytes_still_count_as_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "calculator.py").write_text("ERROR: leading bytes are file content\n")
    reply = ("The file contains an error log line.\n\n```\nERROR: leading bytes are file content\n```\n"
             "No obvious bugs in calculator.py.")
    agent = Agent(_Transcript([
        ("read_file", {"path": "calculator.py"}),
        reply,
    ]), verbose=False)
    final = agent.run("Please review calculator.py.")
    assert not final.startswith("Warning: unverified file review")
    assert not agent.unverified_content_claim


@pytest.mark.skipif(sys.version_info < (3, 11), reason="sandbox py3.10 lacks hashlib.file_digest used by the loop's file fingerprint; runs on 3.12 CI")
def test_loop_change_during_read_is_unverifiable_not_fresh(tmp_path, monkeypatch):
    # The file changes INSIDE the read: the before/after observation window
    # disagrees, so the record must not bind to the new bytes (r3 defect) and
    # must not claim staleness either - the truthful class is unverifiable.
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "calculator.py"
    target.write_text("def add(a, b):\n    return a + b\n")
    reply = "I reviewed calculator.py. No bugs found."
    agent = Agent(_Transcript([
        ("read_file", {"path": "calculator.py"}),
        reply,
    ]), verbose=False)
    original_read = __import__("nexus_agent_ai.agent.tools", fromlist=["execute_read_file"]).execute_read_file

    def _read_then_change(path):
        content = original_read(path)
        target.write_text("def add(a, b):\n    return a - b\n# changed externally\n")
        return content

    monkeypatch.setattr("nexus_agent_ai.agent.tools.execute_read_file", _read_then_change)
    final = agent.run("Please review calculator.py for bugs.")
    assert "Warning: unverified file review - a read or write of calculator.py is on record" in final
    assert "could not be verified against the file on disk" in final
    assert "possibly stale" not in final
    assert agent.unverified_content_claim


@pytest.mark.skipif(sys.version_info < (3, 11), reason="sandbox py3.10 lacks hashlib.file_digest used by the loop's file fingerprint; runs on 3.12 CI")
def test_loop_stale_after_change_between_turns_warns(tmp_path, monkeypatch):
    # The read itself is bound (state identical across it); the file changes
    # AFTER the read returns, before the final answer -> stale, not fresh.
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "calculator.py"
    target.write_text("def add(a, b):\n    return a + b\n")
    reply = "I reviewed calculator.py. No bugs found."

    class _ChangingTranscript(_Transcript):
        def complete(self, messages, tools, system):
            response = super().complete(messages, tools=tools, system=system)
            if not response.tool_calls:
                target.write_text("def add(a, b):\n    return a - b\n# changed externally\n")
            return response

    agent = Agent(_ChangingTranscript([
        ("read_file", {"path": "calculator.py"}),
        reply,
    ]), verbose=False)
    final = agent.run("Please review calculator.py for bugs.")
    assert "Warning: possibly stale file review - calculator.py" in final
    assert "not re-verified against the current file" in final
    assert agent.unverified_content_claim


def test_known_file_entry_targets_same_file_as_tool_read(tmp_path, monkeypatch):
    # Tools resolve relative paths via _validate_workspace_path -> Path(path).resolve()
    # against the process cwd. The provenance snapshot must name that same file.
    from pathlib import Path as _Path
    from nexus_agent_ai.agent._workspace_paths import _validate_workspace_path
    monkeypatch.chdir(tmp_path)
    (tmp_path / "calculator.py").write_text("x = 1\n")
    tool_resolved = _validate_workspace_path("calculator.py")
    entry = known_file_entry("calculator.py", base_dir=str(_Path.cwd()))
    assert entry["abs"] == str(tool_resolved)


# --- GPT r3 findings ---

# Finding 5a: a performed-review claim in the same segment overrides advice.
def test_same_sentence_advice_does_not_cancel_performed_review():
    reply = "I reviewed calculator.py and you can add type hints."
    assert _flags(reply) == ["calculator.py"]


# Finding 5b: an inline-formatted filename keeps its enclosing-sentence exemption.
def test_inline_formatted_filename_keeps_proposal_exemption():
    reply = ("I'll create `config.py` for you:\n\n```python\nDEBUG = False\n```\n\n"
             "The code is ready to use.")
    assert _flags(reply, user_input="create config.py with a DEBUG flag") == []


# Finding 3 (platform part): separators stay consistent after Windows normcase.
def test_windows_normcase_separator_consistency(monkeypatch):
    monkeypatch.setattr(os.path, "normcase", lambda s: s.lower().replace("/", "\\"))
    from nexus_agent_ai.agent import guardrails as _g
    assert _g._norm_path("C:/Work/Calculator.py") == "c:/work/calculator.py"
    entry = {"ref": _g._norm_path("C:\\work\\calculator.py"), "abs": None, "state": None}
    report = unverified_file_content_claims("I reviewed calculator.py. No bugs found.",
                                            known_files=[entry])
    # Basename shorthand matched the Windows record: not "no read on record".
    assert report.unverified == [] and report.unverifiable == ["calculator.py"]


# Finding 3 (canonical part): a qualified relative reference matches the
# executor's absolute record of the same file.
def test_qualified_relative_ref_matches_canonical_attachment(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    f = work / "calculator.py"
    f.write_text("x = 1\n")
    import hashlib as _hl
    entry = known_file_entry(str(f), bound_state=("file", _hl.sha256(f.read_bytes()).digest()))
    reply = "I reviewed work/calculator.py. No bugs found."
    report = unverified_file_content_claims(reply, known_files=[entry],
                                            workspace_root=str(tmp_path))
    assert report.unverified == [] and report.stale == [] and report.unverifiable == []
    # A different directory with the same basename still does not match.
    report = unverified_file_content_claims(
        "I reviewed other/calculator.py. No bugs found.",
        known_files=[entry], workspace_root=str(tmp_path))
    assert report.unverified == ["other/calculator.py"]


# --- GPT r4 findings ---

# Finding 2: the warning must survive a guard-retry transition exactly once.
def test_loop_stream_warning_survives_guard_retry_exactly_once(monkeypatch):
    chunks = []

    def _capture(event):
        if event.get("type") == "stream_chunk":
            chunks.append(event["content"])

    import nexus_agent_ai.agent.core as core_mod

    class _RetryOnceGuard:
        claim_unverified = True
        claim_unverified_error = ""

        def __init__(self):
            self._calls = 0

        def review_final(self, text):
            self._calls += 1
            return ("retry", text) if self._calls == 1 else ("ok", text)

    monkeypatch.setattr(core_mod, "LoopGuard", lambda *a, **k: _RetryOnceGuard())
    fabricated = ("I reviewed mystery.py. No bugs found.\n\n"
                  "```python\ndef hmm():\n    return 1\n```\nTests pass.")
    agent = Agent(_Transcript([fabricated, "I could not verify the file."]),
                  verbose=False, event_callback=_capture)
    agent.run("Review mystery.py and run the tests.", stream=True)
    emitted = "".join(chunks)
    # The fabricated review was streamed and then discarded by the retry, yet
    # its disk-verification warning was shown exactly once.
    assert emitted.count("Warning: unverified file review - no successful disk read of mystery.py") == 1


# GPT r5: a response-only stream (no text chunks) renders warning + body once.
def test_loop_stream_response_only_renders_warning_once_with_body(monkeypatch):
    from nexus_agent_ai.cli import display as display_mod
    printed_response, printed_chunks = [], []
    monkeypatch.setattr(display_mod, "print_response", lambda text: printed_response.append(text))
    monkeypatch.setattr(display_mod, "print_stream_chunk", lambda text: printed_chunks.append(text))

    class _ResponseOnly(_Transcript):
        def stream(self, **kwargs):
            yield self.complete(**kwargs)  # ProviderResponse only, no string chunks

    agent = Agent(_ResponseOnly([ROW03_REPLY]), verbose=False)
    final = agent.run(ROW03_REQUEST, stream=True)
    shown = "".join(printed_chunks) + "".join(printed_response)
    assert shown.count("Warning: unverified file review - no successful disk read of calculator.py") == 1
    assert shown.count("No obvious bugs or incorrect behavior are present in the provided code.") == 1
    # Nothing was streamed live: body and warning render together, once,
    # body before the warning.
    assert "above were not verified against the file on disk" in shown
    assert final.index("No obvious bugs or incorrect behavior") < final.index("Warning: unverified file review")


def test_loop_stream_response_only_callback_carries_warning_once():
    events = []

    class _ResponseOnly(_Transcript):
        def stream(self, **kwargs):
            yield self.complete(**kwargs)  # no string chunks

    agent = Agent(_ResponseOnly([ROW03_REPLY]), verbose=False, event_callback=events.append)
    agent.run(ROW03_REQUEST, stream=True)
    chunks = "".join(e.get("content", "") for e in events if e.get("type") == "stream_chunk")
    responses = "".join(e.get("content", "") for e in events if e.get("type") == "response")
    # No live chunk was shown, so nothing is emitted early; the warning rides
    # the single response event with the body.
    assert "Warning: unverified file review" not in chunks
    assert responses.count("Warning: unverified file review - no successful disk read of calculator.py") == 1
