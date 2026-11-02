import copy
import json

import pytest

from nexus_agent_ai.agent.memory import is_genuine_user_message
from nexus_agent_ai.agent.outcomes import (
    current_claim_correction, fitted_claim_view, mutation_metadata,
)


def event(path, *, number=1, turn="t1", segment="s1", status="unchanged",
          success=False, failed=True, **extra):
    return {"event": f"e{number}", "turn": turn, "segment": segment,
            "path": str(path), "action": "patch_file", "status": status,
            "success": success, "failed": failed, **extra}


def assistant(text, events, *, turn="t1", segment="s1", **extra):
    return {"role": "assistant", "content": text,
            "metadata": {"turn": turn, "segment": segment, **mutation_metadata(events)}, **extra}


def corrections(view):
    return [message for message in view if message.get("metadata", {}).get("outcome_correction")]


@pytest.mark.parametrize("text", [
    "The subtract function has been successfully added to calculator.py.",
    "Your updated calculator.py contains a subtract function.",
    "Here is your updated calculator.py containing subtract.",
    "Updated calculator.py with the subtract function.",
    "I added a subtract function to calculator.py.",
    "The subtract function was written to calculator.py.",
    "calculator.py was updated.",
])
def test_e08_e09_state_shapes_preserved_without_semantic_absence(tmp_path, text):
    messages = [{"role": "user", "content": "Please add subtract."},
                assistant(text, [event(tmp_path / "calculator.py")])]
    before = copy.deepcopy(messages)
    view = fitted_claim_view(messages)
    assert view[1]["content"] == text
    assert "claim_redactions" not in view[1]["metadata"]
    assert len(corrections(view)) == 1
    note = corrections(view)[0]
    assert note["role"] == "user" and not is_genuine_user_message(note)
    assert "no executor write" in note["content"]
    assert not any(word in note["content"].lower() for word in ("absent", "does not exist", "false", "contradicted"))
    assert messages == before


@pytest.mark.parametrize("text", [
    "I will update calculator.py.", "You can update calculator.py.",
    "I did not update calculator.py.", "I have not updated calculator.py.",
    "For example: I updated calculator.py.", '"I updated calculator.py."',
    "For example: calculator.py was updated.",
    "```python\n# I updated calculator.py.\n```",
    "~~~\nI updated calculator.py.\n~~~",
    "calculator.py contains add().", "calculator.py contains-add().",
    "I updated unknown.py.", "I updated other.py using calculator.py.",
])
def test_advice_examples_code_and_unknown_prose_unchanged(tmp_path, text):
    messages = [{"role": "user", "content": "I updated calculator.py."},
                assistant(text, [event(tmp_path / "calculator.py")])]
    assert fitted_claim_view(messages) == messages
    assert current_claim_correction(text, messages) == ("", [])


def test_direct_actor_mixed_true_failed_unknown_and_audit_bytes(tmp_path):
    events = [event(tmp_path / "good.py", status="changed", success=True, failed=False),
              event(tmp_path / "bad.py", number=2)]
    text = "I updated good.py and bad.py.\r\nI updated unknown.py.\r\nUnknown state is preserved."
    messages = [{"role": "user", "content": "I updated bad.py."}, assistant(text, events)]
    before = copy.deepcopy(messages)
    view = fitted_claim_view(messages)
    assert "good.py" in view[1]["content"] and "unknown.py" in view[1]["content"]
    assert "bad.py" not in view[1]["content"]
    assert view[1]["content"].endswith("\r\nUnknown state is preserved.")
    assert view[1]["metadata"]["claim_redactions"] == [
        {"turn": "t1", "file": str(tmp_path / "bad.py"), "trigger": "failed_segment_actor_mutation"}]
    assert messages == before and messages[1]["content"].encode() == text.encode()
    view[0]["content"] = "disposable"
    assert messages == before


def test_same_file_success_and_failure_bind_exact_segments(tmp_path):
    path = tmp_path / "calculator.py"
    success = event(path, number=1, segment="s1", status="changed", success=True, failed=False)
    failure = event(path, number=2, segment="s2")
    messages = [{"role": "user", "content": "Update it twice"},
                assistant("I updated calculator.py.", [success]),
                assistant("I updated calculator.py.", [failure], segment="s2")]
    view = fitted_claim_view(messages)
    assert view[1]["content"] == messages[1]["content"]
    assert view[2]["content"] != messages[2]["content"]
    assert len(corrections(view)) == 1
    reverse = [{"role": "user", "content": "Try twice"},
               assistant("I updated calculator.py.", [failure], segment="s2"),
               assistant("I updated calculator.py.", [success])]
    view = fitted_claim_view(reverse)
    assert view[1]["content"] != reverse[1]["content"]
    assert view[-1]["content"] == reverse[-1]["content"]


def test_prior_success_updated_description_preserved(tmp_path):
    path = tmp_path / "calculator.py"
    messages = [assistant("I updated calculator.py.", [event(path, status="changed", success=True, failed=False)])]
    text = "Your updated calculator.py contains a subtract function."
    assert current_claim_correction(text, messages) == ("", [])
    assert fitted_claim_view(messages) == messages


@pytest.mark.parametrize("extra", [
    {"success": True, "failed": False},
    {"failed": False}, {"status": "unknown"},
    {"segment_mutated": True}, {"segment_complete": False},
    {"segment_failed_only": False},
])
def test_noop_unknown_success_and_incomplete_segments_cannot_contradict(tmp_path, extra):
    record = event(tmp_path / "calculator.py", **extra)
    messages = [assistant("I updated calculator.py.", [record])]
    assert fitted_claim_view(messages) == messages
    assert current_claim_correction("I updated calculator.py.", messages) == ("", [])


def test_old_event_without_typed_failure_cannot_trigger(tmp_path):
    record = event(tmp_path / "calculator.py")
    del record["failed"]
    messages = [assistant("I updated calculator.py.", [record])]
    assert fitted_claim_view(messages) == messages
    assert current_claim_correction("I updated calculator.py.", messages) == ("", [])


def test_same_segment_success_is_not_forgotten(tmp_path):
    path = tmp_path / "calculator.py"
    records = [event(path, status="changed", success=True, failed=False), event(path, number=2)]
    messages = [assistant("I updated calculator.py.", records)]
    assert fitted_claim_view(messages) == messages
    assert current_claim_correction("I updated calculator.py.", messages) == ("", [])


def test_three_clause_limit_shared_across_turn_segments_and_messages(tmp_path):
    path = tmp_path / "calculator.py"
    messages = [{"role": "user", "content": "Four attempts"}]
    for number in range(1, 5):
        segment = f"s{number}"
        messages.append(assistant("I updated calculator.py.", [event(path, number=number, segment=segment)], segment=segment))
    view = fitted_claim_view(messages)
    assistants = [message for message in view if message["role"] == "assistant"]
    assert all(message["content"] != "I updated calculator.py." for message in assistants[:3])
    assert assistants[3]["content"] == "I updated calculator.py."
    facts = [fact for message in assistants for fact in message["metadata"].get("claim_redactions", [])]
    assert len(facts) == 3 and len(corrections(view)) == 3


def test_segment_correction_never_interrupts_tool_result_pairs(tmp_path):
    path = tmp_path / "calculator.py"
    record = event(path)
    call = {"id": "edit", "type": "function", "function": {"name": "patch_file", "arguments": "{}"}}
    messages = [{"role": "user", "content": "Edit it"},
                assistant("I updated calculator.py.", [], tool_calls=[call]),
                {"role": "tool", "tool_call_id": "edit", "content": "ERROR: refused",
                 "metadata": {"turn": "t1", "segment": "s1", **mutation_metadata([record])}},
                assistant("Finished.", [record]),
                {"role": "user", "content": "Genuine user text stays exact."}]
    view = fitted_claim_view(messages)
    assert [message["role"] for message in view] == ["user", "assistant", "tool", "assistant", "user", "user"]
    assert view[2] == messages[2] and view[-1] == messages[-1]
    assert all(message["role"] != "system" for message in view)


def test_current_readonly_correction_pure_and_factual_with_prior_success(tmp_path):
    path = tmp_path / "calculator.py"
    records = [event(path, status="changed", success=True, failed=False),
               event(path, number=2, turn="t2", segment="s2")]
    text = "The subtract function has been successfully added to calculator.py."
    messages = [assistant("Historical state.", records)]
    before = copy.deepcopy(messages)
    suffix, facts = current_claim_correction(text, messages)
    assert suffix == (f'\n\nExecutor record: no successful mutation of "{path}" was observed in the failed edit segment(s); last successful mutation: e1.')
    assert facts == [{"turn": "t2", "file": str(path), "trigger": "current_executor_record"}]
    assert messages == before
    assert text == "The subtract function has been successfully added to calculator.py."


def test_current_no_prior_success_says_none_recorded_and_later_success_clears(tmp_path):
    path = tmp_path / "calculator.py"
    records = [event(path)]
    suffix, _ = current_claim_correction("Updated calculator.py with subtract.", [], session_events=records)
    assert "last successful mutation: none recorded." in suffix
    records.append(event(path, number=2, segment="s2", status="changed", success=True, failed=False))
    assert current_claim_correction("Updated calculator.py with subtract.", [], session_events=records) == ("", [])


def test_current_explicit_ledger_deduplicates_without_reordering(tmp_path):
    path = tmp_path / "calculator.py"
    failure = event(path)
    success = event(path, number=2, segment="s2", status="changed", success=True, failed=False)
    messages = [assistant("Old claim", [failure])]
    records = [failure, failure, success]
    before = copy.deepcopy(records)
    assert current_claim_correction("I updated calculator.py.", messages, session_events=records) == ("", [])
    assert records == before


def test_metadata_bounds_copy_dedup_and_no_legacy_map(tmp_path):
    records = [event(tmp_path / f"file-{number}.py", number=number, segment=f"s{number}") for number in range(40)]
    before = copy.deepcopy(records)
    metadata = mutation_metadata([*records, records[-1]])
    assert len({record["path"] for record in metadata["mutation_outcomes"]}) <= 8
    assert len(json.dumps(metadata, ensure_ascii=False)) <= 2048
    assert len({record["event"] for record in metadata["mutation_outcomes"]}) == len(metadata["mutation_outcomes"])
    assert metadata["mutation_outcomes"][-1]["event"] == "e39"
    assert records == before
    with pytest.raises(TypeError):
        mutation_metadata({str(tmp_path / "calculator.py"): "unchanged"})


def test_metadata_clipping_never_turns_forgotten_success_into_failure(tmp_path):
    records = [event(tmp_path / "calculator.py", status="changed", success=True, failed=False)]
    records.extend(event(tmp_path / "calculator.py", number=number) for number in range(2, 30))
    metadata = mutation_metadata(records)
    assert metadata["mutation_outcomes"]
    assert all(record["segment_complete"] is False for record in metadata["mutation_outcomes"])
    messages = [{"role": "assistant", "content": "I updated calculator.py.",
                 "metadata": {"turn": "t1", "segment": "s1", **metadata}}]
    assert fitted_claim_view(messages) == messages
    assert current_claim_correction("I updated calculator.py.", messages) == ("", [])


def test_ambiguous_same_basename_does_not_guess_touched_file(tmp_path):
    records = [event(tmp_path / "left" / "calculator.py"),
               event(tmp_path / "right" / "calculator.py", number=2)]
    messages = [assistant("I updated calculator.py.", records)]
    assert fitted_claim_view(messages) == messages


def test_similarly_named_untouched_paths_not_detected(tmp_path):
    messages = [assistant("I updated my-calculator.py.", [event(tmp_path / "calculator.py")])]
    assert fitted_claim_view(messages) == messages


def test_history_traversal_never_reads_filesystem(tmp_path, monkeypatch):
    import builtins
    import os

    messages = [assistant("I updated calculator.py.", [event(tmp_path / "calculator.py")])]

    def forbidden(*args, **kwargs):
        raise AssertionError("history must not inspect current file state")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(os, "stat", forbidden)
        patch.setattr(os, "lstat", forbidden)
        view = fitted_claim_view(messages)
        suffix, _ = current_claim_correction("I updated calculator.py.", messages)
    assert corrections(view) and "Executor record:" in suffix


def test_later_duplicate_metadata_cannot_hide_incomplete_segment(tmp_path):
    record = event(tmp_path / "calculator.py")
    incomplete = {**record, "segment_complete": False}
    messages = [
        assistant("I updated calculator.py.", [record]),
        {"role": "tool", "tool_call_id": "edit", "content": "refused",
         "metadata": {"turn": "t1", "segment": "s1", **mutation_metadata([incomplete])}},
    ]
    assert fitted_claim_view(messages) == messages
    assert current_claim_correction("I updated calculator.py.", messages) == ("", [])
