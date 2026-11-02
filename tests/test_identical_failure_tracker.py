import hashlib

import pytest

from nexus_agent_ai.agent.guardrails import IdenticalFailureTracker


STATE = ("file", hashlib.sha256(b"original file\n").hexdigest())
CHANGED = ("file", hashlib.sha256(b"changed file\n").hexdigest())
ARGS = {"path": "example.py", "target": "old\n", "replacement": "new\n"}
ERROR = "ERROR: Target content not found in 'example.py'. Ensure the target matches exactly."


def observe(tracker, args=None, result=ERROR, before=STATE, after=STATE, name="patch_file"):
    return tracker.observe(name, ARGS if args is None else args, result, before, after)


def test_second_identical_failure_stops():
    tracker = IdenticalFailureTracker()
    assert not observe(tracker)
    assert observe(tracker)
    assert tracker.last_error_class == "target-mismatch"


def test_actual_sha256_digest_bytes_support_repeated_failure():
    tracker = IdenticalFailureTracker()
    state = ("file", hashlib.sha256(b"original file\n").digest())
    changed = ("file", hashlib.sha256(b"changed file\n").digest())
    assert not observe(tracker, before=state, after=state)
    assert observe(tracker, before=state, after=state)
    assert not observe(tracker, before=state, after=changed)
    assert not observe(tracker, before=changed, after=changed)
    assert observe(tracker, before=changed, after=changed)


def test_other_failures_do_not_erase_matching_key():
    tracker = IdenticalFailureTracker()
    assert not observe(tracker)
    assert not observe(tracker, result="ERROR: File not found: example.py")
    assert observe(tracker)


@pytest.mark.parametrize("success", ["done", "", "No changes: already requested content."])
def test_success_clears_all_evidence(success):
    tracker = IdenticalFailureTracker()
    assert not observe(tracker)
    assert not observe(tracker, result=success, before=None, after=None)
    assert tracker.last_error_class is None
    assert not observe(tracker)
    assert observe(tracker)


def test_begin_turn_resets_identical_independent_request():
    tracker = IdenticalFailureTracker()
    assert not observe(tracker)
    tracker.begin_turn()
    assert tracker.last_error_class is None
    assert not observe(tracker)
    assert observe(tracker)


def test_changed_file_hash_is_a_different_key():
    tracker = IdenticalFailureTracker()
    assert not observe(tracker)
    assert not observe(tracker, before=CHANGED, after=CHANGED)
    assert observe(tracker, before=CHANGED, after=CHANGED)


@pytest.mark.parametrize("before,after", [
    (STATE, CHANGED), (STATE, None), (None, STATE), (None, None),
    (("file", None), ("file", None)), (("file", "not-a-sha256"), ("file", "not-a-sha256")),
    (("missing", None), STATE), (STATE, ("missing", None)),
])
def test_unverified_or_mutating_call_cannot_count_as_repeat(before, after):
    tracker = IdenticalFailureTracker()
    assert not observe(tracker)
    assert not observe(tracker, before=before, after=after)
    assert not observe(tracker, before=before, after=after)
    assert not observe(tracker)
    assert observe(tracker)


@pytest.mark.parametrize("state", [("missing", None), ("no-target", None)])
def test_verified_missing_and_explicit_targetless_states(state):
    tracker = IdenticalFailureTracker()
    assert not observe(tracker, before=state, after=state)
    assert observe(tracker, before=state, after=state)


def test_tool_names_do_not_conflate():
    tracker = IdenticalFailureTracker()
    assert not observe(tracker, name="patch_file")
    assert not observe(tracker, name="write_file")
    assert observe(tracker, name="patch_file")


@pytest.mark.parametrize("left,right", [
    ("line", "line\n"), ("line\n", "line\r\n"),
    ("    line", "\tline"), ("line ", "line"),
    (" a\nb", "a\n b"), ("é", "e\u0301"),
    (False, 0), (True, 1), (1, 1.0), (None, "null"),
    ([1, 2], [2, 1]), ({"x": False}, {"x": 0}),
])
@pytest.mark.parametrize("field", ["target", "replacement", "content"])
def test_exact_strings_and_json_values_remain_distinct(field, left, right):
    tracker = IdenticalFailureTracker()
    left_args = dict(ARGS, **{field: left})
    right_args = dict(ARGS, **{field: right})
    assert not observe(tracker, args=left_args)
    assert not observe(tracker, args=right_args)
    assert observe(tracker, args=left_args)


def test_canonical_dict_order_and_omitted_false_match_without_mutation():
    tracker = IdenticalFailureTracker()
    first = dict(ARGS, metadata={"b": 2, "a": 1})
    second = {"metadata": {"a": 1, "b": 2}, "replacement": "new\n",
              "target": "old\n", "path": "example.py", "allow_multiple": False}
    assert not observe(tracker, args=first)
    assert observe(tracker, args=second)
    assert "allow_multiple" not in first


@pytest.mark.parametrize("value", [True, 0, None, "false"])
def test_only_boolean_false_is_normalized(value):
    tracker = IdenticalFailureTracker()
    assert not observe(tracker)
    assert not observe(tracker, args=dict(ARGS, allow_multiple=value))
    assert observe(tracker)


def test_other_tools_do_not_normalize_patch_default():
    tracker = IdenticalFailureTracker()
    assert not observe(tracker, name="write_file")
    assert not observe(tracker, name="write_file", args=dict(ARGS, allow_multiple=False))


def test_path_strings_are_not_speculatively_resolved():
    tracker = IdenticalFailureTracker()
    assert not observe(tracker)
    assert not observe(tracker, args=dict(ARGS, path="./example.py"))


@pytest.mark.parametrize("result,label", [
    ("ERROR: Read 'example.py' with read_file before editing it, so the edit is current.", "unread-file"),
    ("ERROR: 'example.py' changed on disk since you last read it. Read it again.", "stale-file"),
    ("ERROR: File not found: example.py", "missing-file"),
    (ERROR, "target-mismatch"),
    ("ERROR: Target content found 2 times in 'example.py'. Provide more context.", "ambiguous-target"),
    ("ERROR: Python syntax validation failed at line 3: invalid syntax. File unchanged.", "syntax-validation"),
    ("ERROR: This overwrite of 'example.py' would delete code. Not approved.", "removal-not-approved"),
    ("ERROR: Security Sandbox Access Denied: private path", "sandbox-access-denied"),
    ("ERROR: Permission denied: private path", "permission-denied"),
])
def test_fixed_error_classes_do_not_echo_payloads(result, label):
    tracker = IdenticalFailureTracker()
    assert not observe(tracker, result=result)
    assert tracker.last_error_class == label
    assert observe(tracker, result=result)
    assert not observe(tracker, result="ERROR: Unrecognized error kind")
    assert tracker.last_error_class == "unclassified-tool-error"


def test_all_main_error_classes_are_distinct():
    tracker = IdenticalFailureTracker()
    results = [ERROR, "ERROR: File not found: example.py",
               "ERROR: Read 'example.py' with read_file before editing it, please.",
               "ERROR: 'example.py' changed on disk since you last read it.",
               "ERROR: Python syntax validation failed at line 1: invalid syntax.",
               "ERROR: This patch to 'example.py' would delete code."]
    labels = set()
    for result in results:
        assert not observe(tracker, result=result)
        labels.add(tracker.last_error_class)
    assert len(labels) == len(results)


def test_supplemental_data_and_hints_do_not_change_error_class():
    tracker = IdenticalFailureTracker()
    assert not observe(tracker, result=ERROR + "\nDATA: unsafe secret text\nHint: do something")
    assert observe(tracker, result=ERROR + "\nDATA: other file text\nHint: something else")
    assert tracker.last_error_class == "target-mismatch"
    assert "unsafe secret" not in repr(tracker._failures)


def test_unrecognized_errors_have_private_distinct_bounded_identities():
    tracker = IdenticalFailureTracker()
    first = "ERROR: unfamiliar-first\nDATA: private data"
    second = "ERROR: unfamiliar-second\nDATA: other data"
    assert not observe(tracker, result=first)
    assert not observe(tracker, result=second)
    assert observe(tracker, result=first)
    assert tracker.last_error_class == "unclassified-tool-error"
    assert "unfamiliar" not in repr(tracker._failures)
    assert "private data" not in repr(tracker._failures)


def test_state_is_bounded_and_evicted_keys_need_two_new_observations():
    tracker = IdenticalFailureTracker(max_entries=3)
    for index in range(20):
        assert not observe(tracker, args=dict(ARGS, target=str(index)))
        assert len(tracker._failures) <= 3
    assert not observe(tracker, args=dict(ARGS, target="0"))
    assert observe(tracker, args=dict(ARGS, target="0"))
    assert len(tracker._failures) == 3
    tracker.begin_turn()
    assert not tracker._failures


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_invalid_capacity_is_rejected(maximum):
    with pytest.raises(ValueError):
        IdenticalFailureTracker(max_entries=maximum)


def test_non_json_arguments_never_produce_repeat():
    tracker = IdenticalFailureTracker()
    assert not observe(tracker, args={"path": "example.py", "value": object()})
    assert not observe(tracker, args={"path": "example.py", "value": float("nan")})
