"""Synthetic, provider-free replay contracts; no runtime or SQLite fixtures."""
import copy
import json
from contextlib import contextmanager

import pytest

from nexus_agent_ai import budget_replay as replay
from nexus_agent_ai.agent import budget_diagnostics, context
from nexus_agent_ai.agent.core import Agent
from nexus_agent_ai.agent.refusals import execution_result
from nexus_agent_ai.agent.result_budget import guard_tool_result


RECORD = {"version": 1, "files": [], "topics": [], "tools": []}


def snapshot_for(messages, *, budget=2000, record=None, adaptive=False,
                 reserve=512, window=None, overhead=11, result_tokens=0, boost=None):
    settings = {
        "context_window": window or budget + reserve + overhead,
        "scale": 1.0, "configured_reserve": reserve, "initial_reserve": reserve,
        "minimum_output": min(reserve, 256), "adaptive": adaptive, "boost": boost,
        "system_tokens": 7, "tool_tokens": overhead - 7,
        "result_tokens": result_tokens, "artifact_store_available": True,
        "recovery_injected": False,
    }
    snapshot = {
        "schema_version": 1, "kind": "planner", "id": 1,
        "stage": "outbound_initial", "row": "seed",
        "session_id": "r11-budget-synthetic", "run_id": "offline-fixture",
        "iteration": 1, "replay_qualified": True, "truncated": False,
        "canonical_messages": copy.deepcopy(messages),
        "projected_messages": copy.deepcopy(messages), "session_record": record,
        "settings": settings, "attempts": [], "previous_success": None,
        "input_hash": replay.stable_hash(messages),
        "transition": {"append_only": False, "common_prefix": 0, "new_messages": [
            {"index": i, "hash": replay.stable_hash(m), "role": m["role"]}
            for i, m in enumerate(messages)
        ]},
    }
    # Fixtures exercise the unchanged fitter directly, not replay's helper.
    current = reserve
    if adaptive and boost:
        current = max(current, min(boost, max(min(reserve, 256),
                      settings["context_window"] - overhead - 256)))
    while True:
        history_budget = settings["context_window"] - current - overhead
        attempt = {"reserve": current, "budget_tokens": history_budget,
                   "session_record": copy.deepcopy(record),
                   "projected_messages": copy.deepcopy(messages),
                   "settings": copy.deepcopy(settings)}
        try:
            fitted, stats = context.fit_messages(
                copy.deepcopy(messages), history_budget - result_tokens,
                session_record=copy.deepcopy(record))
            stats.update(request_window=settings["context_window"], reserve=current,
                         budget_tokens=history_budget, recovery_injected=False)
            stats["artifacts"] = [{k: v for k, v in a.items() if k != "content"}
                                  for a in stats["artifacts"]]
            stats["fitted_messages_hash"] = replay.stable_hash(fitted)
            stats["fitted_tokens"] = stats["tokens"]
            attempt["stats"] = stats
            snapshot["result"] = {"success": True, "stats": stats}
            snapshot["attempts"].append(attempt)
            break
        except context.ContextBudgetError as exc:
            snapshot["attempts"].append(attempt)
            if not adaptive or current <= min(reserve, 256):
                error = {"class": type(exc).__name__, "text": str(exc)}
                attempt["error"] = error
                snapshot["result"] = {"success": False, "error": error}
                break
            current = max(min(reserve, 256), current // 2)
    return snapshot


@pytest.fixture(autouse=True)
def prohibit_runtime_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Replay must not initialize Agent, providers, tools or storage")
    monkeypatch.setattr(Agent, "__init__", forbidden)
    monkeypatch.setattr(Agent, "run", forbidden)
    # All tests use only the planner allocated via __new__; guard storage stays
    # in-memory and canonical history is never a replay destination.
    monkeypatch.setattr(replay._Memory, "_forbidden_write", forbidden)


def test_success_is_token_identical_without_mutation():
    messages = [{"role": "user", "content": "My synthetic name is Zephyr-QX7."}]
    snapshot = snapshot_for(messages)
    before = copy.deepcopy(snapshot)
    report = replay.replay_snapshot(snapshot)
    assert report["status"] == "reproduced"
    assert report["token_identical"] is True
    assert report["attempts_match"] is True
    assert report["hashes"]["captured_fitted"] == report["hashes"]["replayed_fitted"]
    assert report["replayed_result"]["stats"]["tokens"] == context.history_tokens(messages)
    assert report["overhead"] == {"system_tokens": 7, "tool_tokens": 4, "surrogate": True}
    assert report["transition"]["sole_cause_established"] is False
    assert snapshot == before


def test_successful_omission_pair_counts_and_artifacts_are_reported():
    messages = [
        {"role": "user", "content": "Old read"},
        {"role": "assistant", "content": "old payload " * 1000},
        {"role": "user", "content": "Previous"},
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "Active"},
    ]
    snapshot = snapshot_for(messages, budget=600, record=RECORD)
    report = replay.replay_snapshot(snapshot)
    assert report["token_identical"] is True
    stats = report["replayed_result"]["stats"]
    assert stats["compacted_messages"] == 1
    assert stats["omission_tokens"] > 0
    assert stats["turns_before"] == 3
    assert all("content" not in artifact for artifact in stats["artifacts"])
    assert stats["artifacts"][0]["byte_len"] == len(messages[1]["content"].encode())


def test_actual_coverage_gap_reproduces_exact_class_and_text():
    messages = [{"role": "user", "content": "Protected request " * 20}]
    minimum = context.history_tokens(messages) + context.history_tokens(context._omission_pair())
    snapshot = snapshot_for(messages, budget=minimum - 1, record=RECORD)
    assert snapshot["result"]["success"] is False
    assert snapshot["result"]["error"]["text"].startswith("Historical omission coverage gap:")
    report = replay.replay_snapshot(snapshot)
    assert report["status"] == "reproduced"
    assert report["error_match"] is True
    assert report["classification"] == "irreducible"
    assert report["protected_lower_bound"]["tokens"] > report["effective_budget_tokens"]
    assert report["replayed_result"]["error"] == snapshot["result"]["error"]


def test_adaptive_boost_and_minimum_output_failure_sequences():
    snapshot = snapshot_for([{"role": "user", "content": "q" * 1000}],
                            budget=1000, adaptive=True, boost=1024)
    report = replay.replay_snapshot(snapshot)
    assert report["token_identical"]
    assert [a["reserve"] for a in report["attempts"]] == [1024]
    assert report["attempts"][0]["budget_tokens"] == 488
    snapshot = snapshot_for([{"role": "user", "content": "q" * 10000}],
                            reserve=1024, window=1111, adaptive=True)
    report = replay.replay_snapshot(snapshot)
    assert report["error_match"]
    assert [a["reserve"] for a in report["attempts"]] == [1024, 512, 256]


def test_previous_turn_text_contributes_to_irreducible_bound():
    messages = [
        {"role": "user", "content": "Previous"},
        {"role": "assistant", "content": "Protected previous-turn text " * 1000},
        {"role": "user", "content": "Current"},
    ]
    report = replay.replay_snapshot(snapshot_for(messages, budget=100))
    assert report["error_match"]
    assert report["classification"] == "irreducible"
    bound = report["protected_lower_bound"]
    assert bound["message_tokens"] == context.history_tokens(messages)
    assert bound["message_indices"] == [0, 1, 2]
    assert bound["messages"][1]["reason"] == "previous_structure"
    assert bound["messages"][1]["lower_bound_tokens"] == context.message_tokens(messages[1])
    assert bound["messages"][1]["blanked_tool_payloads"] == 0


def test_previous_text_plus_active_and_record_pair_exact_boundary():
    messages = [
        {"role": "user", "content": "Previous user text"},
        {"role": "assistant", "content": "Previous assistant text"},
        {"role": "user", "content": "Active user text"},
    ]
    floor = context.history_tokens(messages) + context.history_tokens(context._omission_pair())
    exact = replay.replay_snapshot(snapshot_for(messages, budget=floor, record=RECORD))
    assert exact["token_identical"]
    assert exact["protected_lower_bound"]["tokens"] == floor
    short = replay.replay_snapshot(snapshot_for(messages, budget=floor - 1, record=RECORD))
    assert short["error_match"]
    assert short["classification"] == "irreducible"
    assert short["protected_lower_bound"]["tokens"] == short["effective_budget_tokens"] + 1


def test_previous_tool_body_is_blank_but_bindings_arguments_and_text_contribute():
    messages = [
        {"role": "user", "content": "Keep previous text"},
        {"role": "assistant", "content": "Keep assistant text", "tool_calls": [
            {"id": "previous-read", "type": "function", "function": {
                "name": "read_file", "arguments": {"path": "synthetic.txt"}}}]},
        {"role": "tool", "tool_call_id": "previous-read", "content": "payload" * 1000},
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "Current"},
    ]
    original = copy.deepcopy(messages)
    minimal = copy.deepcopy(messages)
    minimal[2]["content"] = ""
    bound = replay._protected_bound(messages, RECORD)
    assert bound["message_tokens"] == context.history_tokens(minimal)
    assert bound["tokens"] == context.history_tokens(minimal) + context.history_tokens(context._omission_pair())
    contribution = next(item for item in bound["messages"] if item["index"] == 2)
    assert contribution["blanked_tool_payloads"] == 1
    assert contribution["lower_bound_tokens"] == context.message_tokens(minimal[2])
    assert contribution["input_tokens"] > contribution["lower_bound_tokens"]
    assert bound["messages"][1]["input_hash"] == bound["messages"][1]["lower_bound_hash"]
    assert messages == original


def test_pinned_previous_tool_body_is_not_blank():
    messages = [
        {"role": "user", "content": "Previous"},
        {"role": "tool", "tool_call_id": "pinned", "content": "keep" * 1000,
         "metadata": {"authority": True}},
        {"role": "user", "content": "Current"},
    ]
    bound = replay._protected_bound(messages, None)
    contribution = next(item for item in bound["messages"] if item["index"] == 1)
    assert contribution["reason"] == "pinned"
    assert contribution["blanked_tool_payloads"] == 0
    assert contribution["lower_bound_tokens"] == context.message_tokens(messages[1])


def test_missing_impossibility_evidence_remains_unresolved():
    messages = [
        {"role": "user", "content": "z" * 300},
        {"role": "user", "content": "Previous"},
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "Current"},
    ]
    report = replay.replay_snapshot(snapshot_for(messages, budget=100))
    assert report["error_match"]
    assert report["protected_lower_bound"]["tokens"] <= report["effective_budget_tokens"]
    assert report["classification"] == "unresolved"
    assert report["feasible_witness"] is None


def test_checkpoint_reports_actual_pair_counts_and_projected_arithmetic():
    messages = [{"role": "user", "content": "active"}]
    snapshot = snapshot_for(messages, budget=10, record=RECORD)
    pair = context._omission_pair()
    snapshot["attempts"][0]["checkpoint"] = {
        "starts": [0], "actual_evicted": 0, "actual_closed": 0,
        "actual_pair": pair, "next_pair": None, "needs_record": True,
        "actual_pair_tokens": context.history_tokens(pair),
        "projected_pair_tokens": context.history_tokens(pair),
        "message_tokens_remaining": [context.message_tokens(messages[0])],
        "remaining_tokens": context.history_tokens(messages), "budget_tokens": 10,
    }
    report = replay.replay_snapshot(snapshot)
    accounting = report["checkpoint_accounting"][0]
    assert accounting["arithmetic_match"]
    assert accounting["turns_before"] == 1
    assert accounting["actual_evicted"] == 0
    assert accounting["actual_pair_tokens"] == context.history_tokens(pair)


def test_adaptive_attempts_follow_exact_halving_sequence():
    messages = [{"role": "user", "content": "q" * 1000}]
    # Initial budget 76 cannot fit; 588 after halving does fit.
    snapshot = snapshot_for(messages, adaptive=True, reserve=1024, window=1111)
    report = replay.replay_snapshot(snapshot)
    assert report["token_identical"] is True
    assert [a["reserve"] for a in report["attempts"]] == [1024, 512]
    assert [a["budget_tokens"] for a in report["attempts"]] == [76, 588]
    assert report["attempts"][0]["error"]["class"] == "ContextBudgetError"


def test_changed_adaptive_attempt_or_fitted_hash_is_not_reproduced():
    snapshot = snapshot_for([{"role": "user", "content": "small"}])
    snapshot["attempts"][0]["reserve"] += 1
    assert replay.replay_snapshot(snapshot)["status"] == "mismatch"
    snapshot = snapshot_for([{"role": "user", "content": "small"}])
    snapshot["result"]["stats"]["fitted_messages_hash"] = "0" * 64
    assert replay.replay_snapshot(snapshot)["token_identical"] is False


def test_integer_result_preflight_charge_is_not_lost():
    messages = [{"role": "user", "content": "active"}]
    snapshot = snapshot_for(messages, budget=200, result_tokens=190)
    report = replay.replay_snapshot(snapshot)
    assert report["error_match"] is True
    assert report["effective_budget_tokens"] == 10
    assert report["classification"] == "irreducible"


def receipt_snapshot(*, budget=40, events=(), retention=True):
    messages = [
        {"role": "user", "content": "Read the synthetic file."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "current", "type": "function", "function": {
                "name": "read_file", "arguments": {"path": "synthetic.txt"}}}]},
        {"role": "tool", "tool_call_id": "current",
         "content": "Outcome: read_file completed.\nOutput discarded."},
    ]
    snapshot = snapshot_for(messages, budget=budget, record=RECORD)
    snapshot["kind"] = "receipt"
    snapshot["stage"] = "receipt"
    snapshot["receipt"] = {
        "tool_name": "read_file", "raw_result": "synthetic raw output " * 1000,
        "executed": True, "outcome": "read_file completed.",
        "retention_available": retention, "retention_events": list(events),
        "current_call_id": "current",
    }
    try:
        guard_tool_result("read_file", execution_result("source", executed=True,
                          outcome="read_file completed."), memory=object(), fits=lambda value: False)
    except context.ContextBudgetError as exc:
        snapshot["result"] = {"success": False, "error": {"class": type(exc).__name__, "text": str(exc)}}
    return snapshot


def test_receipt_floor_replays_real_guard_and_preserves_execution_status():
    snapshot = receipt_snapshot()
    before = copy.deepcopy(snapshot)
    report = replay.replay_snapshot(snapshot)
    assert report["error_match"] is True
    assert report["status"] == "reproduced"
    assert report["classification"] == "unresolved"
    assert len(report["receipt_probes"]) == 4  # raw, artifact, both truthful discarded receipts
    assert report["retention_events_consumed"] == []
    assert all(not probe["result"]["success"] for probe in report["receipt_probes"])
    assert snapshot == before


@pytest.mark.parametrize("event", ["success", "full", "error"])
def test_in_memory_retention_stub_follows_events_without_writes(event):
    memory = replay._ReceiptMemory({"retention_events": [event], "retention_available": True})
    plan = {"content": "never persisted", "artifact_id": "identity"}
    if event == "error":
        with pytest.raises(RuntimeError, match="Captured artifact retention error"):
            memory.retain_tool_output(plan)
    else:
        assert memory.retain_tool_output(plan) is (event == "success")
    assert memory.used == [event]
    assert plan == {"content": "never persisted", "artifact_id": "identity"}


def test_missing_or_extra_retention_event_is_explicitly_unreplayable():
    snapshot = receipt_snapshot(events=["success"])
    report = replay.replay_snapshot(snapshot)
    assert report["status"] == "unreplayable"
    assert any("Unconsumed" in item for item in report["unresolved_limitations"])
    memory = replay._ReceiptMemory({"retention_events": [], "retention_available": True})
    with pytest.raises(replay.Unreplayable, match="Missing captured retention"):
        memory.retain_tool_output({})
    assert memory.missing_event


def test_receipt_requires_exact_current_call_binding():
    snapshot = receipt_snapshot()
    snapshot["receipt"]["current_call_id"] = "absent"
    assert replay.replay_snapshot(snapshot)["status"] == "unreplayable"


def test_instrumentation_is_suspended_during_replay(monkeypatch):
    active = []
    original = Agent._plan_fit

    @contextmanager
    def suspended():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def checked(agent, *args, **kwargs):
        assert active == [True]
        return original(agent, *args, **kwargs)

    monkeypatch.setattr(budget_diagnostics, "suspended", suspended)
    monkeypatch.setattr(Agent, "_plan_fit", checked)
    assert replay.replay_snapshot(snapshot_for([{"role": "user", "content": "x"}]))["token_identical"]
    assert active == []


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(schema_version=2),
    lambda s: s.update(truncated=True),
    lambda s: s.update(replay_qualified=False),
    lambda s: s.update(kind="unsupported"),
    lambda s: s.update(input_hash="bad"),
    lambda s: s.pop("projected_messages"),
    lambda s: s["settings"].pop("adaptive"),
    lambda s: s["settings"].update(system_tokens=10**30),
    lambda s: s["attempts"][0].pop("session_record"),
    lambda s: s["attempts"][0].pop("projected_messages"),
    lambda s: s["attempts"][0].pop("settings"),
    lambda s: s.update(session_record={"version": 999}),
    lambda s: s["transition"]["new_messages"][0].update(hash="bad"),
])
def test_incomplete_or_unsupported_snapshots_fail_explicitly(mutate):
    snapshot = snapshot_for([{"role": "user", "content": "x"}])
    mutate(snapshot)
    report = replay.replay_snapshot(snapshot)
    assert report["status"] == "unreplayable"
    assert not report["replay_qualified"]
    assert report["classification"] == "unresolved"
    assert report["unresolved_limitations"]


def test_error_class_text_must_match_and_proximity_is_not_causality():
    snapshot = snapshot_for([{"role": "user", "content": "x" * 1000}], budget=10)
    snapshot["result"]["error"]["text"] += " changed"
    report = replay.replay_snapshot(snapshot)
    assert report["status"] == "mismatch"
    assert not report["error_match"]
    assert report["classification"] == "unresolved"
    assert not report["transition"]["sole_cause_established"]


def test_cli_reads_only_snapshot_and_exclusively_creates_report(tmp_path, capsys):
    path = tmp_path / "snapshot.json"
    snapshot = snapshot_for([{"role": "user", "content": "offline"}])
    original = json.dumps(snapshot)
    path.write_text(original)
    assert replay.main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["token_identical"]
    output = tmp_path / "report.json"
    assert replay.main([str(path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "reproduced"
    assert path.read_text() == original
    with pytest.raises(FileExistsError):
        replay.main([str(path), "--output", str(output)])
    with pytest.raises(SystemExit):
        replay.main([str(path), "--output", str(path)])
    assert path.read_text() == original


def test_cli_malformed_and_truncated_json_is_explicit(tmp_path, capsys):
    path = tmp_path / "truncated.json"
    path.write_text('{"schema_version":1,')
    assert replay.main([str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unreplayable"


def test_cli_oversized_file_is_not_loaded(tmp_path, monkeypatch, capsys):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (replay.MAX_SNAPSHOT_BYTES + 1))
    monkeypatch.setattr(json, "load", lambda *a, **k: pytest.fail("must check bound before loading"))
    assert replay.main([str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unreplayable"
