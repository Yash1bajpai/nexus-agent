"""Offline, synthetic-snapshot replay of the unchanged context budget policy.

This module constructs no providers, tools, databases or Agent runtime. Numeric
instruction/tool overhead is represented by a non-private synthetic string.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .agent import context
from .agent.core import Agent
from .agent.refusals import execution_result
from .agent.result_budget import guard_tool_result

MAX_SNAPSHOT_BYTES = 512 * 1024
MAX_MESSAGES = 4096
MAX_ATTEMPTS = 64
MAX_TOKEN_VALUE = 1_000_000


class Unreplayable(ValueError):
    """The snapshot lacks sufficient exact, supported replay inputs."""


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition, message):
    if not condition:
        raise Unreplayable(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"Invalid {name}")
    return value


def _messages(value, name):
    _require(isinstance(value, list) and len(value) <= MAX_MESSAGES, f"Invalid {name}")
    _require(all(isinstance(m, dict) and isinstance(m.get("role"), str)
                 and isinstance(m.get("metadata", {}), dict) for m in value),
             f"Invalid {name} message")
    return value


def _error(exc):
    return {"class": type(exc).__name__, "text": str(exc)}


def _validate_error(value):
    _require(isinstance(value, dict) and isinstance(value.get("class"), str)
             and isinstance(value.get("text"), str), "Missing exact captured error")


def _settings(value):
    _require(isinstance(value, dict), "Missing planner settings")
    required = {"context_window", "scale", "configured_reserve", "initial_reserve",
                "minimum_output", "adaptive", "boost", "system_tokens", "tool_tokens",
                "result_tokens", "artifact_store_available", "recovery_injected"}
    _require(required <= value.keys(), "Incomplete planner settings")
    for key in ("context_window", "configured_reserve", "initial_reserve"):
        _require(_integer(value[key], key, 1) <= MAX_TOKEN_VALUE, f"Excessive {key}")
    for key in ("minimum_output", "system_tokens", "tool_tokens", "result_tokens"):
        _require(_integer(value[key], key) <= MAX_TOKEN_VALUE, f"Excessive {key}")
    _require(value["system_tokens"] >= 1, "System overhead cannot be represented exactly")
    _require(type(value["scale"]) in (int, float) and math.isfinite(value["scale"])
             and value["scale"] > 0, "Invalid scale")
    _require(0 < int(value["context_window"] * value["scale"]) <= MAX_TOKEN_VALUE,
             "Unsupported request window")
    for key in ("adaptive", "artifact_store_available", "recovery_injected"):
        _require(type(value[key]) is bool, f"Invalid {key}")
    if value["boost"] is not None:
        _require(_integer(value["boost"], "boost", 1) <= MAX_TOKEN_VALUE, "Excessive boost")
    _require(value["minimum_output"] == min(value["configured_reserve"], 256),
             "Unsupported minimum output policy")
    return value


def _record(value):
    _require(value is None or context._validated_session_record(value) == value,
             "Invalid session record")
    return value


def _validate(snapshot):
    _require(isinstance(snapshot, dict), "Snapshot must be a JSON object")
    _require(snapshot.get("schema_version") == 1 and type(snapshot.get("schema_version")) is int,
             "Unsupported snapshot schema version")
    _require(snapshot.get("kind") in ("planner", "receipt"), "Unsupported snapshot kind")
    _require(snapshot.get("truncated") is False, "Truncated snapshot is not replayable")
    _require(snapshot.get("replay_qualified") is True, "Snapshot is not replay-qualified")
    _require(snapshot.get("session_id") == "r11-budget-synthetic", "Not a synthetic snapshot")
    _require(snapshot.get("row") in {"seed", *(f"filler-{i}" for i in range(1, 7))},
             "Unknown synthetic row")
    _require(isinstance(snapshot.get("run_id"), str) and bool(snapshot["run_id"]), "Missing run id")
    _integer(snapshot.get("id"), "snapshot id", 1)
    _require(isinstance(snapshot.get("stage"), str), "Missing stage")
    _require("iteration" in snapshot and "previous_success" in snapshot and "transition" in snapshot,
             "Incomplete snapshot provenance")
    _messages(snapshot.get("canonical_messages"), "canonical_messages")
    _messages(snapshot.get("projected_messages"), "projected_messages")
    _require("session_record" in snapshot, "Missing session record")
    _record(snapshot["session_record"])
    _settings(snapshot.get("settings"))
    _require(snapshot.get("input_hash") == stable_hash(snapshot["canonical_messages"]),
             "Canonical input hash mismatch")
    attempts = snapshot.get("attempts")
    _require(isinstance(attempts, list) and 0 < len(attempts) <= MAX_ATTEMPTS,
             "Missing or excessive planner attempts")
    for attempt in attempts:
        _require(isinstance(attempt, dict), "Invalid planner attempt")
        _integer(attempt.get("reserve"), "attempt reserve", 1)
        _require(type(attempt.get("budget_tokens")) is int, "Invalid attempt budget")
        _require("session_record" in attempt, "Missing attempt session record")
        _record(attempt["session_record"])
        _messages(attempt.get("projected_messages"), "attempt projected_messages")
        _settings(attempt.get("settings"))
        _require(attempt["projected_messages"] == snapshot["projected_messages"]
                 and attempt["settings"] == snapshot["settings"]
                 and attempt["session_record"] == snapshot["session_record"],
                 "Attempt inputs differ from plan inputs")
    result = snapshot.get("result")
    _require(isinstance(result, dict) and type(result.get("success")) is bool,
             "Missing captured result")
    if not result["success"]:
        _validate_error(result.get("error"))
    else:
        stats = result.get("stats")
        _require(isinstance(stats, dict) and isinstance(stats.get("fitted_messages_hash"), str)
                 and type(stats.get("tokens")) is int, "Missing fitted hash or token total")
    if snapshot["kind"] == "receipt":
        receipt = snapshot.get("receipt")
        _require(isinstance(receipt, dict), "Missing receipt inputs")
        for key in ("tool_name", "current_call_id"):
            _require(isinstance(receipt.get(key), str) and bool(receipt[key]), f"Invalid receipt {key}")
        _require(isinstance(receipt.get("raw_result"), str), "Invalid receipt raw_result")
        _require("executed" in receipt and (receipt["executed"] is None
                 or type(receipt["executed"]) is bool), "Missing receipt execution status")
        _require("outcome" in receipt and type(receipt.get("retention_available")) is bool,
                 "Missing receipt retention/outcome evidence")
        events = receipt.get("retention_events")
        _require(isinstance(events, list) and len(events) <= MAX_ATTEMPTS
                 and all(event in ("success", "full", "error") for event in events),
                 "Unsupported retention events")
        _require(receipt["retention_available"] or not events,
                 "Unavailable retention has captured retention events")
    return snapshot


class _Memory:
    def __init__(self, record, artifact_store_available):
        self.record = copy.deepcopy(record)
        if artifact_store_available:
            # _plan_fit checks availability but never invokes this method.
            self.store_artifacts = self._forbidden_write

    @staticmethod
    def _forbidden_write(*args, **kwargs):
        raise AssertionError("Offline replay must never store artifacts")

    def get_session_metadata(self, key):
        _require(key == "SESSION_RECORD", "Unexpected planner memory access")
        return copy.deepcopy(self.record)


def _agent(settings, record):
    agent = Agent.__new__(Agent)
    agent.context_window = settings["context_window"]
    agent.output_reserve = settings["configured_reserve"]
    agent._edit_recovery = None
    agent.provider = SimpleNamespace(is_local=settings["adaptive"],
                                     _supports_output_budget=settings["adaptive"])
    overhead = settings["system_tokens"] + settings["tool_tokens"]
    agent.system = "s" * math.ceil((overhead - 1) * context.CHARS_PER_TOKEN)
    _require(context.estimate_tokens(agent.system) == overhead, "Inexact overhead surrogate")
    agent.memory = _Memory(record, settings["artifact_store_available"])
    return agent


def _clean_view(messages):
    view = context.fitted_claim_view(copy.deepcopy(messages))
    clean = copy.deepcopy(view)
    for message in clean:
        if not context.is_genuine_user_message(message):
            metadata = {k: v for k, v in message.get("metadata", {}).items()
                        if k not in ("mutation_outcomes", "turn", "segment")}
            if metadata:
                message["metadata"] = metadata
            else:
                message.pop("metadata", None)
    return view, clean


def _protected_bound(messages, record):
    """Bound immutable messages and N-1 text/structure independently of fitting.

    Only ordinary named tool-result payloads in N-1 may be replaced. Emptying
    those slots underestimates any legal replacement's serialized cost while
    retaining user/assistant text, call arguments and result bindings. Whole
    pinned and active messages remain exact; older unpinned messages contribute
    nothing. No captured checkpoint or greedy fitter remainder supplies proof.
    """
    view, clean = _clean_view(messages)
    starts = [i for i, m in enumerate(view) if context.is_genuine_user_message(m)]
    active = starts[-1] if starts else 0
    previous = starts[-2] if len(starts) >= 2 else active
    pinned = context._pinned_read_messages(view)
    pinned.update(i for i, m in enumerate(view) if context._pinned_evidence(m))
    contributions, calls = [], {}
    for index, message in enumerate(clean):
        for call_id, name, arguments in context._tool_calls(message):
            calls[call_id] = (name or "unknown_tool", context._call_paths(arguments))
        reason = ("pinned" if index in pinned else
                  "active" if index >= active else
                  "previous_structure" if index >= previous else None)
        if reason is None:
            continue
        minimal = copy.deepcopy(message)
        omitted = 0
        if reason == "previous_structure":
            for owner, key, name in context._payloads(minimal, calls):
                if name is not None:
                    owner[key] = ""
                    omitted += 1
        contributions.append({
            "index": index, "reason": reason,
            "input_tokens": context.message_tokens(message),
            "lower_bound_tokens": context.message_tokens(minimal),
            "blanked_tool_payloads": omitted,
            "input_hash": stable_hash(message), "lower_bound_hash": stable_hash(minimal),
        })
    message_total = sum(item["lower_bound_tokens"] for item in contributions)
    pair_floor = context.history_tokens(context._omission_pair()) if record is not None else 0
    return {"message_indices": [item["index"] for item in contributions],
            "messages": contributions, "message_tokens": message_total,
            "mandatory_record_pair_tokens": pair_floor,
            "tokens": message_total + pair_floor}


def _witness(messages, budget, record):
    """Look only for exact, unmodified, legal-prefix witnesses, not heuristics.

    No rewriting, compaction, changed protections, omission wording or skipped
    incomplete/pinned prefix is permitted. Reservation headroom is also charged.
    Failure to find this restricted witness does not prove infeasibility.
    """
    view, clean = _clean_view(messages)
    starts = [i for i, m in enumerate(view) if context.is_genuine_user_message(m)]
    pinned = context._pinned_read_messages(view)
    pinned.update(i for i, m in enumerate(view) if context._pinned_evidence(m))
    active = starts[-1] if starts else 0
    cut, closed = 0, 0
    for evicted in range(max(1, len(starts) - 1)):
        needs_record = record is not None or evicted > 0
        total = context.history_tokens(clean[cut:])
        floor = context._omission_floor(view, starts, pinned, needs_record,
                                       evicted_turns=evicted, evicted_closed_turns=closed)
        if total + floor <= budget:
            try:
                if needs_record:
                    fields = context._collect_omission_fields(view[:active], record)
                    pair, _ = context._fit_omission_fields(
                        fields, budget - total, evicted_turns=evicted,
                        turns_before=len(starts), evicted_closed_turns=closed)
                else:
                    pair = []
                fitted = pair + clean[cut:]
                if context.history_tokens(fitted) <= budget:
                    return {"cut": cut, "evicted_turns": evicted,
                            "evicted_closed_turns": closed, "tokens": context.history_tokens(fitted),
                            "omission_tokens": context.history_tokens(pair),
                            "reserved_pair_floor": floor, "fitted_messages_hash": stable_hash(fitted),
                            "construction": "unchanged messages after eligible completed prefix; existing omission builder"}
            except context.ContextBudgetError:
                pass
        if evicted >= len(starts) - 2:
            break
        start, end = starts[evicted], starts[evicted + 1]
        if (start != cut or any(i in pinned for i in range(start, end))
                or not context._completed_turn(view[start:end])):
            break
        cut = end
        closed += int(context._executor_closed_turn(view[start:end]))
    return None


def _safe_stats(stats, fitted, recovery_injected=False):
    safe = copy.deepcopy(stats)
    for artifact in safe.get("artifacts", []):
        artifact.pop("content", None)
    safe["fitted_messages_hash"] = stable_hash(fitted)
    safe["fitted_tokens"] = context.history_tokens(fitted)
    if "recovery_injected" in safe:
        safe["recovery_injected"] = recovery_injected
    return safe


def _plan(snapshot, messages=None):
    settings = snapshot["settings"]
    messages = copy.deepcopy(snapshot["projected_messages"] if messages is None else messages)
    record = snapshot["session_record"]
    window = int(settings["context_window"] * settings["scale"])
    overhead = settings["system_tokens"] + settings["tool_tokens"]
    reserve = settings["initial_reserve"]
    if settings["adaptive"] and settings["boost"]:
        reserve = max(reserve, min(settings["boost"],
                      max(settings["minimum_output"], window - overhead - 256)))
    attempts = []
    while True:
        budget = window - reserve - overhead
        attempt = {"reserve": reserve, "budget_tokens": budget,
                   "effective_budget_tokens": budget - settings["result_tokens"]}
        try:
            if budget <= 0:
                raise context.ContextBudgetError(
                    f"Configured context budget ({window} tokens) cannot fit instructions, tools and answer reserve "
                    f"({overhead + reserve} tokens). Adjust the configured context/output budgets within "
                    "the endpoint's documented capacity, or use fewer tools.")
            fitted, stats = context.fit_messages(messages, budget - settings["result_tokens"],
                                                 session_record=copy.deepcopy(record))
            if stats.get("artifacts") and not settings["artifact_store_available"]:
                raise context.ContextBudgetError(
                    "Surgical context compaction needs the existing SQLite history artifact store; "
                    "no unretained artifact references can be sent.")
            attempt["stats"] = _safe_stats(stats, fitted)
        except context.ContextBudgetError as exc:
            attempt["error"] = _error(exc)
        attempts.append(attempt)
        if "error" not in attempt or not settings["adaptive"] or reserve <= settings["minimum_output"]:
            break
        reserve = max(settings["minimum_output"], reserve // 2)
        _require(len(attempts) < MAX_ATTEMPTS, "Excessive adaptive attempt sequence")
    # Execute the actual planner as well as checking each arithmetic attempt.
    agent = _agent(settings, record)
    try:
        fitted, stats = agent._plan_fit(messages, [], settings["scale"],
                                       output_reserve=settings["initial_reserve"],
                                       boost=settings["boost"], result_tokens=settings["result_tokens"])
        _require("error" not in attempts[-1], "Planner and fitter attempt outcomes disagree")
        result = {"success": True,
                  "stats": _safe_stats(stats, fitted, settings["recovery_injected"])}
    except context.ContextBudgetError as exc:
        _require(attempts[-1].get("error") == _error(exc), "Planner and fitter errors disagree")
        result = {"success": False, "error": _error(exc)}
    return result, attempts


def _attempt_matches(captured, actual):
    if len(captured) != len(actual):
        return False
    for expected, replayed in zip(captured, actual):
        if any(expected[key] != replayed[key] for key in ("reserve", "budget_tokens")):
            return False
        if "error" in expected and not expected["error"].get("inferred", False):
            if any(expected["error"].get(key) != replayed.get("error", {}).get(key)
                   for key in ("class", "text")):
                return False
        if "stats" in expected:
            for key in ("tokens", "fitted_messages_hash"):
                if key in expected["stats"] and expected["stats"][key] != replayed.get("stats", {}).get(key):
                    return False
    return True


class _ReceiptMemory:
    def __init__(self, receipt):
        self.events = list(receipt["retention_events"])
        self.used = []
        self.missing_event = False
        if receipt["retention_available"]:
            self.retain_tool_output = self._retain

    def _retain(self, plan):
        if len(self.used) >= len(self.events):
            self.missing_event = True
            raise Unreplayable("Missing captured retention event")
        event = self.events[len(self.used)]
        self.used.append(event)
        if event == "error":
            raise RuntimeError("Captured artifact retention error (in-memory replay)")
        return event == "success"


def _receipt(snapshot):
    receipt = snapshot["receipt"]
    template = copy.deepcopy(snapshot["projected_messages"])
    matches = [i for i, m in enumerate(template) if m.get("role") == "tool"
               and m.get("tool_call_id") == receipt["current_call_id"]]
    _require(len(matches) == 1, "Receipt template does not identify exactly one current tool result")
    index = matches[0]
    memory = _ReceiptMemory(receipt)
    probes = []

    def fits(content):
        candidate = copy.deepcopy(template)
        candidate[index]["content"] = str(content)
        result, attempts = _plan(snapshot, candidate)
        probes.append({"content_hash": stable_hash(str(content)),
                       "content_tokens": context.estimate_tokens(str(content)),
                       "result": result, "attempts": attempts})
        return result["success"]

    source = execution_result(receipt["raw_result"], executed=receipt["executed"],
                              outcome=receipt["outcome"])
    source.executed = receipt["executed"]
    try:
        fitted = guard_tool_result(receipt["tool_name"], source, memory=memory, fits=fits)
        result = {"success": True, "content_hash": stable_hash(str(fitted)),
                  "content_tokens": context.estimate_tokens(str(fitted))}
    except context.ContextBudgetError as exc:
        result = {"success": False, "error": _error(exc)}
    _require(not memory.missing_event, "Missing captured retention event")
    _require(memory.used == receipt["retention_events"], "Unconsumed captured retention events")
    return result, probes, memory.used


def _transition(snapshot):
    transition = snapshot["transition"]
    _require(isinstance(transition, dict) and type(transition.get("append_only")) is bool
             and type(transition.get("common_prefix")) is int
             and isinstance(transition.get("new_messages"), list), "Missing transition evidence")
    messages = snapshot["canonical_messages"]
    _require(0 <= transition["common_prefix"] <= len(messages), "Invalid transition prefix")
    rows = []
    for item in transition["new_messages"]:
        _require(isinstance(item, dict), "Invalid transition message")
        index = _integer(item.get("index"), "transition index")
        _require(index < len(messages), "Transition index outside canonical input")
        message = messages[index]
        _require(item.get("hash") == stable_hash(message) and item.get("role") == message["role"],
                 "Transition message hash or role mismatch")
        rows.append({**item, "tokens": context.message_tokens(message)})
    _require([item["index"] for item in rows]
             == list(range(transition["common_prefix"], len(messages))),
             "Transition suffix is incomplete or duplicated")
    return {"append_only": transition["append_only"], "common_prefix": transition["common_prefix"],
            "previous_success": snapshot["previous_success"], "new_messages": rows,
            "new_message_tokens": sum(item["tokens"] for item in rows),
            "sole_cause_established": False}


def _checkpoint_accounting(checkpoint):
    if checkpoint is None:
        return {"available": False}
    _require(isinstance(checkpoint, dict), "Invalid checkpoint evidence")
    keys = {"starts", "actual_evicted", "actual_closed", "actual_pair", "next_pair",
            "actual_pair_tokens", "projected_pair_tokens", "needs_record",
            "message_tokens_remaining", "remaining_tokens", "budget_tokens"}
    if not keys <= checkpoint.keys():
        return {"available": False, "limitation": "Incomplete captured checkpoint"}
    starts = checkpoint["starts"]
    _require(isinstance(starts, list), "Invalid checkpoint turn starts")
    evicted = _integer(checkpoint["actual_evicted"], "checkpoint eviction count")
    closed = _integer(checkpoint["actual_closed"], "checkpoint closed count")
    _require(evicted <= len(starts) and closed <= evicted, "Invalid checkpoint counts")
    expected_pair = (context._omission_pair(
        evicted_turns=evicted, turns_before=len(starts), evicted_closed_turns=closed,
    ) if checkpoint["needs_record"] else [])
    actual_pair = _messages(checkpoint["actual_pair"], "checkpoint actual pair")
    next_pair = checkpoint["next_pair"]
    if next_pair is not None:
        _messages(next_pair, "checkpoint projected pair")
    tokens = checkpoint["message_tokens_remaining"]
    _require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens),
             "Invalid checkpoint remaining token list")
    actual_cost = context.history_tokens(actual_pair)
    projected_cost = max(actual_cost, context.history_tokens(next_pair or []))
    arithmetic_match = (actual_pair == expected_pair
                        and actual_cost == checkpoint["actual_pair_tokens"]
                        and projected_cost == checkpoint["projected_pair_tokens"]
                        and sum(tokens) == checkpoint["remaining_tokens"])
    return {"available": True, "arithmetic_match": arithmetic_match,
            "turns_before": len(starts), "actual_evicted": evicted, "actual_closed": closed,
            "actual_pair_tokens": actual_cost, "projected_pair_tokens": projected_cost,
            "actual_pair_hash": stable_hash(actual_pair),
            "projected_pair_hash": stable_hash(next_pair) if next_pair is not None else None,
            "remaining_tokens": sum(tokens), "effective_budget_tokens": checkpoint["budget_tokens"],
            "prefix_eligibility": checkpoint.get("prefix_eligibility"),
            "limitation": "Arithmetic and actual pair wording checked; surgical intermediate state is captured evidence."}


def replay_snapshot(snapshot: dict) -> dict:
    """Return a JSON-safe report; absent evidence is explicitly unreplayable.

    This callable never writes, loads credentials or executes a model/tool. It
    suspends diagnostic hooks, including when invoked inside an active capture.
    """
    report = {"status": "unreplayable", "reproduced": False, "replay_qualified": False,
              "kind": snapshot.get("kind") if isinstance(snapshot, dict) else None,
              "classification": "unresolved", "token_identical": False,
              "error_match": False, "unresolved_limitations": []}
    try:
        _require(len(json.dumps(snapshot, ensure_ascii=False, allow_nan=False,
                                separators=(",", ":")).encode("utf-8"))
                 <= MAX_SNAPSHOT_BYTES, "Snapshot exceeds replay byte bound")
        _validate(snapshot)
        # Imported lazily to keep the replay import independent of activation.
        from .agent.budget_diagnostics import suspended
        with suspended():
            transition = _transition(snapshot)
            planner, attempts = _plan(snapshot)
            attempt_match = _attempt_matches(snapshot["attempts"], attempts)
            probes, retention = [], []
            if snapshot["kind"] == "receipt":
                actual, probes, retention = _receipt(snapshot)
            else:
                actual = planner
            expected = snapshot["result"]
            error_match = (not expected["success"] and not actual["success"]
                           and all(expected["error"][k] == actual["error"][k]
                                   for k in ("class", "text")))
            accounting_match = (not expected["success"] or not actual["success"]
                                or all(actual["stats"].get(key) == value
                                       for key, value in expected["stats"].items()))
            token_identical = (snapshot["kind"] == "planner" and expected["success"]
                               and actual["success"] and attempt_match and accounting_match
                               and expected["stats"]["fitted_messages_hash"] == actual["stats"]["fitted_messages_hash"]
                               and expected["stats"]["tokens"] == actual["stats"]["tokens"])
            reproduced = attempt_match and (error_match or token_identical)
            bound = _protected_bound(snapshot["projected_messages"], snapshot["session_record"])
            budget = attempts[-1]["effective_budget_tokens"]
            witness = None
            classification = "other_path" if actual["success"] else "unresolved"
            if reproduced and not actual["success"] and snapshot["kind"] == "planner":
                if bound["tokens"] > budget:
                    classification = "irreducible"
                elif actual["error"]["text"].startswith("Historical omission coverage gap:"):
                    witness = _witness(snapshot["projected_messages"], budget, snapshot["session_record"])
                    if witness is not None:
                        classification = "reservation_defect"
            limitations = ["Synthetic numeric system/tool overhead surrogate; no private instructions or schemas replayed.",
                           "Post-projection input replay does not validate the original recovery projection.",
                           "Token counts are the unchanged estimator, not model tokenizer counts.",
                           "New-message contributions establish neither sole cause nor model readiness."]
            if classification == "unresolved":
                limitations.append("No rigorous impossibility bound or legal feasible reservation witness establishes a cause.")
            if snapshot["kind"] == "receipt":
                limitations.append("Receipt probes replay exact captured template and retention events; guard-level cause remains unresolved.")
            if snapshot["previous_success"] is None:
                limitations.append("No linked preceding successful actual outbound fit is available in this snapshot.")
            report.update(status="reproduced" if reproduced else "mismatch", reproduced=reproduced,
                          replay_qualified=True, id=snapshot["id"], captured_success=expected["success"],
                          classification=classification, token_identical=token_identical,
                          error_match=error_match, attempts_match=attempt_match,
                          accounting_match=accounting_match,
                          checkpoint_accounting=[_checkpoint_accounting(a.get("checkpoint"))
                                                 for a in snapshot["attempts"]],
                          hashes={"canonical_input": stable_hash(snapshot["canonical_messages"]),
                                  "projected_input": stable_hash(snapshot["projected_messages"]),
                                  "captured_fitted": expected.get("stats", {}).get("fitted_messages_hash"),
                                  "replayed_fitted": actual.get("stats", {}).get("fitted_messages_hash")},
                          input_tokens={"canonical": context.history_tokens(snapshot["canonical_messages"]),
                                        "projected": context.history_tokens(snapshot["projected_messages"])},
                          overhead={"system_tokens": snapshot["settings"]["system_tokens"],
                                    "tool_tokens": snapshot["settings"]["tool_tokens"], "surrogate": True},
                          captured_result=expected, replayed_result=actual, attempts=attempts,
                          checkpoints=[a.get("checkpoint") for a in snapshot["attempts"]],
                          checkpoint_evidence="captured intermediate state; pair wording and count arithmetic checked",
                          protected_lower_bound=bound, effective_budget_tokens=budget,
                          feasible_witness=witness, transition=transition,
                          receipt_probes=probes, retention_events_consumed=retention,
                          unresolved_limitations=limitations)
    except (Unreplayable, ValueError, TypeError, KeyError, OverflowError, RecursionError, AttributeError) as exc:
        report.update(status="unreplayable", reproduced=False, replay_qualified=False,
                      token_identical=False, error_match=False, classification="unresolved")
        report["unresolved_limitations"].append(f"Unreplayable: {exc}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        _require(args.snapshot.stat().st_size <= MAX_SNAPSHOT_BYTES, "Snapshot exceeds replay byte bound")
        with args.snapshot.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        report = replay_snapshot(snapshot)
    except (OSError, ValueError, TypeError) as exc:
        report = {"status": "unreplayable", "reproduced": False, "replay_qualified": False,
                  "classification": "unresolved", "token_identical": False, "error_match": False,
                  "unresolved_limitations": [f"Unreplayable: {exc}"]}
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        if args.output.resolve() == args.snapshot.resolve():
            parser.error("--output must not overwrite the snapshot")
        # Exclusive creation avoids modifying existing canonical/artifact files.
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    else:
        print(encoded, end="")
    return 0 if report["status"] == "reproduced" else 2


if __name__ == "__main__":
    raise SystemExit(main())
