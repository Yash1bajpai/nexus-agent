"""Explicitly scoped, bounded observations for the R11 synthetic replay only.

Nothing reads history, inspects a provider, or writes files without an explicit
validated scope and the exact synthetic turn. This module never enables itself
from environment variables. Its decorators do not alter planner decisions.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import traceback
import uuid

SESSION_ID = "r11-budget-synthetic"
ROWS = ("seed", *(f"filler-{i}" for i in range(1, 7)))
MAX_CAPTURE_BYTES = 512 * 1024
_SCOPE = ContextVar("r11_scope", default=None)
_TURN = ContextVar("r11_turn", default=None)
_PLAN = ContextVar("r11_plan", default=None)
_STAGE = ContextVar("r11_stage", default=None)
_RECEIPT = ContextVar("r11_receipt", default=None)


def frozen_argv(row):
    if row not in ROWS:
        raise ValueError("Unknown synthetic row")
    prompt = (
        "My name is Zephyr-QX7. Please read the file nexus_cfg_v9.yaml with your read_file tool and tell me its owner."
        if row == "seed" else
        f"Read evict_filler.txt with your read_file tool and count how many times the word alpha appears in section {int(row[-1])}."
    )
    return ["chat", "-p", "local", "--no-stream", "--persist", "--session", SESSION_ID, prompt]


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def stable_hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


class _Oversize(ValueError):
    pass


def _copy(value, limit):
    # iterencode prevents unbounded aggregate buffers for large message arrays.
    chunks, used = [], 0
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":"))
    for chunk in encoder.iterencode(value):
        used += len(chunk.encode("utf-8"))
        if used > limit:
            raise _Oversize("snapshot exceeds per-file capture allowance")
        chunks.append(chunk)
    return json.loads("".join(chunks))


def _atomic(path, value):
    payload = _encoded(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".r11-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_scope(root: Path) -> dict:
    """Create an exclusively new isolated run; never reuse an existing root."""
    root = Path(root).absolute()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    root = root.resolve()
    state = root / "state"
    home, workspace, output = state / "home", state / "workspace", root / "output"
    home.mkdir(mode=0o700, parents=True)
    workspace.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    scope = dict(schema_version=1, session_id=SESSION_ID, run_id=str(uuid.uuid4()),
                 root=str(root), state=str(state), home=str(home), workspace=str(workspace),
                 output=str(output), db_path=str(home / ".nexus-agent" / "history.db"),
                 scope_path=str(root / "scope.json"), max_capture_bytes=MAX_CAPTURE_BYTES)
    _atomic(root / "scope.json", scope)
    return scope


def _valid_scope(scope, scope_path=None):
    try:
        root = Path(scope["root"])
        if not root.is_absolute() or root.resolve() != root:
            return False
        run = uuid.UUID(scope["run_id"])
        if str(run) != scope["run_id"] or run.version != 4:
            return False
        expected = {"state": root / "state", "home": root / "state/home",
                    "workspace": root / "state/workspace", "output": root / "output",
                    "db_path": root / "state/home/.nexus-agent/history.db",
                    "scope_path": root / "scope.json"}
        for key, path in expected.items():
            if scope.get(key) != str(path) or path.resolve() != path:
                return False
        if scope_path is not None and Path(scope_path).absolute() != expected["scope_path"]:
            return False
        if scope.get("schema_version") != 1 or scope.get("session_id") != SESSION_ID:
            return False
        cap = scope.get("max_capture_bytes")
        if type(cap) is not int or not 8192 <= cap <= MAX_CAPTURE_BYTES:
            return False
        if any(not Path(scope[key]).is_dir() for key in ("root", "state", "home", "workspace", "output")):
            return False
        return (os.environ.get("HOME") == scope["home"]
                and os.environ.get("USERPROFILE") == scope["home"]
                and str(Path.cwd()) == scope["workspace"])
    except Exception:
        return False


@contextmanager
def activate_scope(scope_path: Path, row: str):
    """Invalid/mismatched scopes are a no-op, not an alternate launch mode."""
    scope = None
    try:
        path = Path(scope_path)
        if row in ROWS and path.stat().st_size <= 8192:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if _valid_scope(candidate, path):
                scope = {**candidate, "row": row}
    except Exception:
        pass
    token = _SCOPE.set(scope)
    turn_token = _TURN.set(None)
    try:
        yield scope
    finally:
        _TURN.reset(turn_token)
        _SCOPE.reset(token)

@contextmanager
def suspended():
    tokens = [(variable, variable.set(None)) for variable in
              (_SCOPE, _TURN, _PLAN, _STAGE, _RECEIPT)]
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def _qualified(agent, scope):
    try:
        if not _valid_scope(scope):
            return False
        memory = agent.memory
        return (memory.session_id == SESSION_ID
                and str(Path(memory.db_path).absolute()) == scope["db_path"])
    except Exception:
        return False


def _turn(agent=None):
    turn = _TURN.get()
    if turn is None or (agent is not None and turn["agent"] is not agent):
        return None
    if not _qualified(turn["agent"], turn["scope"]):
        return None
    return turn


def planner_active():
    return _PLAN.get() is not None and _turn() is not None


def _limit(turn):
    # Two snapshots plus bounded manifest fit the hard per-invocation allowance.
    return (turn["scope"]["max_capture_bytes"] - 4096) // 2


def _error(exc):
    return {"class": type(exc).__name__, "text": str(exc),
            "traceback": "".join(traceback.format_tb(exc.__traceback__))}


def _mark(plan, reason):
    plan["replay_qualified"] = False
    plan["truncated"] = True
    plan["reason"] = reason
    limitations = plan.setdefault("limitations", [])
    if reason not in limitations and len(limitations) < 8:
        limitations.append(reason)


def _base(turn, kind="planner"):
    turn["sequence"] += 1
    scope = turn["scope"]
    return dict(schema_version=1, kind=kind, id=turn["sequence"], stage="unscoped_plan",
                row=scope["row"], session_id=SESSION_ID, run_id=scope["run_id"],
                iteration=turn["iteration"], replay_qualified=True, truncated=False,
                canonical_messages=None, projected_messages=None, session_record=None,
                pending_calls=None, settings={}, attempts=[], result={}, input_hash=None,
                previous_success=turn["success_ref"], transition=None)

def _capture_pending(turn, plan):
    try:
        pending = turn["agent"]._pending_calls
        if not isinstance(pending, list) or len(pending) > 256:
            raise ValueError("pending call list missing or exceeds capture allowance")
        entries, incomplete = [], False
        for item in pending:
            call = item["call"]
            name, identity, state = getattr(call, "name", None), getattr(call, "id", None), item.get("state")
            if not isinstance(name, str) or not name:
                name, incomplete = None, True
            if not isinstance(identity, str) or not identity:
                identity, incomplete = None, True
            if state not in ("pending", "started", "returned", "recorded"):
                state, incomplete = None, True
            result = item.get("result")
            executed = getattr(result, "executed", None) if isinstance(result, str) else None
            entries.append({"name": name, "id": identity, "state": state,
                            "executed": executed if type(executed) is bool else None})
        plan["pending_calls"] = _copy(entries, _limit(turn))
        if incomplete:
            _mark(plan, "pending call name, identity or state is unavailable")
    except Exception as exc:
        _mark(plan, f"pending call state unavailable: {type(exc).__name__}")



def _transition(turn, messages):
    previous = turn.get("success_messages")
    common = 0
    if previous is not None:
        for left, right in zip(previous, messages):
            if left != right:
                break
            common += 1
    return {"append_only": previous is not None and common == len(previous),
            "common_prefix": common,
            "new_messages": [{"index": index, "hash": stable_hash(message),
                              "role": message.get("role")} for index, message in
                             enumerate(messages[common:], common)]}


def _snapshot(turn, plan):
    try:
        return _copy(plan, _limit(turn))
    except Exception as exc:
        return {key: plan.get(key) for key in
                ("schema_version", "kind", "id", "stage", "row", "session_id", "run_id", "iteration", "input_hash", "previous_success")} | {
                    "replay_qualified": False, "truncated": True,
                    "reason": plan.get("reason", f"snapshot unavailable: {type(exc).__name__}"),
                    "limitations": plan.get("limitations") or [f"snapshot unavailable: {type(exc).__name__}"],
                    "result": {"success": bool(plan.get("result", {}).get("success"))}}


def _write_capture(turn, plan, failure=False):
    scope = turn["scope"]
    if scope.get("_fatal_written") or not _qualified(turn["agent"], scope):
        return
    data = _snapshot(turn, plan)
    filename = "failure.json" if failure else "latest_success.json"
    directory = Path(scope["output"]) / scope["row"]
    if directory.resolve() != directory:
        return
    _atomic(directory / filename, data)
    reference = {"id": data["id"], "file": filename, "input_hash": data.get("input_hash"),
                 "replay_qualified": data["replay_qualified"]}
    if failure:
        scope["_fatal_written"] = True
        reference.update(kind=data["kind"], previous_success=data.get("previous_success"))
        turn["failure_ref"] = reference
    else:
        turn["success_ref"] = {key: reference[key] for key in ("id", "file", "input_hash")}
        turn["success_manifest"] = reference
        turn["success_messages"] = data.get("canonical_messages")
    manifest = dict(schema_version=1, row=scope["row"], session_id=SESSION_ID,
                    run_id=scope["run_id"], replay_qualified=bool(turn.get("failure_ref", {}).get("replay_qualified")),
                    latest_success=turn.get("success_manifest"), failure=turn.get("failure_ref"),
                    truncated=data["truncated"])
    if data.get("reason"):
        manifest["reason"] = data["reason"]
    _atomic(directory / "manifest.json", manifest)


def _safe_write(turn, plan, failure=False):
    try:
        _write_capture(turn, plan, failure)
    except Exception:
        # Diagnostic I/O must never replace a successful result or original error.
        pass


def capture_turn(function):
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        scope = _SCOPE.get()
        if scope is None:
            return function(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
            agent = bound.arguments.get("self", args[0] if args else None)
            enabled = (_qualified(agent, scope)
                       and bound.arguments.get("user_input") == frozen_argv(scope["row"])[-1])
        except Exception:
            enabled = False
        if not enabled:
            token = _TURN.set(None)
            try:
                return function(*args, **kwargs)
            finally:
                _TURN.reset(token)
        turn = {"agent": agent, "scope": scope, "sequence": 0, "iteration": 0,
                "success_ref": None, "success_messages": None, "last_plan": None,
                "last_error": None, "guard_failure": None}
        token = _TURN.set(turn)
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            try:
                guarded = turn.get("guard_failure")
                if guarded is not None and guarded[0] is exc:
                    plan = guarded[1]
                else:
                    plan = turn.get("last_plan")
                    if plan is None:
                        plan = _base(turn)
                        _mark(plan, "real turn error has no captured planner input")
                plan["result"] = {**plan.get("result", {}), "success": False, "error": _error(exc)}
                plan["previous_success"] = turn["success_ref"]
                plan["real_turn_failure"] = True
                if (guarded is None or guarded[0] is not exc) and turn.get("last_error") is not exc:
                    plan["replay_qualified"] = False
                    plan["reason"] = "escaping error did not originate in the captured planner or guard"
                _safe_write(turn, plan, failure=True)
            except Exception:
                pass
            raise
        finally:
            _TURN.reset(token)
    return wrapped


def set_iteration(agent, iteration):
    try:
        turn = _turn(agent)
        if turn is not None:
            turn["iteration"] = int(iteration)
    except Exception:
        pass


def stage(name):
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            if _TURN.get() is None:
                return function(*args, **kwargs)
            try:
                bound = signature.bind(*args, **kwargs)
                agent = bound.arguments.get("self", args[0] if args else None)
                turn = _turn(agent)
                if turn is None:
                    raise ValueError("Inactive synthetic turn")
                label, call_id = name, None
                if name == "tool_result_fit_probe":
                    label = "preflight_integer_probe" if bound.arguments.get("result_tokens", 0) else name
                    call_id = bound.arguments["current"]["call"].id
                elif name == "outbound_fit":
                    recorded = any(item.get("state") == "recorded" for item in getattr(agent, "_pending_calls", ()))
                    label = "outbound_post_result_fit" if recorded else "outbound_initial_fit"
                state = {"name": label, "current_call_id": call_id, "plan": None}
            except Exception:
                return function(*args, **kwargs)
            token = _STAGE.set(state)
            try:
                result = function(*args, **kwargs)
                if name == "outbound_fit" and state["plan"] is not None:
                    plan = state["plan"]
                    if plan.get("result", {}).get("success"):
                        plan["actual_fit_completed"] = True
                        plan["delivery_claim"] = "planned actual fit only; not proof of server receipt"
                        _safe_write(turn, plan)
                return result
            except BaseException as exc:
                if name == "outbound_fit":
                    turn["last_error"] = exc
                    if state["plan"] is not None:
                        turn["last_plan"] = state["plan"]
                        if state["plan"].get("result", {}).get("success"):
                            state["plan"]["replay_qualified"] = False
                            state["plan"]["reason"] = "actual fit failed after planner returned; planner replay alone cannot reproduce this error"
                raise
            finally:
                _STAGE.reset(token)
        return wrapped
    return decorate


def _stats(stats, fitted, limit):
    clean = {key: value for key, value in stats.items() if key != "artifacts"}
    clean["artifacts"] = [{key: value for key, value in artifact.items() if key != "content"}
                          for artifact in stats.get("artifacts", ())]
    clean["fitted_messages_hash"] = stable_hash(fitted)
    clean["fitted_tokens"] = stats.get("tokens")
    return _copy(clean, limit)


def capture_plan(function):
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        if _TURN.get() is None:
            return function(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
            turn = _turn(bound.arguments.get("self", args[0] if args else None))
            if turn is None:
                raise ValueError("Inactive synthetic turn")
            plan = _base(turn)
            _capture_pending(turn, plan)
            state = _STAGE.get()
            if state is not None:
                plan["stage"] = state["name"]
                plan["current_call_id"] = state["current_call_id"]
                state["plan"] = plan
            try:
                messages = _copy(bound.arguments["messages"], _limit(turn))
                plan["canonical_messages"] = messages
                plan["input_hash"] = stable_hash(messages)
                plan["transition"] = _transition(turn, messages)
            except Exception as exc:
                _mark(plan, f"canonical input unavailable: {type(exc).__name__}")
            turn["last_plan"] = plan
        except Exception:
            return function(*args, **kwargs)
        token = _PLAN.set(plan)
        try:
            fitted, stats = result = function(*args, **kwargs)
            try:
                saved = _stats(stats, fitted, _limit(turn))
                plan["result"] = {"success": True, "stats": saved,
                                  "fitted_messages_hash": saved["fitted_messages_hash"],
                                  "fitted_tokens": saved["fitted_tokens"]}
                if plan["attempts"]:
                    plan["attempts"][-1]["stats"] = saved
                if plan["projected_messages"] is None or not plan["attempts"]:
                    plan["replay_qualified"] = False
                    plan["reason"] = "projected planner settings or reserve attempts not captured"
            except Exception as exc:
                _mark(plan, f"fitted stats unavailable: {type(exc).__name__}")
            return result
        except BaseException as exc:
            try:
                plan["result"] = {"success": False, "error": _error(exc)}
                if plan["attempts"]:
                    plan["attempts"][-1]["error"] = _error(exc)
                if plan["projected_messages"] is None or not plan["attempts"]:
                    plan["replay_qualified"] = False
                    plan["reason"] = "projected planner settings or reserve attempts not captured"
                turn["last_error"] = exc
            except Exception:
                pass
            raise
        finally:
            _PLAN.reset(token)
    return wrapped


def plan_projected(agent, messages, **settings):
    try:
        turn, plan = _turn(agent), _PLAN.get()
        if turn is None or plan is None:
            return
        plan["projected_messages"] = _copy(messages, _limit(turn))
        plan["settings"] = _copy(settings, 8192)
    except Exception as exc:
        if _PLAN.get() is not None:
            _mark(_PLAN.get(), f"projected input unavailable: {type(exc).__name__}")


def plan_attempt(agent, reserve, budget_tokens, session_record):
    try:
        turn, plan = _turn(agent), _PLAN.get()
        if turn is None or plan is None:
            return
        from .context import _validated_session_record
        record = _copy(_validated_session_record(session_record), 8192)
        plan["session_record"] = record
        if len(plan["attempts"]) >= 64:
            _mark(plan, "reserve attempt count exceeds capture allowance")
            return
        plan["attempts"].append({"reserve": reserve, "budget_tokens": budget_tokens,
                                 "session_record": record, "settings": plan["settings"],
                                 "projected_messages": plan["projected_messages"]})
    except Exception as exc:
        if _PLAN.get() is not None:
            _mark(_PLAN.get(), f"attempt unavailable: {type(exc).__name__}")

def plan_rejection(agent, exc):
    """Record the actual rejected reserve without observing or changing policy."""
    try:
        turn, plan = _turn(agent), _PLAN.get()
        if turn is not None and plan is not None and plan["attempts"]:
            plan["attempts"][-1]["error"] = _error(exc)
    except Exception as diagnostic_error:
        if _PLAN.get() is not None:
            _mark(_PLAN.get(), f"reserve rejection unavailable: {type(diagnostic_error).__name__}")



def fitter_checkpoint(**state):
    try:
        turn, plan = _turn(), _PLAN.get()
        if turn is not None and plan is not None and plan["attempts"]:
            plan["attempts"][-1]["checkpoint"] = _copy(state, _limit(turn))
            if "accounting_error" in state:
                plan["replay_qualified"] = False
                plan["reason"] = "checkpoint accounting unavailable: " + str(state["accounting_error"])[:100]
    except Exception as exc:
        if _PLAN.get() is not None:
            _mark(_PLAN.get(), f"checkpoint unavailable: {type(exc).__name__}")


def receipt_retention(status):
    try:
        receipt = _RECEIPT.get()
        if receipt is not None:
            if status not in ("success", "full", "error"):
                receipt["invalid_event"] = True
            elif len(receipt["retention_events"]) < 8:
                receipt["retention_events"].append(status)
            else:
                receipt["invalid_event"] = True
    except Exception:
        pass


def capture_receipt(function):
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        if _TURN.get() is None:
            return function(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
            turn = _turn()
            if turn is None or bound.arguments.get("memory") is not turn["agent"].memory:
                raise ValueError("Inactive synthetic receipt")
            from .refusals import execution_ran
            source = bound.arguments["result"]
            receipt = {"tool_name": bound.arguments["name"], "raw_result": str(source),
                       "executed": execution_ran(source), "outcome": getattr(source, "outcome", "unknown"),
                       "retention_available": callable(getattr(bound.arguments["memory"], "retain_tool_output", None)),
                       "retention_events": []}
            starting_sequence = turn["sequence"]
        except Exception:
            return function(*args, **kwargs)
        token = _RECEIPT.set(receipt)
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            try:
                template = turn.get("last_plan")
                # Closure helpers use unrelated length probes: they cannot replace
                # the original guard failure or reuse an earlier planner template.
                if (turn["guard_failure"] is None and template is not None
                        and template["id"] > starting_sequence
                        and template.get("stage") == "tool_result_fit_probe"):
                    plan = _copy(template, _limit(turn))
                    plan["kind"] = "receipt"
                    receipt["current_call_id"] = plan.get("current_call_id")
                    plan["receipt"] = _copy(receipt, _limit(turn))
                    plan["result"] = {"success": False, "error": _error(exc)}
                    if receipt.get("invalid_event") or receipt["current_call_id"] is None:
                        plan["replay_qualified"] = False
                        plan["reason"] = "receipt retention events or current call binding incomplete"
                    turn["guard_failure"] = (exc, plan)
            except Exception as diagnostic_error:
                plan = _base(turn, "receipt")
                _mark(plan, f"receipt source/template unavailable: {type(diagnostic_error).__name__}")
                turn["guard_failure"] = turn["guard_failure"] or (exc, plan)
            raise
        finally:
            _RECEIPT.reset(token)
    return wrapped
