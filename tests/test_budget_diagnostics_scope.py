"""Deterministic scope/capture tests without providers, models, or SQLite."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus_agent_ai.agent import budget_diagnostics as diagnostics
from nexus_agent_ai.agent.context import ContextBudgetError
from nexus_agent_ai.agent.refusals import execution_result


@pytest.fixture
def scope(tmp_path, monkeypatch):
    value = diagnostics.create_scope(tmp_path / "synthetic")
    monkeypatch.setenv("HOME", value["home"])
    monkeypatch.setenv("USERPROFILE", value["home"])
    monkeypatch.chdir(value["workspace"])
    return value


def load(scope, name):
    return json.loads((Path(scope["output"]) / "seed" / name).read_text())


class FakeAgent:
    def __init__(self, scope, action="success"):
        self.memory = SimpleNamespace(session_id=diagnostics.SESSION_ID,
                                      db_path=Path(scope["db_path"]),
                                      retain_tool_output=lambda plan: True)
        self.action = action
        self._pending_calls = []
        self.failure = ContextBudgetError("Historical omission coverage gap: synthetic fixture")
        self.messages = [{"role": "user", "content": diagnostics.frozen_argv("seed")[-1]}]
        self.raw = execution_result("synthetic raw output", executed=True, outcome="completed")
        self.call = {"call": SimpleNamespace(id="call-1", name="read_file"),
                     "state": "recorded", "result": self.raw}
        self.calls = 0

    @diagnostics.capture_plan
    def _plan_fit(self, messages, tools, scale=1.0, *, output_reserve=None, boost=None, result_tokens=0):
        self.calls += 1
        diagnostics.plan_projected(
            self, messages, context_window=4096, scale=scale,
            configured_reserve=512, initial_reserve=512, minimum_output=256,
            adaptive=True, boost=boost, system_tokens=10, tool_tokens=20,
            result_tokens=result_tokens, artifact_store_available=True,
            recovery_injected=False,
        )
        diagnostics.plan_attempt(self, 512, 3554, None)
        diagnostics.fitter_checkpoint(budget_tokens=3554 - result_tokens,
                                      protected_lower_bound_tokens=20,
                                      starts=[0], active_start=0, previous_start=0,
                                      cut=0, prefix_eligibility=[])
        if self.action == "adaptive":
            try:
                raise ContextBudgetError("exact first reserve rejection")
            except ContextBudgetError as exc:
                diagnostics.plan_rejection(self, exc)
            diagnostics.plan_attempt(self, 256, 3810, None)
            diagnostics.fitter_checkpoint(budget_tokens=3810, protected_lower_bound_tokens=20)
        if self.action in ("fatal", "receipt", "expected_probe") and len(messages) > 1:
            raise self.failure
        stats = {"tokens": 20, "artifacts": [{"artifact_id": "identity", "tool_name": "read_file",
                                               "byte_len": 10, "content": "NEVER_PERSIST_STATS_SOURCE"}]}
        return messages, stats

    @diagnostics.stage("outbound_fit")
    def _fit(self, messages, tools, scale=1.0):
        fitted, _stats = self._plan_fit(messages, tools, scale=scale)
        if self.action == "retention_error":
            raise self.failure
        return fitted

    @diagnostics.stage("tool_result_fit_probe")
    def _tool_result_fits(self, current, content, tools, scale=1.0, *, result_tokens=0):
        try:
            self._plan_fit(self.messages + [{"role": "tool", "tool_call_id": current["call"].id,
                                             "content": str(content)}], tools, scale,
                           result_tokens=result_tokens)
        except ContextBudgetError:
            return False
        return True

    @diagnostics.capture_turn
    def run(self, user_input, stream=False, require_edit=False):
        diagnostics.set_iteration(self, 1)
        if self.action == "huge":
            self.messages[0]["content"] = "x" * diagnostics.MAX_CAPTURE_BYTES
        self._fit(messages=self.messages, tools=[])
        if self.action == "fatal":
            self._pending_calls = [self.call]
            self._fit(self.messages + [{"role": "assistant", "content": "new assistant payload"}], [])
        elif self.action == "expected_probe":
            assert not self._tool_result_fits(self.call, "candidate", [], result_tokens=100)
        elif self.action == "receipt":
            try:
                synthetic_guard("read_file", self.raw, memory=self.memory,
                                fits=lambda value: self._tool_result_fits(self.call, value, []))
            except ContextBudgetError:
                # A closure guard has no planner calls and must not reuse or
                # overwrite the real failed receipt's prospective template.
                try:
                    synthetic_guard("closure", "other", memory=self.memory, fits=lambda value: False)
                except ContextBudgetError:
                    pass
                raise
        return "original stdout result"


@diagnostics.capture_receipt
def synthetic_guard(name, result, *, memory, fits):
    if fits(result):
        return result
    diagnostics.receipt_retention("full")
    fits("minimum truthful receipt")
    raise ContextBudgetError("truthful receipt cannot fit")


def run(agent, scope, prompt=None):
    with diagnostics.activate_scope(Path(scope["scope_path"]), "seed"):
        return agent.run(diagnostics.frozen_argv("seed")[-1] if prompt is None else prompt)


def test_scope_is_exclusively_new_and_paths_are_isolated(scope):
    assert Path(scope["home"]).is_dir()
    assert Path(scope["workspace"]).is_dir()
    assert scope["db_path"] == str(Path(scope["home"]) / ".nexus-agent/history.db")
    assert load_scope(scope)["run_id"] == scope["run_id"]
    with pytest.raises(FileExistsError):
        diagnostics.create_scope(Path(scope["root"]))


def load_scope(scope):
    return json.loads(Path(scope["scope_path"]).read_text())


def test_disabled_and_environment_opt_in_do_not_capture(scope, monkeypatch):
    monkeypatch.setenv("NEXUS_BUDGET_DIAGNOSTICS", scope["scope_path"])
    agent = FakeAgent(scope)
    assert agent.run(diagnostics.frozen_argv("seed")[-1]) == "original stdout result"
    assert list(Path(scope["output"]).iterdir()) == []
    assert not diagnostics.planner_active()


@pytest.mark.parametrize("mismatch", ["home", "userprofile", "cwd", "session", "db", "run", "row", "prompt"])
def test_wrong_scope_never_writes(scope, monkeypatch, tmp_path, mismatch):
    agent = FakeAgent(scope)
    row, prompt = "seed", diagnostics.frozen_argv("seed")[-1]
    if mismatch == "home":
        monkeypatch.setenv("HOME", str(tmp_path))
    elif mismatch == "userprofile":
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
    elif mismatch == "cwd":
        monkeypatch.chdir(tmp_path)
    elif mismatch == "session":
        agent.memory.session_id = "ordinary-session"
    elif mismatch == "db":
        agent.memory.db_path = tmp_path / "ordinary.db"
    elif mismatch == "run":
        value = load_scope(scope)
        value["run_id"] = "not-a-uuid"
        Path(scope["scope_path"]).write_text(json.dumps(value))
    elif mismatch == "row":
        row = "filler-7"
    else:
        prompt = "ordinary prompt"
    with diagnostics.activate_scope(Path(scope["scope_path"]), row):
        assert agent.run(prompt) == "original stdout result"
    assert list(Path(scope["output"]).iterdir()) == []


def test_latest_success_and_first_real_fatal_have_exact_linkage(scope):
    agent = FakeAgent(scope, "fatal")
    with pytest.raises(ContextBudgetError) as caught:
        run(agent, scope)
    assert caught.value is agent.failure
    success = load(scope, "latest_success.json")
    failure = load(scope, "failure.json")
    manifest = load(scope, "manifest.json")
    assert success["actual_fit_completed"] is True
    assert success["stage"] == "outbound_initial_fit"
    assert failure["stage"] == "outbound_post_result_fit"
    assert success["pending_calls"] == []
    assert failure["pending_calls"] == [
        {"name": "read_file", "id": "call-1", "state": "recorded", "executed": True}
    ]
    assert failure["previous_success"] == {"id": success["id"], "file": "latest_success.json",
                                            "input_hash": success["input_hash"]}
    assert failure["transition"]["append_only"] is True
    assert failure["transition"]["new_messages"][0]["index"] == 1
    assert failure["input_hash"] == diagnostics.stable_hash(failure["canonical_messages"])
    assert failure["attempts"][0]["settings"] == failure["settings"]
    assert failure["attempts"][0]["projected_messages"] == failure["projected_messages"]
    assert manifest["replay_qualified"] is True
    assert failure["real_turn_failure"] is True
    stats = success["result"]["stats"]
    assert "content" not in stats["artifacts"][0]
    assert "NEVER_PERSIST_STATS_SOURCE" not in json.dumps(success)
    assert stats["fitted_messages_hash"] == diagnostics.stable_hash(success["canonical_messages"])


def test_pending_recorded_state_does_not_infer_execution_from_plain_prose(scope):
    agent = FakeAgent(scope, "fatal")
    agent.call["result"] = "Executed successfully according to untyped prose"
    with pytest.raises(ContextBudgetError):
        run(agent, scope)
    failure = load(scope, "failure.json")
    assert failure["pending_calls"][0]["executed"] is None
    assert failure["replay_qualified"] is True


def test_missing_pending_identity_marks_capture_incomplete(scope):
    agent = FakeAgent(scope, "fatal")
    del agent.call["call"].name
    with pytest.raises(ContextBudgetError):
        run(agent, scope)
    failure = load(scope, "failure.json")
    assert failure["pending_calls"][0]["name"] is None
    assert failure["replay_qualified"] is False
    assert "pending call" in failure["reason"]


def test_expected_probe_failure_is_not_a_fatal_capture(scope):
    agent = FakeAgent(scope, "expected_probe")
    assert run(agent, scope) == "original stdout result"
    assert not (Path(scope["output"]) / "seed/failure.json").exists()
    assert load(scope, "latest_success.json")["stage"] == "outbound_initial_fit"
    assert load(scope, "manifest.json")["failure"] is None


def test_adaptive_success_preserves_actual_prior_rejection(scope):
    assert run(FakeAgent(scope, "adaptive"), scope) == "original stdout result"
    success = load(scope, "latest_success.json")
    assert success["result"]["success"] is True
    assert len(success["attempts"]) == 2
    first, second = success["attempts"]
    assert first["error"]["class"] == "ContextBudgetError"
    assert first["error"]["text"] == "exact first reserve rejection"
    assert "inferred" not in first["error"]
    assert "error" not in second
    assert second["stats"]["fitted_tokens"] == 20
    assert load(scope, "manifest.json")["failure"] is None


def test_success_not_committed_before_real_retention_returns(scope):
    agent = FakeAgent(scope, "retention_error")
    with pytest.raises(ContextBudgetError) as caught:
        run(agent, scope)
    assert caught.value is agent.failure
    assert not (Path(scope["output"]) / "seed/latest_success.json").exists()
    failure = load(scope, "failure.json")
    assert failure["previous_success"] is None
    assert failure["replay_qualified"] is False
    assert "after planner returned" in failure["reason"]


def test_receipt_source_is_bound_to_real_guard_not_closure(scope):
    agent = FakeAgent(scope, "receipt")
    with pytest.raises(ContextBudgetError, match="truthful receipt cannot fit"):
        run(agent, scope)
    failure = load(scope, "failure.json")
    assert failure["kind"] == "receipt"
    assert failure["receipt"] == {"tool_name": "read_file", "raw_result": str(agent.raw),
                                    "executed": True, "outcome": "completed",
                                    "retention_available": True, "retention_events": ["full"],
                                    "current_call_id": "call-1"}
    assert failure["projected_messages"][-1]["content"] == "minimum truthful receipt"
    assert failure["result"]["error"]["text"] == "truthful receipt cannot fit"


def test_oversized_input_is_valid_explicit_bounded_marker(scope):
    assert run(FakeAgent(scope, "huge"), scope) == "original stdout result"
    value = load(scope, "latest_success.json")
    assert value["replay_qualified"] is False
    assert value["truncated"] is True
    assert value["reason"]
    total = sum(path.stat().st_size for path in (Path(scope["output"]) / "seed").iterdir())
    assert total <= scope["max_capture_bytes"]


@pytest.mark.parametrize("error_length", [6000, diagnostics.MAX_CAPTURE_BYTES + 1])
def test_exception_text_is_exact_when_qualified_or_explicitly_unreplayable(scope, error_length):
    agent = FakeAgent(scope, "fatal")
    text = "synthetic exact failure " + "x" * error_length
    agent.failure = ContextBudgetError(text)
    with pytest.raises(ContextBudgetError) as caught:
        run(agent, scope)
    assert caught.value is agent.failure
    value = load(scope, "failure.json")
    if error_length == 6000:
        assert value["replay_qualified"] is True
        assert value["truncated"] is False
        assert value["result"]["error"]["text"] == text
        assert value["attempts"][-1]["error"]["text"] == text
    else:
        assert value["replay_qualified"] is False
        assert value["truncated"] is True
        assert value["reason"]
        assert value["limitations"]
    total = sum(path.stat().st_size for path in (Path(scope["output"]) / "seed").iterdir())
    assert total <= scope["max_capture_bytes"]


def test_writer_errors_preserve_result_and_original_exception(scope, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("synthetic writer failure")
    monkeypatch.setattr(diagnostics, "_atomic", fail)
    assert run(FakeAgent(scope), scope) == "original stdout result"
    failing = FakeAgent(scope, "fatal")
    with pytest.raises(ContextBudgetError) as caught:
        run(failing, scope)
    assert caught.value is failing.failure


def test_suspended_capture_and_signature_preservation(scope):
    assert str(inspect.signature(FakeAgent.run)) == "(self, user_input, stream=False, require_edit=False)"
    with diagnostics.activate_scope(Path(scope["scope_path"]), "seed"):
        with diagnostics.suspended():
            FakeAgent(scope).run(diagnostics.frozen_argv("seed")[-1])
    assert list(Path(scope["output"]).iterdir()) == []


def test_first_real_failure_is_not_replaced_by_later_turn_or_closure(scope):
    with diagnostics.activate_scope(Path(scope["scope_path"]), "seed"):
        with pytest.raises(ContextBudgetError):
            FakeAgent(scope, "fatal").run(diagnostics.frozen_argv("seed")[-1])
        first = load(scope, "failure.json")
        later = FakeAgent(scope, "fatal")
        later.failure = ContextBudgetError("different later failure")
        with pytest.raises(ContextBudgetError):
            later.run(diagnostics.frozen_argv("seed")[-1])
    assert load(scope, "failure.json") == first



def test_cli_rejects_changed_argv_without_importing_or_capturing(scope):
    from nexus_agent_ai.diagnostic_cli import main
    with pytest.raises(SystemExit) as caught:
        main(["--scope", scope["scope_path"], "--row", "seed", "--", "chat", "ordinary"])
    assert caught.value.code == 2
    assert list(Path(scope["output"]).iterdir()) == []


def test_cli_preserves_normal_entrypoint_stdout_and_exit(scope, monkeypatch, capsys):
    import sys
    from nexus_agent_ai.diagnostic_cli import main

    def normal_app(*, args, prog_name):
        assert args == diagnostics.frozen_argv("seed")
        assert prog_name == "nexus-agent"
        print("unmodified normal CLI stdout")
        raise SystemExit(7)

    monkeypatch.setitem(sys.modules, "nexus_agent_ai.cli.app", SimpleNamespace(app=normal_app))
    with pytest.raises(SystemExit) as caught:
        main(["--scope", scope["scope_path"], "--row", "seed", "--", *diagnostics.frozen_argv("seed")])
    assert caught.value.code == 7
    assert capsys.readouterr().out == "unmodified normal CLI stdout\n"
