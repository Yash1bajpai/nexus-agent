import pytest
from typer.testing import CliRunner

from nexus_agent_ai.utils import model_select as ms
from nexus_agent_ai.cli import app as appmod


def test_pick_model_uses_available_ram_with_headroom():
    assert ms.pick_model(8.0)["filename"] == "LFM2.5-2.6B-Q6_K.gguf"
    # 2 GB free: not even the compact 2.6B fits with headroom -> small model
    small = ms.pick_model(2.0)
    assert small["filename"] == ms.SMALL_MODEL["filename"]
    assert "2.0 GB" in small["reason"]


def test_pick_model_middle_tier_and_context_scaling():
    assert ms.pick_model(3.5)["filename"] == "LFM2.5-2.6B-Q5_K_M.gguf"
    # A bigger context window needs more RAM, so the same free RAM steps down
    assert ms.pick_model(3.1, context_size=4096)["filename"] == "LFM2.5-2.6B-Q4_0.gguf"
    assert ms.pick_model(3.1, context_size=16384)["filename"] == ms.SMALL_MODEL["filename"]


def test_pick_model_unknown_ram_is_safe():
    assert ms.pick_model(None)["filename"] == ms.SMALL_MODEL["filename"]


def test_pick_model_android_skips_q6():
    assert ms.pick_model(8.0, android=True)["filename"] == "LFM2.5-2.6B-Q5_K_M.gguf"


def test_confirm_download_shows_size_speed_and_asks():
    lines = []
    choice = ms.pick_model(8.0)
    result = ms.confirm_download(choice, ask=lambda p: "y", interactive=True,
                                 speed_probe=lambda *a, **k: 0.04, out=lines.append)
    assert result["filename"] == choice["filename"]
    text = "\n".join(lines)
    assert "2.2 GB" in text and "0.0 MB/s" in text or "0.0" in text
    assert "resumes" in text


def test_confirm_download_can_pick_small_or_decline():
    choice = ms.pick_model(8.0)
    common = dict(interactive=True, speed_probe=lambda *a, **k: 5.0, out=lambda m: None)
    assert ms.confirm_download(choice, ask=lambda p: "s", **common)["filename"] == ms.SMALL_MODEL["filename"]
    assert ms.confirm_download(choice, ask=lambda p: "n", **common) is None


def test_confirm_download_non_interactive_does_not_prompt():
    def boom(_):
        raise AssertionError("must not prompt")
    choice = ms.pick_model(8.0)
    assert ms.confirm_download(choice, ask=boom, interactive=False,
                               speed_probe=lambda *a, **k: None, out=lambda m: None) is choice


def test_format_eta():
    assert "speed unknown" in ms.format_eta(2.0, None)
    assert "h" in ms.format_eta(2.2, 0.04)
    assert "min" in ms.format_eta(2.2, 10.0)


def test_available_ram_is_number_or_none():
    value = ms.available_ram_gb()
    assert value is None or value > 0


def test_fit_local_model_respects_env_override(monkeypatch):
    monkeypatch.setenv("NEXUS_AGENT_MODEL_FILENAME", "x.gguf")
    monkeypatch.setattr(ms, "confirm_download", lambda *a, **k: pytest.fail("should not ask"))
    from nexus_agent_ai.providers.local_provider import LocalProvider
    prov = LocalProvider()
    assert appmod._fit_local_model(prov, None, {}) is prov


def test_fit_local_model_low_ram_asks_then_uses_small(monkeypatch):
    monkeypatch.delenv("NEXUS_AGENT_MODEL_REPO", raising=False)
    monkeypatch.delenv("NEXUS_AGENT_MODEL_FILENAME", raising=False)
    monkeypatch.setattr(ms, "available_ram_gb", lambda: 1.5)
    monkeypatch.setattr(ms, "is_cached", lambda r, f: False)
    seen = {}
    def fake_confirm(choice, ctx, out=None):
        seen["choice"] = choice
        return choice
    monkeypatch.setattr(ms, "confirm_download", fake_confirm)
    from nexus_agent_ai.providers.local_provider import LocalProvider
    prov = appmod._fit_local_model(LocalProvider(), None, {})
    assert seen["choice"]["filename"] == ms.SMALL_MODEL["filename"]
    assert prov.filename == ms.SMALL_MODEL["filename"]


def test_fit_local_model_declined_exits(monkeypatch):
    import typer
    monkeypatch.delenv("NEXUS_AGENT_MODEL_REPO", raising=False)
    monkeypatch.delenv("NEXUS_AGENT_MODEL_FILENAME", raising=False)
    monkeypatch.setattr(ms, "available_ram_gb", lambda: 8)
    monkeypatch.setattr(ms, "is_cached", lambda r, f: False)
    monkeypatch.setattr(ms, "confirm_download", lambda *a, **k: None)
    from nexus_agent_ai.providers.local_provider import LocalProvider
    with pytest.raises(typer.Exit):
        appmod._fit_local_model(LocalProvider(), None, {})


def test_fit_local_model_keeps_cached_default(monkeypatch):
    monkeypatch.delenv("NEXUS_AGENT_MODEL_REPO", raising=False)
    monkeypatch.delenv("NEXUS_AGENT_MODEL_FILENAME", raising=False)
    monkeypatch.setattr(ms, "available_ram_gb", lambda: 1.5)
    monkeypatch.setattr(ms, "is_cached", lambda r, f: f == "LFM2.5-2.6B-Q6_K.gguf")
    monkeypatch.setattr(ms, "confirm_download", lambda *a, **k: pytest.fail("should not ask"))
    from nexus_agent_ai.providers.local_provider import LocalProvider
    prov = LocalProvider()
    assert appmod._fit_local_model(prov, None, {}) is prov


def test_onboarding_recommendation_downgrades_on_low_free_ram(monkeypatch):
    from nexus_agent_ai.cli import onboarding
    monkeypatch.setattr(onboarding, "_is_android", lambda: False)
    cfg = onboarding.recommended_model_config({"ram_gb": 4, "available_ram_gb": 1.2})
    assert cfg["filename"] == ms.SMALL_MODEL["filename"]
    assert cfg["repo"] == ms.SMALL_MODEL["repo"]
    assert "1.2 GB free" in cfg["note"]


def test_chat_without_query_opens_repl(monkeypatch):
    called = {}
    monkeypatch.setattr(appmod, "repl", lambda **kw: called.update(kw))
    appmod.chat(query=None, provider="local", verbose=False, no_stream=False, max_iterations=10,
                persist=False, session=None, context_size=None)
    assert called["provider"] == "local"


def test_doctor_command_runs_offline():
    result = CliRunner().invoke(appmod.app, ["doctor", "--no-network"])
    assert "Python" in result.output and "Model fit" in result.output
    assert result.exit_code in (0, 1)


def test_doctor_flags_old_python(monkeypatch):
    from nexus_agent_ai.utils import doctor
    class V(tuple):
        major, minor, micro = 3, 9, 1
    monkeypatch.setattr(doctor.sys, "version_info", V((3, 9, 1)))
    checks = doctor.run_checks(probe_network=False)
    assert checks[0][0] == "fail"
