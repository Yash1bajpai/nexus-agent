"""Does overriding the user profile folder isolate EVERYTHING Nexus stores? (Windows: USERPROFILE, POSIX: HOME)
Run in a fresh process with the override already set. Argument: the REAL profile folder (to prove it stays untouched).
Fake key only. Prints PASS/FAIL lines; exit 1 on any FAIL."""
import os, sys
from pathlib import Path
real = Path(sys.argv[1])
fake_home = Path(sys.argv[2])
ok = []
def chk(label, cond, extra=""):
    ok.append(cond); print(("PASS " if cond else "FAIL ") + label, extra, flush=True)

before = sorted(p.name for p in real.iterdir()) if real.exists() else []
chk("Path.home() follows the override", Path.home().resolve() == fake_home.resolve(), f"{Path.home()}")
from nexus_agent_ai.cli import onboarding
from nexus_agent_ai.utils import config, model_download, doctor
from nexus_agent_ai.providers import local_provider
FAKE = "AIzaFAKEFAKEFAKEFAKE0123456789abcdef"
onboarding._write_env_key("GEMINI_API_KEY", FAKE)
onboarding._mark_initialized()
inside = lambda p: fake_home.resolve() in Path(p).resolve().parents or Path(p).resolve() == fake_home.resolve()
chk("settings/keys file is under the override", inside(onboarding.ENV_FILE) and onboarding.ENV_FILE.exists(), str(onboarding.ENV_FILE))
chk("first-run flag is under the override", inside(onboarding.INIT_FILE) and onboarding.INIT_FILE.exists(), str(onboarding.INIT_FILE))
chk("config dir is under the override", inside(config.USER_CONFIG_DIR), str(config.USER_CONFIG_DIR))
chk("model folder is under the override", inside(model_download.models_dir()), str(model_download.models_dir()))
chk("engine folder is under the override", inside(local_provider._NEXUS_HOME), str(local_provider._NEXUS_HOME))
chk("first-run wizard would show (no flag in the real profile is consulted)", not (real / ".nexus_agent_initialized").exists() or inside(onboarding.INIT_FILE))
after = sorted(p.name for p in real.iterdir()) if real.exists() else []
chk("real profile folder gained no new entries", before == after, f"new: {sorted(set(after) - set(before))}")
def _has_fake(p):
    try:
        return p.is_file() and p.stat().st_size < 100000 and FAKE in p.read_text(errors="ignore")
    except OSError:
        return False  # unreadable system files cannot hold what this process just wrote
leak = [str(p) for p in (real.rglob("*") if real.exists() else []) if _has_fake(p)]
chk("fake key not found anywhere in the real profile", not leak, str(leak[:3]))
sys.exit(0 if all(ok) else 1)
