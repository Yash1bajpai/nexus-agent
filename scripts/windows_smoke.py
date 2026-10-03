"""Real-Windows smoke checks run by .github/workflows/windows.yml. Exits nonzero on any failed assert."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

assert os.name == "nt" or "--allow-posix" in sys.argv, "must run on Windows"
home = Path(tempfile.mkdtemp(prefix="nx home "))
os.environ["USERPROFILE"] = str(home)
os.environ["HOME"] = str(home)

from nexus_agent_ai.cli import onboarding as o  # noqa: E402
from nexus_agent_ai.agent.persistence import get_workspace_session_id  # noqa: E402
from nexus_agent_ai.agent import tools as T  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))


# 1. key file ACL: only the current user may have access
o.USER_CONFIG_DIR = home / ".nexus-agent"
o.ENV_FILE = o.USER_CONFIG_DIR / ".env"
o.USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
o._restrict_permissions(o.USER_CONFIG_DIR, 0o700)
o._write_env_key("OPENAI_API_KEY", "sk-test-not-real")
if os.name == "nt":
    acl = subprocess.run(["icacls", str(o.ENV_FILE)], capture_output=True, text=True).stdout
    print(acl)
    user = os.environ.get("USERNAME", "").lower()
    lines = [l.strip().lower() for l in acl.splitlines() if ":(" in l]
    check("env ACL has only current user", lines and all(user in l for l in lines), str(lines))
    bad = [l for l in lines if any(g in l for g in ("everyone", "builtin\\users", "authenticated users"))]
    check("env ACL has no broad group", not bad, str(bad))
else:
    check("env mode 0600", (o.ENV_FILE.stat().st_mode & 0o777) == 0o600)
check("env content written", "sk-test-not-real" in o.ENV_FILE.read_text(encoding="utf-8"))

# 2. session ids from drive-letter / spaced / unicode paths
d = Path(tempfile.mkdtemp(prefix="my proj é "))
a = get_workspace_session_id(d)
b = get_workspace_session_id(Path(str(d).upper()) if os.name == "nt" else d)
check("session id safe chars", all(c.isalnum() or c in "-_" for c in a), a)
check("session id case-insensitive on Windows", a == b or os.name != "nt", f"{a} vs {b}")
check("session id from drive path", a.endswith(tuple("0123456789abcdef")), str(d))

# 3. CRLF patching in a path with spaces and unicode
os.chdir(d)
(d / "calc.py").write_bytes(b"def add(a,b):\r\n    return a - b\r\n")
T.reset_file_tracking()
r0 = T.execute_patch_file("calc.py", "return a - b", "return a + b")
check("patch without read refused", r0.startswith("ERROR"), r0[:80])
T.execute_read_file("calc.py")
r1 = T.execute_patch_file("calc.py", "def add(a,b):\n    return a - b", "def add(a,b):\n    return a + b")
data = (d / "calc.py").read_bytes()
check("CRLF patch applied", b"return a + b" in data, r1[:80])
check("CRLF endings preserved", data.count(b"\r\n") == 2 and b"\n" not in data.replace(b"\r\n", b""))

# 4. HF symlink warning silenced
import nexus_agent_ai.cli.app  # noqa: E402,F401
check("HF symlink warning disabled", os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING") == "1" or os.name != "nt")

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
