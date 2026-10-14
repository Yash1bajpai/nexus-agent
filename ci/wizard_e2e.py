"""First-run wizard end to end on Linux/macOS through a real pseudo-terminal, in a throwaway HOME.

Types a FAKE Gemini key, picks the Qwen2.5-Coder model from the model list, accepts the download, lets the
REPL start and asks one trivial question. Then checks: the choice and the key were saved globally with owner-only
permissions, the fake key never appears in output or in the working folders, and a second launch from a different
folder uses the same saved model. Scratch only; answers are typed by this script.
Usage: python wizard_e2e.py <nexus-agent-ai exe> <timeout seconds>
"""
import os, pty, re, select, stat, subprocess, sys, tempfile, time

EXE, TMO = sys.argv[1], int(sys.argv[2])
home = tempfile.mkdtemp(prefix="home_")
dirA = tempfile.mkdtemp(prefix="folderA_")
dirB = tempfile.mkdtemp(prefix="folderB_")
FAKE = "AIzaFAKEFAKEFAKEFAKE0123456789abcdef"
env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and k not in ("DEFAULT_PROVIDER",) and not k.startswith("NEXUS_AGENT_")}
env.update(HOME=home, USERPROFILE=home, TERM="xterm")
strip = lambda b: re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", b.decode(errors="replace"))

pid, fd = pty.fork()
if pid == 0:
    os.chdir(dirA)
    os.execve(EXE, [EXE], env)
out = b""; sent = set(); t0 = time.time(); asked_q = False; done = False
while time.time() - t0 < TMO:
    r, _, _ = select.select([fd], [], [], 2)
    if not r:
        continue
    try:
        d = os.read(fd, 8192)
    except OSError:
        break
    if not d:
        break
    out += d
    txt = strip(out)
    def reply(key, marker, answer):
        if key not in sent and marker in txt[-1500:]:
            sent.add(key); os.write(fd, answer.encode() + b"\n")
    reply("gem", "Enter Google Gemini API key", FAKE)
    reply("ant", "Enter Anthropic Claude API key", "")
    reply("oai", "Enter OpenAI GPT API key", "")
    reply("prov", "Enter number (or Enter to keep current)", "")
    if "model" not in sent and "Enter number (or Enter for the suggested model)" in txt[-1500:]:
        lines = txt.split("Choose a model")[-1]
        n = [m.group(1) for m in re.finditer(r"^\s*(\d+)\. qwen2\.5-coder", lines, re.M)]
        sent.add("model"); os.write(fd, (n[0] if n else "").encode() + b"\n")
    reply("anyway", "Use it anyway?", "y")
    reply("dl", "Download model weights now?", "y")
    if not asked_q and "Type /help" in txt:
        time.sleep(3); os.write(fd, b"say hi in five words\n"); asked_q = True
    if asked_q and "Tokens:" in txt and not done:
        done = True; time.sleep(1); os.write(fd, b"/exit\n"); time.sleep(3); break
try:
    os.kill(pid, 9)
except OSError:
    pass
log = strip(out)
open("wizard_e2e.log", "w").write(log.replace(FAKE, "<FAKE-KEY>"))
nx = os.path.join(home, ".nexus-agent")
envf = os.path.join(nx, ".env")
txtenv = open(envf).read() if os.path.exists(envf) else ""
mode = stat.S_IMODE(os.stat(envf).st_mode) if os.path.exists(envf) else None
checks = []
def chk(label, cond, extra=""):
    checks.append(cond); print(("PASS " if cond else "FAIL ") + label, extra, flush=True)
chk("wizard reached the REPL and got an answer", "Tokens:" in log)
chk("model list was shown", "Choose a model" in log)
chk("chosen model saved globally", "NEXUS_AGENT_MODEL_FILENAME=qwen2.5-coder-3b-instruct-q4_k_m.gguf" in txtenv)
chk("fake key saved globally", FAKE in txtenv)
if os.name == "posix":
    chk(".env is owner-only", mode == 0o600, oct(mode) if mode is not None else "missing")
chk("fake key not printed in the session output", FAKE not in log)
leak = [os.path.join(r, f) for d in (dirA, dirB) for r, _, fs in os.walk(d) for f in fs if FAKE in open(os.path.join(r, f), errors="ignore").read()]
chk("fake key not written into the working folders", not leak, str(leak))
p = subprocess.run([EXE.replace("nexus-agent-ai", "nexus-agent"), "chat", "-p", "local", "say hi in five words"], cwd=dirB, env=env, capture_output=True, text=True, timeout=600)
o = strip((p.stdout + p.stderr).encode())
chk("second launch from another folder succeeded", p.returncode == 0, f"exit {p.returncode}")
chk("second launch uses the saved model", "qwen2.5-coder-3b-instruct-q4_k_m.gguf" in o)
chk("fake key not in second launch output", FAKE not in o)
print("model line:", [l.strip() for l in o.splitlines() if "Provider:" in l][:1])
sys.exit(0 if all(checks) else 1)
