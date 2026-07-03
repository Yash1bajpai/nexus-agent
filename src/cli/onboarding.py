"""
Nexus-Agent Onboarding Wizard — runs once on first launch.
Detects first run via ~/.nexus_agent_initialized.
Covers: welcome banner, API key setup, system spec detection, default provider.
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure safe output encoding on Windows terminals to prevent UnicodeEncodeError
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _rich = True
except ImportError:
    _rich = False

console = Console() if _rich else None
INIT_FILE = Path.home() / ".nexus_agent_initialized"
ENV_FILE = Path(__file__).resolve().parent.parent.parent.parent / ".env"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _print(text: str = "", style: str = ""):
    if console:
        console.print(text, style=style)
    else:
        print(text)

def _input(prompt: str) -> str:
    if console:
        console.print(f"[bold cyan]{prompt}[/bold cyan]", end="")
    else:
        print(prompt, end="")
    sys.stdout.flush()
    return input()

# ─── System Spec Detection ────────────────────────────────────────────────────

def detect_system_specs() -> dict:
    specs = {"ram_gb": 0, "cpu_cores": 0, "gpu": None, "cpu_name": None, "avx2": False}

    try:
        import psutil
        mem = psutil.virtual_memory()
        specs["ram_gb"] = round(mem.total / (1024 ** 3), 1)
        specs["cpu_cores"] = psutil.cpu_count(logical=False) or psutil.cpu_count()
    except ImportError:
        # psutil missing — default to 8GB middle tier, not lowest
        specs["ram_gb"] = -1  # sentinel: unknown

    # Try to detect CPU name + AVX2 support
    try:
        import platform
        cpu_name = platform.processor()
        if not cpu_name:
            import subprocess as _sp
            r = _sp.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=3)
            cpu_name = r.stdout.strip().split("\n")[-1].strip() if r.returncode == 0 else ""
        specs["cpu_name"] = cpu_name or None
    except Exception:
        pass

    # Detect AVX2 support (indicates modern CPU, good for llama.cpp)
    try:
        import subprocess as _sp
        r = _sp.run(["python", "-c",
                     "import platform; print('avx2' in platform.processor().lower())"],
                    capture_output=True, text=True, timeout=3)
        # Alternative: check via cpuinfo if available
        try:
            import cpuinfo
            flags = cpuinfo.get_cpu_info().get("flags", [])
            specs["avx2"] = "avx2" in flags
        except ImportError:
            # Conservative fallback: assume modern CPU if it's an Intel 10th gen+ or AMD Zen 2+
            cpu = (specs.get("cpu_name") or "").lower()
            specs["avx2"] = any(x in cpu for x in ["i5-1", "i7-1", "i9-1", "i5-12", "i7-12",
                                                     "i5-11", "i7-11", "ryzen 5 5", "ryzen 7 5",
                                                     "ryzen 5 7", "ryzen 7 7", "i5-10", "i7-10"])
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            gpu_info = lines[0].strip()
            name, mem_mb = [x.strip() for x in gpu_info.split(",")]
            specs["gpu"] = f"{name} ({int(mem_mb)//1024}GB VRAM)"
    except Exception:
        pass

    return specs

def suggest_local_model(specs: dict) -> tuple[str, str]:
    """
    Returns (model_suggestion, note) based on detected hardware.

    Tier logic:
      GPU any          → 7B+ models
      RAM > 16GB       → 7B models comfortably
      4 < RAM <= 16GB  → 3B-class models (Phi-3, Qwen2.5-3B, Llama-3.2-3B)
      RAM <= 4GB       → TinyLlama 1.1B only
      RAM unknown (-1) → default to 3B middle tier (don't assume worst)
    """
    ram = specs.get("ram_gb", 0)
    gpu = specs.get("gpu")
    avx2 = specs.get("avx2", False)
    cpu_name = specs.get("cpu_name") or ""

    avx2_note = " (AVX2 detected — fast inference)" if avx2 else ""

    if gpu:
        suggestion = f"Llama3-8B-Q5_K_M (~5GB VRAM){avx2_note}"
        note = f"GPU detected: {gpu}. 7B models will run at full speed."
    elif ram == -1:
        # psutil missing / unknown RAM — use middle tier, not lowest
        suggestion = f"Phi-3-Mini-3.8B-Q4_K_M (~2.3GB RAM){avx2_note}"
        note = "RAM could not be detected. Assuming ≥8GB — recommending 3B class. Adjust if your RAM is lower."
    elif ram > 16:
        suggestion = f"Mistral-7B-Q4_K_M (~4.1GB RAM){avx2_note}"
        note = f"{ram}GB RAM detected. 7B models will run comfortably."
    elif ram > 4:
        suggestion = f"Phi-3-Mini-3.8B-Q4_K_M (~2.3GB RAM){avx2_note}"
        note = (
            f"{ram}GB RAM detected. 3B-class models (Phi-3, Qwen2.5-3B, Llama-3.2-3B) "
            f"run well at Q4_K_M quantization."
        )
    else:
        suggestion = "TinyLlama-1.1B-Q4_K_M (637MB)"
        note = f"{ram}GB RAM detected. Only very small models are safe. TinyLlama recommended."

    return suggestion, note

# ─── Steps ────────────────────────────────────────────────────────────────────

def _show_welcome():
    if console:
        panel = Panel(
            "[bold white]Welcome to Nexus-Agent[/bold white]\n"
            "[dim]Autonomous AI Coding Agent[/dim]\n\n"
            "  Built by [cyan]Yash Bajpai[/cyan]\n"
            "  [dim]github.com/Yash1bajpai[/dim]\n"
            "  [dim]linkedin.com/in/yash-bajpai-b5a86332a[/dim]",
            border_style="cyan",
            expand=False,
            padding=(1, 4),
        )
        console.print()
        console.print(panel)
        console.print()
    else:
        print("\n+--------------------------------------+")
        print("|        Welcome to Nexus-Agent        |")
        print("|  Built by Yash Bajpai                |")
        print("|  github.com/Yash1bajpai              |")
        print("|  linkedin.com/in/yash-bajpai-b5a86332a |")
        print("+--------------------------------------+\n")

def _step_api_keys():
    """[1/3] - Check which API keys are set and offer to add missing ones."""
    _print("\n[bold][[1/3]][/bold] [cyan]API Key Setup[/cyan]" if console else "\n[1/3] API Key Setup")

    keys_to_check = [
        ("GEMINI_API_KEY", "Google Gemini"),
        ("ANTHROPIC_API_KEY", "Anthropic Claude"),
        ("OPENAI_API_KEY", "OpenAI GPT"),
    ]

    missing = []
    for env_key, label in keys_to_check:
        val = os.getenv(env_key, "")
        if val:
            _print(f"  [green][OK][/green] {label} - configured" if console else f"  [OK] {label} - configured")
        else:
            _print(f"  [red][MISSING][/red] {label} - missing" if console else f"  [MISSING] {label} - missing")
            missing.append((env_key, label))

    if missing:
        _print()
        _print("  [dim]You can add missing keys now or skip (press Enter to skip each).[/dim]" if console else
               "  You can add missing keys now or skip (press Enter to skip each).")
        for env_key, label in missing:
            try:
                val = _input(f"  Enter {label} API key (or Enter to skip): ").strip()
                if val:
                    _write_env_key(env_key, val)
                    os.environ[env_key] = val
                    _print(f"  [green]Saved {env_key} to .env[/green]" if console else f"  Saved {env_key} to .env")
            except (EOFError, KeyboardInterrupt):
                break

def _write_env_key(key: str, value: str):
    """Append or update a key in the .env file."""
    env_path = ENV_FILE
    if not env_path.exists():
        env_path.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}\n")
    env_path.write_text("".join(lines), encoding="utf-8")

def _step_system_specs():
    """[2/3] - Detect RAM, CPU, GPU and recommend a local model."""
    _print("\n[bold][[2/3]][/bold] [cyan]System Detection[/cyan]" if console else "\n[2/3] System Detection")

    specs = detect_system_specs()
    ram = specs["ram_gb"]
    cores = specs["cpu_cores"]
    gpu = specs["gpu"]
    cpu_name = specs.get("cpu_name") or ""
    avx2 = specs.get("avx2", False)

    ram_display = f"{ram} GB" if ram and ram > 0 else "Unknown (defaulting to middle tier)"
    avx2_display = "Yes [OK]" if avx2 else "Not detected"

    if console:
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column(style="dim")
        t.add_column()
        t.add_row("RAM", ram_display)
        t.add_row("CPU Cores", str(cores) if cores else "Unknown")
        if cpu_name:
            t.add_row("CPU", cpu_name[:60])
        t.add_row("AVX2", avx2_display)
        t.add_row("GPU", gpu or "Not detected")
        console.print(t)
    else:
        print(f"  RAM:       {ram_display}")
        print(f"  CPU Cores: {cores or 'Unknown'}")
        if cpu_name:
            print(f"  CPU:       {cpu_name[:60]}")
        print(f"  AVX2:      {avx2_display}")
        print(f"  GPU:       {gpu or 'Not detected'}")

    suggestion, note = suggest_local_model(specs)
    _print(f"\n  [bold]Recommended local model:[/bold] {suggestion}" if console else f"\n  Recommended local model: {suggestion}")
    _print(f"  [dim]{note}[/dim]" if console else f"  {note}")
    _print("  [dim]Note: These are conservative estimates. Closing browsers/IDEs frees RAM for larger models.[/dim]" if console else
           "  Note: These are conservative estimates. Closing browsers/IDEs frees RAM for larger models.")
    _print("  [dim]Local model support coming in V2. API providers active now.[/dim]" if console else
           "  Local model support coming in V2. API providers active now.")


def _step_default_provider() -> str:
    """[3/3] - Let the user pick their default provider."""
    _print("\n[bold][[3/3]][/bold] [cyan]Default Provider[/cyan]" if console else "\n[3/3] Default Provider")

    options = ["gemini", "anthropic", "openai", "auto"]
    _print("  Choose your default AI provider:")
    for i, opt in enumerate(options, 1):
        note = " (auto-fallback chain)" if opt == "auto" else ""
        _print(f"    {i}. {opt}{note}")

    current = os.getenv("DEFAULT_PROVIDER", "anthropic")
    _print(f"  [dim]Current default: {current}[/dim]" if console else f"  Current default: {current}")

    try:
        choice = _input("  Enter number (or Enter to keep current): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            selected = options[int(choice) - 1]
            _write_env_key("DEFAULT_PROVIDER", selected)
            os.environ["DEFAULT_PROVIDER"] = selected
            _print(f"  [green]Default provider set to: {selected}[/green]" if console else f"  Default provider set to: {selected}")
            return selected
    except (EOFError, KeyboardInterrupt):
        pass

    return current

# --- Main Entry ---

def run_if_first_time():
    """Run the onboarding wizard if this is the first launch. No-op on subsequent runs."""
    if INIT_FILE.exists():
        return

    _show_welcome()
    _step_api_keys()
    _step_system_specs()
    _step_default_provider()

    # Mark as initialized
    try:
        INIT_FILE.write_text("Nexus-Agent initialized.\n", encoding="utf-8")
    except Exception:
        pass

    _print()
    _print("[bold green][OK] Setup complete! Starting Nexus-Agent...[/bold green]\n" if console else
           "\n[OK] Setup complete! Starting Nexus-Agent...\n")
