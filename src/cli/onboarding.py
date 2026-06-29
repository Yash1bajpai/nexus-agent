"""
DevMind Onboarding Wizard — runs once on first launch.
Detects first run via ~/.devmind_initialized.
Covers: welcome banner, API key setup, system spec detection, default provider.
"""

import os
import subprocess
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _rich = True
except ImportError:
    _rich = False

console = Console() if _rich else None
INIT_FILE = Path.home() / ".devmind_initialized"
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
    specs = {"ram_gb": 0, "cpu_cores": 0, "gpu": None}

    try:
        import psutil
        mem = psutil.virtual_memory()
        specs["ram_gb"] = round(mem.total / (1024 ** 3), 1)
        specs["cpu_cores"] = psutil.cpu_count(logical=False) or psutil.cpu_count()
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            gpu_info = lines[0].strip()
            name, mem_mb = [x.strip() for x in gpu_info.split(",")]
            specs["gpu"] = f"{name} ({int(mem_mb)//1024}GB VRAM)"
    except Exception:
        pass

    return specs

def suggest_local_model(specs: dict) -> str:
    ram = specs.get("ram_gb", 0)
    gpu = specs.get("gpu")

    if gpu:
        return f"Llama3-8B-Q5 (GPU detected: {gpu})"
    elif ram >= 16:
        return "Mistral-7B-Q4 (~4GB RAM usage)"
    elif ram >= 8:
        return "Phi-3-Mini-Q4 (~2.5GB RAM usage)"
    else:
        return "TinyLlama-1.1B-Q4 (637MB) — lightest option"

# ─── Steps ────────────────────────────────────────────────────────────────────

def _show_welcome():
    if console:
        panel = Panel(
            "[bold white]Welcome to DevMind[/bold white]\n"
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
        print("\n╔══════════════════════════════════════╗")
        print("║         Welcome to DevMind           ║")
        print("║  Built by Yash Bajpai                ║")
        print("║  github.com/Yash1bajpai              ║")
        print("║  linkedin.com/in/yash-bajpai-b5a86332a ║")
        print("╚══════════════════════════════════════╝\n")

def _step_api_keys():
    """[1/3] — Check which API keys are set and offer to add missing ones."""
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
            _print(f"  [green]✓[/green] {label} — configured" if console else f"  ✓ {label} — configured")
        else:
            _print(f"  [red]✗[/red] {label} — missing" if console else f"  ✗ {label} — missing")
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
    """[2/3] — Detect RAM, CPU, GPU and recommend a local model."""
    _print("\n[bold][[2/3]][/bold] [cyan]System Detection[/cyan]" if console else "\n[2/3] System Detection")

    specs = detect_system_specs()
    ram = specs["ram_gb"]
    cores = specs["cpu_cores"]
    gpu = specs["gpu"]

    if console:
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column(style="dim")
        t.add_column()
        t.add_row("RAM", f"{ram} GB" if ram else "Unknown")
        t.add_row("CPU Cores", str(cores) if cores else "Unknown")
        t.add_row("GPU", gpu or "Not detected")
        console.print(t)
    else:
        print(f"  RAM:       {ram} GB" if ram else "  RAM:       Unknown")
        print(f"  CPU Cores: {cores}" if cores else "  CPU Cores: Unknown")
        print(f"  GPU:       {gpu or 'Not detected'}")

    suggestion = suggest_local_model(specs)
    _print(f"\n  [bold]Recommended local model:[/bold] {suggestion}" if console else f"\n  Recommended local model: {suggestion}")
    _print("  [dim]Note: Local model support coming in V2. API providers active now.[/dim]" if console else
           "  Note: Local model support coming in V2. API providers active now.")

def _step_default_provider() -> str:
    """[3/3] — Let the user pick their default provider."""
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

# ─── Main Entry ───────────────────────────────────────────────────────────────

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
        INIT_FILE.write_text("DevMind initialized.\n", encoding="utf-8")
    except Exception:
        pass

    _print()
    _print("[bold green]✓ Setup complete! Starting DevMind...[/bold green]\n" if console else
           "\n✓ Setup complete! Starting DevMind...\n")
