import sys
from typing import Any, Dict

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

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

console = Console()

def print_header(provider_name: str, model_name: str, mode: str = ""):
    """Print the assistant start panel."""
    content = f"Provider: {provider_name.capitalize()} ({model_name})"
    if mode:
        content += f"\nMode: {mode}"
    panel = Panel(content, title="Nexus-Agent", border_style="cyan")
    console.print(panel)

def create_status(message: str = "Thinking...") -> Any:
    """Create and start a Rich live status spinner."""
    status = console.status(f"[bold cyan]{message}[/bold cyan]", spinner="line")
    status.start()
    return status

def update_status(status: Any, message: str):
    """Update the text of an active status spinner."""
    if status is not None:
        status.update(f"[bold cyan]{message}[/bold cyan]")

def stop_status(status: Any):
    """Cleanly stop and hide the active status spinner."""
    if status is not None:
        status.stop()

def print_tool_call(tool_name: str, args: Dict[str, Any]):
    """Print tool invocation info. For run_code, shows full code with syntax highlighting."""
    if tool_name == "run_code" and "code" in args:
        console.print(f"\n  [bold yellow][ACTION][/bold yellow] {tool_name}(language={args.get('language', 'python')})")
        code = args["code"]
        syntax = Syntax(code, args.get("language", "python"), theme="monokai", line_numbers=True)
        console.print(syntax)
    else:
        args_str = ", ".join(
            f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
            for k, v in args.items()
        )
        console.print(f"\n  [bold yellow][ACTION][/bold yellow] {tool_name}({args_str})")

def print_tool_result(result: str, duration: float, tool_name: str = ""):
    """Print tool execution completion and duration. Shows multi-line preview for run_code."""
    if tool_name == "run_code":
        lines = result.strip().split("\n")
        preview_lines = lines[:15]
        preview = "\n".join(preview_lines)
        if len(lines) > 15:
            preview += f"\n  [dim]... ({len(lines) - 15} more lines)[/dim]"
        console.print(f"  [bold green][OBSERVE][/bold green] Done ({duration:.1f}s)")
        console.print(f"  [dim]{preview}[/dim]")
    else:
        lines = result.strip().split("\n")
        # Show up to 3 lines of preview for non-code tools
        preview_lines = lines[:3]
        preview = " | ".join(l.strip() for l in preview_lines if l.strip())
        if len(preview) > 120:
            preview = preview[:117] + "..."
        if len(lines) > 3:
            preview += f" ... (+{len(lines)-3} lines)"
        console.print(f"  [bold green][OBSERVE][/bold green] Done ({duration:.1f}s) -> [dim]{preview}[/dim]")


def print_thinking(text: str):
    """Print the agent's reasoning step (verbose ReAct trace)."""
    if not text or not text.strip():
        return
    # Strip duplicate leading [THINKING] tag if already present in text
    cleaned_text = text.strip()
    while cleaned_text.upper().startswith("[THINKING]"):
        cleaned_text = cleaned_text[len("[THINKING]"):].strip()
    if not cleaned_text:
        return
    # Show up to 3 lines of reasoning
    lines = [l for l in cleaned_text.split("\n") if l.strip()]
    preview = " ".join(lines[:3])
    if len(preview) > 200:
        preview = preview[:197] + "..."
    console.print(f"\n  [bold magenta][THINKING][/bold magenta] {preview}")

def print_response(text: str):
    """Print final response panel with Markdown formatting."""
    md = Markdown(text)
    panel = Panel(md, title="Response", border_style="green", expand=False)
    console.print("\n", panel)

def print_footer(tokens: int, cost: float, time: float):
    """Print execution statistics footer."""
    if tokens == 0 or cost == 0.0:
        console.print(f"\n  [dim]Tokens: N/A (local)  |  Est. cost: $0.0000 (offline)  |  Time: {time:.1f}s[/dim]\n")
    else:
        console.print(f"\n  [dim]Tokens: {tokens:,}  |  Est. cost: ${cost:.4f}  |  Time: {time:.1f}s[/dim]\n")

def print_error(message: str):
    """Print error message."""
    console.print(f"[bold red][ERROR][/bold red] {message}", style="red")

def print_warn(message: str):
    """Print a warning message (e.g. rate limit fallback)."""
    console.print(f"[bold yellow][WARN][/bold yellow] {message}")

def print_fallback_switch(from_provider: str, to_provider: str, reason: str = ""):
    """Print a clean provider-switch warning on rate limit or API/auth error."""
    msg = f"{from_provider.capitalize()} failed or rate-limited. Switching to {to_provider.capitalize()}..."
    if reason:
        low = reason.lower()
        if "auth" in low or "key" in low or "401" in low or "permission" in low:
            msg = f"{from_provider.capitalize()} auth/API error. Switching to {to_provider.capitalize()}..."
        elif "rate" in low or "429" in low or "quota" in low:
            msg = f"{from_provider.capitalize()} rate limit hit. Switching to {to_provider.capitalize()}..."
    print_warn(msg)
