import sys
from typing import Any, Dict
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

console = Console()

def print_header(provider_name: str, model_name: str, mode: str = ""):
    """Print the assistant start panel."""
    content = f"Provider: {provider_name.capitalize()} ({model_name})"
    if mode:
        content += f"\nMode: {mode}"
    panel = Panel(content, title="Programmer Assistant", border_style="cyan")
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
    """Print tool invocation info."""
    args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in args.items())
    console.print(f"\n  [bold yellow][TOOL][/bold yellow] {tool_name}({args_str})")

def print_tool_result(result: str, duration: float):
    """Print tool execution completion and duration."""
    preview = result.strip().split("\n")[0]
    if len(preview) > 60:
        preview = preview[:57] + "..."
    console.print(f"  [bold green][OK][/bold green] Done ({duration:.1f}s) -> [dim]{preview}[/dim]")

def print_response(text: str):
    """Print final response panel with Markdown formatting."""
    md = Markdown(text)
    panel = Panel(md, title="Response", border_style="green", expand=False)
    console.print("\n", panel)

def print_footer(tokens: int, cost: float, time: float):
    """Print execution statistics footer."""
    console.print(f"\n  [dim]Tokens: {tokens:,}  |  Est. cost: ${cost:.4f}  |  Time: {time:.1f}s[/dim]\n")

def print_error(message: str):
    """Print error message."""
    console.print(f"[bold red][ERROR][/bold red] {message}", style="red")
