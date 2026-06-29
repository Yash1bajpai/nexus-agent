import time
import typer
from typing import Optional, Any, Tuple
from ..utils.config import DEFAULT_PROVIDER, ConfigError
from ..agent.memory import ConversationMemory
from ..agent.core import Agent
from . import display

app = typer.Typer(help="DevMind — Autonomous AI Coding Agent")

def get_provider_instance(provider_name: Any) -> Tuple[Any, str]:
    """
    Factory to return (provider_instance, resolved_name).
    Returns a resolved name so print_header() always shows the actual provider,
    not a stale or invalid input string (fixes Bug 1).
    """
    if hasattr(provider_name, "default"):
        provider_name = provider_name.default
    if not isinstance(provider_name, str):
        provider_name = str(provider_name or DEFAULT_PROVIDER)
    name_clean = provider_name.lower().strip()

    if name_clean == "anthropic":
        from ..providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(), "anthropic"
    elif name_clean == "gemini":
        from ..providers.gemini_provider import GeminiProvider
        return GeminiProvider(), "gemini"
    elif name_clean in ["openai", "gpt", "gpt-4o"]:
        from ..providers.openai_provider import OpenAIProvider
        return OpenAIProvider(), "openai"
    elif name_clean == "auto":
        from ..providers.fallback_provider import FallbackProvider
        fb = FallbackProvider(start_provider=DEFAULT_PROVIDER)
        fb._warn_fn = display.print_fallback_switch
        return fb, f"auto ({fb._current_name})"
    else:
        display.print_warn(f"Unknown provider '{provider_name}'. Using Anthropic fallback.")
        from ..providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(), "anthropic"

@app.command()
def chat(
    query: str = typer.Argument(..., help="The coding question or instruction for the agent."),
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p", help="LLM provider backend (gemini/anthropic/openai/auto)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose ReAct tool trace."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
):
    """Execute a single-turn chat instruction with autonomous tool calling."""
    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose)

        display.print_header(resolved_name, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))

        start_time = time.time()
        response_text = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time

        if no_stream:
            display.print_response(response_text)
        display.print_footer(agent.total_tokens, agent.estimated_cost, duration)

    except ConfigError as e:
        display.print_error(str(e))
        raise typer.Exit(code=1)
    except Exception as e:
        display.print_error(f"Fatal execution error: {str(e)}")
        raise typer.Exit(code=1)

@app.command()
def repl(
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p", help="LLM provider backend (gemini/anthropic/openai/auto)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose ReAct tool trace."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
):
    """Start an interactive multi-turn REPL chat session."""
    if hasattr(provider, "default"):
        provider = provider.default
    if hasattr(verbose, "default"):
        verbose = bool(verbose.default)
    if hasattr(no_stream, "default"):
        no_stream = bool(no_stream.default)
    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose)

        display.print_header(resolved_name, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))
        typer.echo("Type 'exit' or 'quit' to end the session.\n")

        while True:
            try:
                user_input = typer.prompt(">> You").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    typer.echo("Ending session. Goodbye!")
                    break

                start_time = time.time()
                response_text = agent.run(user_input, stream=not no_stream)
                duration = time.time() - start_time

                if no_stream:
                    display.print_response(response_text)
                display.print_footer(agent.total_tokens, agent.estimated_cost, duration)
            except KeyboardInterrupt:
                typer.echo("\nSession interrupted. Type 'exit' to quit.")
                continue

    except ConfigError as e:
        display.print_error(str(e))
        raise typer.Exit(code=1)
    except Exception as e:
        display.print_error(f"Fatal execution error: {str(e)}")
        raise typer.Exit(code=1)

@app.command()
def review(
    file_path: str = typer.Argument(..., help="Path to code file to review."),
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p", help="LLM provider backend."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose ReAct tool trace."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
):
    """Review code quality, security, and potential bugs in a local file."""
    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose)
        display.print_header(resolved_name, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))
        query = f"Please review the code in '{file_path}'. Use read_file first, analyze for bugs, security issues, and clean code best practices."
        start_time = time.time()
        response_text = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time
        if no_stream:
            display.print_response(response_text)
        display.print_footer(agent.total_tokens, agent.estimated_cost, duration)
    except Exception as e:
        display.print_error(str(e))
        raise typer.Exit(code=1)

@app.command()
def debug(
    file_path: str = typer.Argument(..., help="Path to problematic code file."),
    error: str = typer.Option(..., "--error", "-e", help="Error message or traceback."),
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p", help="LLM provider backend."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose ReAct tool trace."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
):
    """Diagnose and fix an error in a local codebase.

    Example:
        agent debug src/app.py --error "AttributeError: 'NoneType' object has no attribute 'stream'"
    """
    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose)
        display.print_header(resolved_name, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))
        query = f"Debug '{file_path}' given this error traceback:\n{error}\nUse read_file to inspect it, explain the root cause, and use write_file to fix it."
        start_time = time.time()
        response_text = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time
        if no_stream:
            display.print_response(response_text)
        display.print_footer(agent.total_tokens, agent.estimated_cost, duration)
    except Exception as e:
        display.print_error(str(e))
        raise typer.Exit(code=1)

@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Instruction of what code to generate."),
    output: str = typer.Option(..., "--output", "-o", help="Target file path to write generated code."),
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p", help="LLM provider backend."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose ReAct tool trace."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
):
    """Generate code autonomously and save directly to file.

    Example:
        agent generate "Create an async web scraper using aiohttp" --output scraper.py
    """
    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose)
        display.print_header(resolved_name, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))
        query = f"Generate code based on this instruction: '{prompt}'. Write the final production code to '{output}' using write_file."
        start_time = time.time()
        response_text = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time
        if no_stream:
            display.print_response(response_text)
        display.print_footer(agent.total_tokens, agent.estimated_cost, duration)
    except Exception as e:
        display.print_error(str(e))
        raise typer.Exit(code=1)

@app.command()
def commit(
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p", help="LLM provider backend."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm without prompting."),
):
    """Read git diff, generate a conventional commit message, and commit.

    Example:
        agent commit
        agent commit --yes   # skip confirmation prompt
    """
    import subprocess

    # Quick pre-check: is this even a git repo?
    check = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True)
    if check.returncode != 0:
        display.print_error("Not inside a git repository.")
        raise typer.Exit(code=1)

    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=False)
        display.print_header(resolved_name, getattr(prov, "model", "unknown"), mode="Commit Mode")

        query = (
            "Use the git_diff tool to read the current diff. "
            "Then generate a single conventional commit message (format: type(scope): description). "
            "Keep it under 72 characters. "
            "Reply with ONLY the commit message string — no explanation, no quotes, no markdown."
        )

        display.print_warn("Reading git diff and generating commit message...")
        start_time = time.time()
        commit_message = agent.run(query, stream=False)
        duration = time.time() - start_time

        commit_message = commit_message.strip().strip('"').strip("'")

        typer.echo(f"\n  Generated message: [bold]{commit_message}[/bold]" if False else
                   f"\n  Generated message: {commit_message}")

        if not yes:
            confirmed = typer.confirm("\n  Commit with this message?", default=True)
            if not confirmed:
                typer.echo("  Commit aborted.")
                raise typer.Exit(code=0)

        # Execute the commit
        from ..agent.tools import execute_git_commit
        result = execute_git_commit(commit_message)

        if result.startswith("ERROR"):
            display.print_error(result)
            raise typer.Exit(code=1)
        else:
            display.print_warn(result.replace("Committed successfully:\n", ""))
            typer.echo("\n  [green]✓ Committed![/green]" if False else "\n  ✓ Done!")

        display.print_footer(agent.total_tokens, agent.estimated_cost, duration)

    except Exception as e:
        display.print_error(f"Commit failed: {str(e)}")
        raise typer.Exit(code=1)

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version information.")
):
    if version:
        typer.echo("DevMind CLI v2.2.0")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        # Run onboarding wizard on first launch
        try:
            from .onboarding import run_if_first_time
            run_if_first_time()
        except Exception:
            pass  # Never block startup on onboarding errors
        repl(provider=DEFAULT_PROVIDER, verbose=False, no_stream=False)

if __name__ == "__main__":
    app()
