import sys
import time
import typer
from typing import Optional, Any, Tuple

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

from ..utils.config import DEFAULT_PROVIDER, ConfigError
from ..agent.memory import ConversationMemory
from ..agent.core import Agent
from . import display

app = typer.Typer(
    help="Nexus-Agent - Autonomous AI Coding Agent",
    invoke_without_command=True,
)

@app.callback()
def main_callback(ctx: typer.Context):
    """Nexus-Agent — Autonomous AI Coding Agent.

    Run without a subcommand to launch the interactive REPL session.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand → launch REPL with verbose ON by default
        ctx.invoke(repl)


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
    elif name_clean in ["gemini", "gemini-lite", "lite"]:
        from ..providers.gemini_provider import GeminiProvider
        return GeminiProvider(), "gemini"
    elif name_clean in ["openai", "gpt", "gpt-4o"]:
        from ..providers.openai_provider import OpenAIProvider
        return OpenAIProvider(), "openai"
    elif name_clean in ["ollama", "local"]:
        import os
        from ..providers.openai_provider import OpenAIProvider
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
        local_model = os.getenv("LOCAL_MODEL", "qwen2.5-coder:7b")
        return OpenAIProvider(model=local_model, base_url=base_url), f"local ({local_model})"
    elif name_clean in ["openrouter", "laguna", "free"]:
        import os
        from ..providers.openai_provider import OpenAIProvider
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if not or_key:
            raise Exception("OPENROUTER_API_KEY not set in .env")
        return OpenAIProvider(
            model="poolside/laguna-m.1:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=or_key,
        ), "OpenRouter (laguna-m.1:free)"
    elif name_clean in ["mock", "demo"]:
        from ..providers.mock_provider import MockProvider
        return MockProvider(), "Demo (mock)"
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
    max_iterations: int = typer.Option(10, "--max-iterations", "-m", help="Max tool iterations per query (default: 10)."),
):
    """Execute a single-turn chat instruction with autonomous tool calling."""
    if not query or not query.strip():
        display.print_error("Query cannot be empty.")
        raise typer.Exit(code=1)
    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations)

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
    verbose: bool = typer.Option(True, "--verbose/--no-verbose", "-v", help="Show verbose ReAct tool trace (default: ON)."),
    no_stream: bool = typer.Option(True, "--no-stream/--stream", help="Disable output streaming (default: OFF for REPL)."),
    max_iterations: int = typer.Option(10, "--max-iterations", "-m", help="Max tool iterations per query (default: 10)."),
):
    """Start an interactive multi-turn REPL chat session."""
    if hasattr(provider, "default"):
        provider = provider.default
    if hasattr(verbose, "default"):
        verbose = bool(verbose.default)
    if hasattr(no_stream, "default"):
        no_stream = bool(no_stream.default)
    if hasattr(max_iterations, "default"):
        max_iterations = int(max_iterations.default)
    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations)

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

                lower_input = user_input.lower()
                # Detect if we're using MockProvider (shows scripted tool traces)
                from ..providers.mock_provider import MockProvider
                _is_mock = isinstance(prov, MockProvider)

                if lower_input == "commit" or lower_input.startswith("commit "):
                    try:
                        commit(provider=provider, yes=False, verbose=verbose, no_stream=no_stream, max_iterations=max_iterations)
                    except typer.Exit:
                        pass
                    continue
                elif lower_input.startswith("chat "):
                    user_input = user_input[5:].strip().strip('"').strip("'")
                elif lower_input.startswith("review "):
                    file_to_rev = user_input[7:].strip().strip('"').strip("'")
                    if _is_mock:
                        # Mock provider: pass filename — it shows scripted THINKING→ACTION→OBSERVE
                        user_input = f"Review {file_to_rev} and tell me if it's good code"
                    else:
                        # Real provider: pre-read file to avoid tool-call JSON leakage
                        try:
                            import os as _os
                            fpath = _os.path.join(_os.getcwd(), file_to_rev) if not _os.path.isabs(file_to_rev) else file_to_rev
                            with open(fpath, "r", encoding="utf-8", errors="replace") as _f:
                                file_contents = _f.read()
                            user_input = (
                                f"Here is the content of '{file_to_rev}':\n\n```\n{file_contents}\n```\n\n"
                                f"Please review this code. Identify bugs, bad practices, missing type hints, "
                                f"missing docstrings, security issues, and suggest improvements. Be concise."
                            )
                        except FileNotFoundError:
                            display.print_error(f"File not found: {file_to_rev}")
                            continue
                elif lower_input.startswith("debug "):
                    parts = user_input[6:].strip().split("--error")
                    file_to_dbg = parts[0].strip().strip('"').strip("'")
                    err_msg = parts[1].strip().strip('"').strip("'") if len(parts) > 1 else "Error reported by user"
                    if _is_mock:
                        # Mock provider: pass filename + error — shows scripted tool trace
                        user_input = f"Debug {file_to_dbg} --error {err_msg}"
                    else:
                        # Real provider: pre-read file to avoid tool-call JSON leakage
                        try:
                            import os as _os
                            fpath = _os.path.join(_os.getcwd(), file_to_dbg) if not _os.path.isabs(file_to_dbg) else file_to_dbg
                            with open(fpath, "r", encoding="utf-8", errors="replace") as _f:
                                file_contents = _f.read()
                            user_input = (
                                f"Here is the content of '{file_to_dbg}':\n\n```\n{file_contents}\n```\n\n"
                                f"The user reports this error:\n{err_msg}\n\n"
                                f"Identify the root cause and provide the fixed version of the code."
                            )
                        except FileNotFoundError:
                            display.print_error(f"File not found: {file_to_dbg}")
                            continue

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
    max_iterations: int = typer.Option(10, "--max-iterations", "-m", help="Max tool iterations per query (default: 10)."),
):
    """Review code quality, security, and potential bugs in a local file."""
    if not file_path or not file_path.strip():
        display.print_error("File path cannot be empty.")
        raise typer.Exit(code=1)
    try:
        from ..agent.tools import get_readonly_tools
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations, tools=get_readonly_tools())
        display.print_header(resolved_name, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))
        query = f"Please perform a STRICTLY READ-ONLY review of the code in '{file_path}'. Use read_file first, analyze for bugs, security issues, and clean code best practices. Do NOT attempt to modify any files."
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
    max_iterations: int = typer.Option(10, "--max-iterations", "-m", help="Max tool iterations per query (default: 10)."),
):
    """Diagnose and fix an error in a local codebase.

    Example:
        agent debug src/app.py --error "AttributeError: 'NoneType' object has no attribute 'stream'"
    """
    if not file_path or not file_path.strip() or not error or not error.strip():
        display.print_error("File path and error message cannot be empty.")
        raise typer.Exit(code=1)
    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations)
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
    max_iterations: int = typer.Option(10, "--max-iterations", "-m", help="Max tool iterations per query (default: 10)."),
):
    """Generate code autonomously and save directly to file.

    Example:
        agent generate "Create an async web scraper using aiohttp" --output scraper.py
    """
    if not prompt or not prompt.strip() or not output or not output.strip():
        display.print_error("Prompt and output path cannot be empty.")
        raise typer.Exit(code=1)
    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations)
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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose ReAct tool trace."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
    max_iterations: int = typer.Option(10, "--max-iterations", "-m", help="Max tool iterations per query (default: 10)."),
):
    """Read git diff, generate a conventional commit message, and commit.

    Example:
        agent commit
        agent commit --yes   # skip confirmation prompt
    """
    import subprocess

    # Quick pre-check: is this even a git repo?
    check = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check.returncode != 0:
        display.print_error("Not inside a git repository.")
        raise typer.Exit(code=1)

    try:
        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations)
        display.print_header(resolved_name, getattr(prov, "model", "unknown"), mode="Commit Mode")

        from ..agent.tools import execute_git_diff
        diff_text = execute_git_diff()
        if not diff_text or "No changes" in diff_text:
            display.print_error("No staged or unstaged changes found to commit.")
            raise typer.Exit(code=1)

        query = (
            f"Here is the git diff:\n```\n{diff_text[:3000]}\n```\n\n"
            "Generate ONE single line conventional commit message (format: type(scope): description). "
            "Keep it under 72 characters. "
            "Reply ONLY with the commit message line itself. Do NOT output JSON, do NOT output explanations or markdown."
        )

        display.print_warn("Generating commit message...")
        start_time = time.time()
        commit_message = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time

        lines = [line.strip().strip('"').strip("'") for line in commit_message.splitlines() if line.strip()]
        clean_lines = [l for l in lines if not l.startswith('{') and not l.startswith('```') and not l.startswith('#') and not l.startswith('Assuming') and not l.lower().startswith('response:')]
        commit_message = clean_lines[0] if clean_lines else (lines[-1] if lines else "chore: update codebase")

        typer.echo(f"\n  Generated message: {commit_message}")

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
            typer.echo("\n  [OK] Done!")

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
        typer.echo("Nexus-Agent CLI v2.2.1")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        # Run onboarding wizard on first launch
        try:
            from .onboarding import run_if_first_time
            run_if_first_time()
        except Exception:
            pass  # Never block startup on onboarding errors
        repl(provider=DEFAULT_PROVIDER, verbose=False, no_stream=False, max_iterations=10)

if __name__ == "__main__":
    app()
