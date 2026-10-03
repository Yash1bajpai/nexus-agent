import sys
import time
import re
from pathlib import Path
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

from typer.core import TyperGroup
import click

class NaturalAgentGroup(TyperGroup):
    def get_command(self, ctx, cmd_name):
        rv = super().get_command(ctx, cmd_name)
        if rv is not None:
            return rv
        return super().get_command(ctx, "chat")

    def resolve_command(self, ctx, args):
        if not args:
            return super().resolve_command(ctx, args)
        cmd_name = args[0]
        # Check if the first argument is one of our registered subcommands
        if cmd_name not in self.commands and cmd_name not in ["--help", "-h", "--version", "--install-completion", "--show-completion"]:
            cmd = super().get_command(ctx, "chat")
            return "chat", cmd, args
        return super().resolve_command(ctx, args)

app = typer.Typer(
    help="Nexus-Agent - Autonomous AI Coding Agent",
    invoke_without_command=True,
    cls=NaturalAgentGroup,
)


def _fit_local_model(prov, context_size, kwargs):
    """Keep an explicit model choice; otherwise pick by free RAM and ask before downloading."""
    import os
    from ..utils import model_select
    from ..providers.local_provider import LocalProvider
    if os.getenv("NEXUS_AGENT_MODEL_REPO") or os.getenv("NEXUS_AGENT_MODEL_FILENAME"):
        return prov
    from .onboarding import _is_android
    ctx = context_size or getattr(prov, "context_size", 4096)
    free = model_select.available_ram_gb()
    choice = model_select.pick_model(free, ctx, android=_is_android())
    if model_select.is_cached(choice["repo"], choice["filename"]):
        if (choice["repo"], choice["filename"]) == (prov.model_id, prov.filename):
            return prov
        return LocalProvider(model_id=choice["repo"], filename=choice["filename"], **kwargs)
    if model_select.is_cached(prov.model_id, prov.filename):
        display.print_warn(f"Using the downloaded model {prov.filename}. {choice['reason']}")
        return prov
    chosen = model_select.confirm_download(choice, ctx, out=display.print_info)
    if chosen is None:
        display.print_error("Download cancelled. Run again when ready, or set NEXUS_AGENT_MODEL_REPO and NEXUS_AGENT_MODEL_FILENAME to choose a model.")
        raise typer.Exit(code=1)
    return LocalProvider(model_id=chosen["repo"], filename=chosen["filename"], **kwargs)


def _make_local_provider(context_size: Optional[int] = None) -> Tuple[Any, str]:
    """Build the default local provider and pre-flight the inference engine.

    Downloads/verifies llama-server up front (idempotent, ~50 MB once) so a
    missing engine never fails the LFM model mid-query — the failure the
    pre-flight prevents is only discoverable after the agent loop starts.
    """
    from ..providers.local_provider import LocalProvider, ensure_llama_server_binary
    kwargs = {"context_size": context_size} if context_size is not None else {}
    prov = _fit_local_model(LocalProvider(**kwargs), context_size, kwargs)
    prov.setup_model()
    try:
        ensure_llama_server_binary(verbose=False)
    except Exception:
        pass
    return prov, f"Local ({prov.filename})"


def get_provider_instance(provider_name: Any, context_size: Optional[int] = None) -> Tuple[Any, str]:
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
    if context_size is not None:
        if name_clean not in {"local", "liquid", "lfm", "default", "demo"}:
            raise ConfigError("--context-size requires --provider local (not cloud, ollama, or auto).")
        display.print_warn(
            f"Local context: {context_size} tokens. Larger windows use more RAM; "
            "stay within your model's supported context. This does not compact history."
        )

    if name_clean == "anthropic":
        from ..providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(), "anthropic"
    elif name_clean in ["gemini", "gemini-lite", "lite"]:
        from ..providers.gemini_provider import GeminiProvider
        return GeminiProvider(), "gemini"
    elif name_clean in ["openai", "gpt", "gpt-4o"]:
        from ..providers.openai_provider import OpenAIProvider
        return OpenAIProvider(), "openai"
    elif name_clean in ["local", "liquid", "lfm", "default", "demo"]:
        return _make_local_provider(context_size=context_size)
    elif name_clean in ["ollama"]:
        import os
        from ..providers.openai_provider import OpenAIProvider
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
        local_model = os.getenv("LOCAL_MODEL", "llama3.2:3b")
        return OpenAIProvider(model=local_model, base_url=base_url), f"ollama ({local_model})"
    elif name_clean in ["openrouter", "laguna", "free"]:
        import os
        from ..providers.openai_provider import OpenAIProvider
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if not or_key:
            raise Exception("OPENROUTER_API_KEY not set in .env")
        model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
        return OpenAIProvider(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=or_key,
        ), f"OpenRouter ({model.split('/')[-1]})"
    elif name_clean == "auto":
        from ..providers.fallback_provider import FallbackProvider
        fb = FallbackProvider(start_provider=DEFAULT_PROVIDER)
        fb._warn_fn = display.print_fallback_switch
        return fb, f"auto ({fb._current_name})"
    else:
        display.print_warn(f"Unknown provider '{provider_name}'. Using local fallback.")
        return _make_local_provider(context_size=context_size)


@app.command()
def chat(
    query: Optional[str] = typer.Argument(None, help="The coding question or instruction. Leave it out to open the interactive REPL."),
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p", help="LLM provider backend (local/gemini/anthropic/openai/auto)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose ReAct tool trace."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
    max_iterations: int = typer.Option(10, "--max-iterations", "-m", help="Max tool iterations per query (default: 10)."),
    persist: bool = typer.Option(False, "--persist/--no-persist", help="Persist conversation history across sessions (SQLite-backed)."),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID for conversation history (defaults to workspace ID)."),
    context_size: Optional[int] = typer.Option(None, "--context-size", min=512, help="Local llama context window in tokens (default: NEXUS_CONTEXT_SIZE or 4096). Larger windows use more RAM."),
):
    """Execute a single-turn chat instruction with autonomous tool calling."""
    if hasattr(persist, "default"):
        persist = bool(persist.default)
    if hasattr(session, "default"):
        session = session.default
    if hasattr(context_size, "default"):
        context_size = context_size.default
    if query is None or (hasattr(query, "default") and query.default is None):
        # `nexus-agent chat` with no question behaves like `nexus-agent repl`.
        repl(provider=provider, verbose=True, no_stream=True, max_iterations=max_iterations,
             persist=persist, session=session, context_size=context_size)
        return
    if not query.strip():
        display.print_error("Query cannot be empty.")
        raise typer.Exit(code=1)
    if max_iterations <= 0:
        display.print_error("max-iterations must be a positive integer > 0.")
        raise typer.Exit(code=1)
    try:
        prov, resolved_name = get_provider_instance(provider, context_size=context_size) if context_size is not None else get_provider_instance(provider)
        if persist:
            try:
                from ..agent.persistence import SQLiteMemory
                memory = SQLiteMemory(session_id=session)
            except Exception as e:
                display.print_warn(f"SQLite persistence unavailable ({e}), falling back to in-memory.")
                memory = ConversationMemory()
        else:
            memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations)

        model_display = getattr(prov, "model", getattr(prov, "model_id", "liquid-lfm"))
        display.print_header(resolved_name, model_display, mode=getattr(agent, "mode_str", ""))

        start_time = time.time()
        response_text = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time

        if no_stream:
            display.print_response(response_text)
        display.print_footer(agent.total_tokens, agent.estimated_cost, duration)

    except ConfigError as e:
        display.print_error(str(e))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        display.print_error(f"Fatal execution error: {str(e)}")
        raise typer.Exit(code=1)

def _build_repl_completer():
    """Build prompt_toolkit completer for slash commands and workspace file @mentions."""
    try:
        from prompt_toolkit.completion import Completer, Completion
    except ImportError:
        return None

    class NexusReplCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            slash_cmds = [
                ("/help", "Show available REPL commands"),
                ("/clear", "Clear conversation memory"),
                ("/context", "Show context window, usage and recovery guidance"),
                ("/compact", "Shrink older history to free context"),
                ("/commit", "Review staged changes & commit"),
                ("/review", "Review code file: /review <file>"),
                ("/debug", "Debug error in file: /debug <file> -e <err>"),
                ("/history", "View session message history"),
                ("/sessions", "List stored persistent sessions"),
                ("/pull-model", "Pre-download local model"),
                ("/exit", "Exit session"),
            ]
            if text.startswith("/"):
                for cmd, desc in slash_cmds:
                    if cmd.startswith(text):
                        yield Completion(cmd, start_position=-len(text), display_meta=desc)
                return

            last_word = text.split()[-1] if text.split() else ""
            if last_word.startswith("@"):
                prefix = last_word[1:].lower()
                cwd = Path.cwd()
                ignored_dirs = {
                    ".git", ".venv", "venv", "__pycache__", "node_modules",
                    "build", "dist", ".idea", ".vscode", ".pytest_cache", ".nexus-agent"
                }
                matches = 0
                max_matches = 50
                try:
                    import os
                    for root, dirs, files in os.walk(cwd):
                        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ignored_dirs]
                        rel_root = Path(root).relative_to(cwd)
                        if len(rel_root.parts) > 4:
                            dirs.clear()
                            continue
                        for f in sorted(files):
                            if f.startswith(".") or f.endswith((".pyc", ".bak", ".pyd", ".pyo")):
                                continue
                            rel_posix = (rel_root / f).as_posix() if str(rel_root) != "." else f
                            if rel_posix.lower().startswith(prefix) or f.lower().startswith(prefix):
                                yield Completion(f"@{rel_posix}", start_position=-len(last_word), display=rel_posix)
                                matches += 1
                                if matches >= max_matches:
                                    return
                except Exception:
                    pass

    return NexusReplCompleter()

@app.command()
def repl(
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p", help="LLM provider backend (local/gemini/anthropic/openai/auto)."),
    verbose: bool = typer.Option(True, "--verbose/--no-verbose", "-v", help="Show verbose ReAct tool trace (default: ON)."),
    no_stream: bool = typer.Option(True, "--no-stream/--stream", help="Disable output streaming (by default OFF in REPL mode for clean multi-turn prompts)."),
    max_iterations: int = typer.Option(10, "--max-iterations", "-m", help="Max tool iterations per query (default: 10)."),
    persist: bool = typer.Option(False, "--persist/--no-persist", help="Persist conversation history across sessions (SQLite-backed)."),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID for conversation history (defaults to workspace ID)."),
    context_size: Optional[int] = typer.Option(None, "--context-size", min=512, help="Local llama context window in tokens (default: NEXUS_CONTEXT_SIZE or 4096). Larger windows use more RAM."),
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
    if hasattr(persist, "default"):
        persist = bool(persist.default)
    if hasattr(session, "default"):
        session = session.default
    if hasattr(context_size, "default"):
        context_size = context_size.default
    if max_iterations <= 0:
        display.print_error("max-iterations must be a positive integer > 0.")
        raise typer.Exit(code=1)
    try:
        prov, resolved_name = get_provider_instance(provider, context_size=context_size) if context_size is not None else get_provider_instance(provider)
        if persist:
            try:
                from ..agent.persistence import SQLiteMemory
                memory = SQLiteMemory(session_id=session)
                typer.echo(f"  [i]Persistent mode: session '{memory.session_id}' saved to ~/.nexus-agent/history.db[/i]\n")
            except Exception as e:
                display.print_warn(f"SQLite persistence unavailable ({e}), falling back to in-memory.")
                memory = ConversationMemory()
        else:
            memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations)

        model_display = getattr(prov, "model", getattr(prov, "model_id", "liquid-lfm"))
        display.print_header(resolved_name, model_display, mode=getattr(agent, "mode_str", ""))
        typer.echo("Type /help for available commands, or /exit to quit.\n")

        session = None
        if sys.stdin.isatty():
            try:
                from prompt_toolkit import PromptSession
                from prompt_toolkit.history import FileHistory
                history_dir = Path.home() / ".nexus-agent"
                history_dir.mkdir(parents=True, exist_ok=True)
                history_file = str(history_dir / "repl_history")
                completer = _build_repl_completer()
                session = PromptSession(
                    history=FileHistory(history_file),
                    completer=completer,
                )
            except Exception:
                session = None

        while True:
            try:
                if session is not None and sys.stdin.isatty():
                    try:
                        user_input = session.prompt(">> You: ").strip()
                    except KeyboardInterrupt:
                        typer.echo("^C")
                        continue
                else:
                    user_input = typer.prompt(">> You").strip()

                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q", "/exit", "/quit", "/q"]:
                    typer.echo("Ending session. Goodbye!")
                    break

                lower_input = user_input.lower()

                if lower_input in ("/clear", "clear"):
                    agent.memory.clear()
                    display.print_info("Conversation memory cleared.")
                    continue
                elif lower_input == "/context":
                    window = getattr(prov, "context_size", None)
                    if window is None:
                        display.print_info("Context window is managed by this provider; --context-size applies only to local llama engines.")
                    else:
                        display.print_info(
                            f"Local context window: {window} tokens (includes instructions, tools, history and output). "
                            "Restart with --context-size <tokens> to change it. Larger windows use more RAM. "
                            "Old turns are dropped and big items cut automatically when a request would not fit; "
                            "/compact shrinks stored history, /clear clears it (including a persistent session)."
                        )
                        from ..agent import context as _ctx
                        display.print_info(_ctx.meter(agent.context_used(), window))
                    continue
                elif lower_input == "/compact":
                    before, after, info = agent.compact()
                    if not info["removed_messages"]:
                        display.print_info("Nothing to compact yet (only the last 2 turns are kept as they are).")
                    else:
                        display.print_info(f"Compacted {info['turns_summarized']} earlier turn(s): history ~{before} -> ~{after} tokens.")
                    continue
                elif lower_input in ("/help", "help"):
                    display.print_info(
                        "Available REPL commands:\n"
                        "  /clear          - Clear conversation context\n"
                        "  /context        - Show context window, usage and recovery guidance\n"
                        "  /compact        - Shrink older history to free context\n"
                        "  /commit         - Review diff & commit changes\n"
                        "  /review <file>  - Review a source file\n"
                        "  /debug <file>   - Debug an error in a file\n"
                        "  /history        - Show conversation history\n"
                        "  /sessions       - List stored persistent sessions\n"
                        "  /pull-model     - Download local model weights\n"
                        "  /exit, /quit    - Exit the session\n"
                        "  @filename       - Reference file context directly"
                    )
                    continue
                elif lower_input in ("/history", "history"):
                    msgs = agent.memory.get()
                    display.print_info(f"Session history ({len(msgs)} messages):")
                    for m in msgs:
                        role = m.get("role", "unknown")
                        content = str(m.get("content", ""))[:120]
                        typer.echo(f"  [{role.upper()}]: {content}...")
                    continue
                elif lower_input in ("/sessions", "sessions"):
                    try:
                        from ..agent.persistence import SQLiteMemory, get_workspace_session_id
                        sess_list = SQLiteMemory.list_sessions()
                        if not sess_list:
                            display.print_info("No persistent sessions found.")
                        else:
                            curr_sid = getattr(agent.memory, "session_id", get_workspace_session_id())
                            display.print_info(f"Stored persistent sessions ({len(sess_list)}):")
                            for s in sess_list:
                                sid = s["session_id"]
                                count = s["message_count"]
                                active = s["last_active"] or "N/A"
                                current = " (active)" if sid == curr_sid else ""
                                typer.echo(f"  - {sid}{current}: {count} messages (last active: {active})")
                    except Exception as e:
                        display.print_error(f"Failed to list sessions: {e}")
                    continue

                request_tools = None
                if user_input.startswith("/"):
                    user_input = user_input[1:].strip()
                    lower_input = user_input.lower()

                if lower_input == "commit" or lower_input.startswith("commit "):
                    try:
                        commit(provider=provider, yes=False, verbose=verbose, no_stream=no_stream, max_iterations=max_iterations)
                    except typer.Exit:
                        pass
                    continue
                elif lower_input == "pull-model" or lower_input.startswith("pull-model"):
                    try:
                        pull_model_cmd()
                    except typer.Exit:
                        pass
                    continue
                elif lower_input.startswith("generate "):
                    parts = user_input[9:].strip().rsplit("--output", 1)
                    if len(parts) == 1:
                        parts = user_input[9:].strip().rsplit("-o", 1)
                    prompt_str = parts[0].strip().strip('"').strip("'")
                    out_str = parts[1].strip().strip('"').strip("'") if len(parts) > 1 else "generated_code.py"
                    try:
                        generate(prompt=prompt_str, output=out_str, provider=provider, verbose=verbose, no_stream=no_stream, max_iterations=max_iterations)
                    except typer.Exit:
                        pass
                    continue
                elif lower_input.startswith("chat "):
                    user_input = user_input[5:].strip().strip('"').strip("'")
                elif lower_input.startswith("review "):
                    from ..agent.tools import get_readonly_tools
                    request_tools = get_readonly_tools()
                    file_to_rev = user_input[7:].strip().strip('"').strip("'")
                    try:
                        from ..agent.tools import execute_read_file
                        file_contents = execute_read_file(file_to_rev)
                        if file_contents.startswith("ERROR:"):
                            display.print_error(file_contents)
                            continue
                        user_input = (
                            f"Here is the content of '{file_to_rev}':\n\n```\n{file_contents}\n```\n\n"
                            f"Please review this code. Identify bugs, bad practices, missing type hints, "
                            f"missing docstrings, security issues, and suggest improvements. Be concise."
                        )
                    except FileNotFoundError:
                        display.print_error(f"File not found: {file_to_rev}")
                        continue
                elif lower_input.startswith("debug "):
                    parts = re.split(r'--error\s+|-e\s+', user_input[6:].strip(), maxsplit=1)
                    file_to_dbg = parts[0].strip().strip('"').strip("'")
                    err_msg = parts[1].strip().strip('"').strip("'") if len(parts) > 1 else "Error reported by user"
                    try:
                        from ..agent.tools import execute_read_file
                        file_contents = execute_read_file(file_to_dbg)
                        if file_contents.startswith("ERROR:"):
                            display.print_error(file_contents)
                            continue
                        user_input = (
                            f"Here is the content of '{file_to_dbg}':\n\n```\n{file_contents}\n```\n\n"
                            f"The user reports this error:\n{err_msg}\n\n"
                            f"Identify the root cause and provide the fixed version of the code."
                        )
                    except FileNotFoundError:
                        display.print_error(f"File not found: {file_to_dbg}")
                        continue

                start_time = time.time()
                original_tools = agent.tools
                if request_tools is not None:
                    agent.tools = request_tools
                try:
                    response_text = agent.run(user_input, stream=not no_stream)
                finally:
                    agent.tools = original_tools
                duration = time.time() - start_time

                if no_stream:
                    display.print_response(response_text)
                display.print_footer(agent.total_tokens, agent.estimated_cost, duration)
                if getattr(agent, "context_window", None):
                    from ..agent import context as _ctx
                    typer.echo(_ctx.meter(agent.context_used(), agent.context_window))
            except (KeyboardInterrupt, EOFError):
                typer.echo("\nSession ended. Goodbye!")
                break
            except Exception as e:
                display.print_error(f"Error during request: {str(e)}")
                typer.echo("  (Session continues — try another query or 'exit' to quit.)\n")
                continue

    except ConfigError as e:
        display.print_error(str(e))
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
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
    if max_iterations <= 0:
        display.print_error("max-iterations must be a positive integer > 0.")
        raise typer.Exit(code=1)
    try:
        from ..agent.tools import get_readonly_tools, _validate_workspace_path
        validated = _validate_workspace_path(file_path)
        if isinstance(validated, str) and validated.startswith("ERROR:"):
            display.print_error(validated)
            raise typer.Exit(code=1)
        if not Path(file_path).exists():
            display.print_error(f"File not found: {file_path}")
            raise typer.Exit(code=1)

        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations, tools=get_readonly_tools())
        model_display = getattr(prov, "model", getattr(prov, "model_id", "liquid-lfm"))
        display.print_header(resolved_name, model_display, mode=getattr(agent, "mode_str", ""))
        query = f"Please perform a STRICTLY READ-ONLY review of the code in '{file_path}'. Use read_file first, analyze for bugs, security issues, and clean code best practices. Do NOT attempt to modify any files."
        start_time = time.time()
        response_text = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time
        if no_stream:
            display.print_response(response_text)
        display.print_footer(agent.total_tokens, agent.estimated_cost, duration)
    except typer.Exit:
        raise
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
    """Diagnose and fix an error in a local codebase."""
    if not file_path or not file_path.strip() or not error or not error.strip():
        display.print_error("File path and error message cannot be empty.")
        raise typer.Exit(code=1)
    if max_iterations <= 0:
        display.print_error("max-iterations must be a positive integer > 0.")
        raise typer.Exit(code=1)
    try:
        from ..agent.tools import _validate_workspace_path
        validated = _validate_workspace_path(file_path)
        if isinstance(validated, str) and validated.startswith("ERROR:"):
            display.print_error(validated)
            raise typer.Exit(code=1)
        if not Path(file_path).exists():
            display.print_error(f"File not found: {file_path}")
            raise typer.Exit(code=1)

        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations)
        model_display = getattr(prov, "model", getattr(prov, "model_id", "liquid-lfm"))
        display.print_header(resolved_name, model_display, mode=getattr(agent, "mode_str", ""))
        query = f"Debug '{file_path}' given this error traceback:\n{error}\nUse read_file to inspect it carefully, explain the root cause, and if appropriate provide the fixed code."
        start_time = time.time()
        response_text = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time
        if no_stream:
            display.print_response(response_text)
        display.print_footer(agent.total_tokens, agent.estimated_cost, duration)
    except typer.Exit:
        raise
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
    """Generate code autonomously and save directly to file."""
    if not prompt or not prompt.strip() or not output or not output.strip():
        display.print_error("Prompt and output path cannot be empty.")
        raise typer.Exit(code=1)
    if max_iterations <= 0:
        display.print_error("max-iterations must be a positive integer > 0.")
        raise typer.Exit(code=1)
    try:
        from ..agent.tools import _validate_workspace_path
        validated = _validate_workspace_path(output)
        if isinstance(validated, str) and validated.startswith("ERROR:"):
            display.print_error(validated)
            raise typer.Exit(code=1)

        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations)
        model_display = getattr(prov, "model", getattr(prov, "model_id", "liquid-lfm"))
        display.print_header(resolved_name, model_display, mode=getattr(agent, "mode_str", ""))
        query = f"Generate code based on this instruction: '{prompt}'. Write the final production code to '{output}' using write_file."
        start_time = time.time()
        response_text = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time
        if no_stream:
            display.print_response(response_text)
        if not Path(output).exists():
            display.print_error(f"Error: Target file '{output}' was not generated on disk.")
            raise typer.Exit(code=1)
        else:
            typer.echo(f"\n✅ Verified generated file created: {output}\n")
        display.print_footer(agent.total_tokens, agent.estimated_cost, duration)
    except typer.Exit:
        raise
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
    """Read git diff, generate a conventional commit message, and commit."""
    import subprocess

    check = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check.returncode != 0:
        display.print_error("Not inside a git repository.")
        raise typer.Exit(code=1)

    try:
        from ..agent.tools import execute_git_diff
        diff_text = execute_git_diff()
        if not diff_text or "No changes" in diff_text:
            display.print_error("No staged or unstaged changes found to commit.")
            raise typer.Exit(code=1)

        prov, resolved_name = get_provider_instance(provider)
        memory = ConversationMemory()
        from ..agent.tools import get_readonly_tools
        agent = Agent(provider=prov, memory=memory, verbose=verbose, max_iterations=max_iterations, tools=get_readonly_tools())
        model_display = getattr(prov, "model", getattr(prov, "model_id", "liquid-lfm"))
        display.print_header(resolved_name, model_display, mode="Commit Mode")

        query = (
            f"Please inspect the pending changes and produce a single-line conventional commit message.\n"
            f"Diff summary:\n```\n{diff_text[:3000]}\n```\n\n"
            "Format: type(scope): description. Under 72 chars. Output ONLY the commit message string."
        )

        display.print_warn("Generating commit message...")
        start_time = time.time()
        commit_message = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time

        lines = [line.strip().strip('"').strip("'") for line in commit_message.splitlines() if line.strip()]
        clean_lines = [
            l for l in lines
            if not l.startswith('{')
            and not l.startswith('```')
            and not l.startswith('#')
            and not l.startswith('Assuming')
            and not l.lower().startswith('response:')
            and not l.upper().startswith('[THINKING]')
            and not l.upper().startswith('[ACTION]')
            and not l.upper().startswith('[OBSERVE]')
            and not l.upper().startswith('[REASONING]')
        ]
        commit_message = clean_lines[0] if clean_lines else (lines[-1] if lines else "chore: update codebase")

        _failure_markers = ("max tool iterations", "fatal execution error", "execution failed", "timed out")
        if not commit_message.strip() or any(marker in commit_message.lower() for marker in _failure_markers):
            display.print_error("Could not generate a valid commit message. Commit aborted.")
            raise typer.Exit(code=1)

        typer.echo(f"\n  Generated message: {commit_message}")

        if not yes:
            confirmed = typer.confirm("\n  Commit with this message?", default=True)
            if not confirmed:
                typer.echo("  Commit aborted.")
                raise typer.Exit(code=0)

        from ..agent.tools import execute_git_commit
        result = execute_git_commit(commit_message, approved=True)

        if result.startswith("ERROR"):
            display.print_error(result)
            raise typer.Exit(code=1)
        else:
            display.print_warn(result.replace("Committed successfully:\n", ""))
            typer.echo("\n  [OK] Done!")

        display.print_footer(agent.total_tokens, agent.estimated_cost, duration)

    except typer.Exit:
        raise
    except Exception as e:
        display.print_error(f"Commit failed: {str(e)}")
        raise typer.Exit(code=1)

@app.command("pull-model")
def pull_model_cmd(
    repo: str = typer.Option("", "--repo", "-r", help="HuggingFace GGUF repo id to pull from (default: built-in Liquid model or NEXUS_AGENT_MODEL_REPO)."),
    file: str = typer.Option("", "--file", "-f", help="GGUF filename inside the repo (default: built-in quant or NEXUS_AGENT_MODEL_FILENAME)."),
):
    """Download or verify a local GGUF model from any HuggingFace repo."""
    import os
    from ..providers.local_provider import LocalProvider, _DEFAULT_REPO, _DEFAULT_FILENAME
    repo = repo.strip() or os.getenv("NEXUS_AGENT_MODEL_REPO", _DEFAULT_REPO)
    file = file.strip() or os.getenv("NEXUS_AGENT_MODEL_FILENAME", _DEFAULT_FILENAME)
    typer.echo(f"\n🚀 Nexus-Agent — Pulling local model")
    typer.echo(f"   repo: {repo}")
    typer.echo(f"   file: {file}")
    prov = LocalProvider(model_id=repo, filename=file)
    try:
        path = prov.setup_model(verify_download=True)
        typer.echo(f"\n✅ Local Model Ready at: {path}\n")
        typer.echo("To use it as your default local model, add to ~/.nexus-agent/.env:")
        typer.echo(f"  NEXUS_AGENT_MODEL_REPO={repo}")
        typer.echo(f"  NEXUS_AGENT_MODEL_FILENAME={file}")
    except Exception as e:
        display.print_error(f"Failed to download local model: {e}")
        raise typer.Exit(code=1)

    # Also pre-install the llama-server engine (skipped on ARM Linux / if present)
    try:
        from ..providers.local_provider import ensure_llama_server_binary
        ensure_llama_server_binary()
    except Exception:
        pass


@app.command("sessions")
def sessions_cmd(
    delete: Optional[str] = typer.Option(None, "--delete", "-d", help="Delete a specific session by ID, or 'all' to delete all sessions."),
):
    """List or delete stored persistent chat sessions."""
    from ..agent.persistence import SQLiteMemory, get_workspace_session_id
    if delete:
        if delete.lower() == "all":
            all_s = SQLiteMemory.list_sessions()
            for s in all_s:
                SQLiteMemory.delete_session(s["session_id"])
            display.print_info(f"Cleared all {len(all_s)} persistent session(s).")
        else:
            SQLiteMemory.delete_session(delete)
            display.print_info(f"Deleted session '{delete}'.")
        return

    sessions_list = SQLiteMemory.list_sessions()
    if not sessions_list:
        typer.echo("\nNo persistent sessions found.")
        typer.echo("Run 'nexus repl --persist' or 'nexus chat --persist \"...\"' to record sessions.\n")
        return

    curr_ws = get_workspace_session_id()
    typer.echo(f"\nPersistent Sessions ({len(sessions_list)} total):")
    typer.echo(f"Current workspace session: {curr_ws}\n")
    for s in sessions_list:
        sid = s["session_id"]
        count = s["message_count"]
        active = s["last_active"] or "N/A"
        marker = " [CURRENT WORKSPACE]" if sid == curr_ws else ""
        typer.echo(f"  • {sid}{marker}: {count} messages (last active: {active})")
    typer.echo("")


@app.command()
def doctor(
    context_size: int = typer.Option(4096, "--context-size", min=512, help="Context window to size the model fit for."),
    no_network: bool = typer.Option(False, "--no-network", help="Skip the download speed test."),
):
    """Check Python, RAM, disk, network speed and which local model fits."""
    from ..utils.doctor import run_checks
    icons = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]", "info": "[INFO]"}
    failed = False
    for status, label, detail in run_checks(context_size=context_size, probe_network=not no_network):
        typer.echo(f"{icons[status]} {label}: {detail}")
        failed = failed or status == "fail"
    if failed:
        raise typer.Exit(code=1)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version information.")
):
    if version:
        from ..utils.config import get_package_version
        typer.echo(f"Nexus-Agent CLI v{get_package_version()}")
        raise typer.Exit()

    # Bypass onboarding wizard if --help or -h is passed anywhere
    if any(arg in sys.argv for arg in ["--help", "-h"]):
        if ctx.invoked_subcommand is None:
            ctx.invoke(repl)
        return

    if ctx.invoked_subcommand == "doctor":
        return  # diagnostics must work before (and without) the setup wizard

    try:
        from .onboarding import run_if_first_time
        run_if_first_time()
    except Exception:
        pass

    if ctx.invoked_subcommand is None:
        ctx.invoke(repl)


if __name__ == "__main__":
    app()
