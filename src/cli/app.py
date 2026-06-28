import time
import typer
from typing import Optional, Any
from ..utils.config import DEFAULT_PROVIDER, ConfigError
from ..agent.memory import ConversationMemory
from ..agent.core import Agent
from . import display

app = typer.Typer(help="Programmer Assistant CLI Coding Agent")

def get_provider_instance(provider_name: Any):
    """Factory to return the selected LLM provider."""
    if hasattr(provider_name, "default"):
        provider_name = provider_name.default
    if not isinstance(provider_name, str):
        provider_name = str(provider_name or DEFAULT_PROVIDER)
    name_clean = provider_name.lower().strip()
    if name_clean == "anthropic":
        from ..providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    elif name_clean == "gemini":
        from ..providers.gemini_provider import GeminiProvider
        return GeminiProvider()
    elif name_clean in ["openai", "gpt", "gpt-4o"]:
        from ..providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    else:
        display.print_error(f"Provider '{provider_name}' is not implemented. Using Anthropic fallback.")
        from ..providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()

@app.command()
def chat(
    query: str = typer.Argument(..., help="The coding question or instruction for the agent."),
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p", help="LLM provider backend."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose tool logs."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
):
    """Execute a single-turn chat instruction with autonomous tool calling."""
    try:
        prov = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose)

        display.print_header(provider, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))

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
    provider: str = typer.Option(DEFAULT_PROVIDER, "--provider", "-p", help="LLM provider backend."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose tool logs."),
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
        prov = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose)

        display.print_header(provider, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))
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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose tool logs."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
):
    """Review code quality, security, and potential bugs in a local file."""
    try:
        prov = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose)
        display.print_header(provider, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))
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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose tool logs."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
):
    """Diagnose and fix an error in a local codebase."""
    try:
        prov = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose)
        display.print_header(provider, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))
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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose tool logs."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable output streaming."),
):
    """Generate code autonomously and save directly to file."""
    try:
        prov = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory, verbose=verbose)
        display.print_header(provider, getattr(prov, "model", "unknown"), mode=getattr(agent, "mode_str", ""))
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

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version information.")
):
    if version:
        typer.echo("Programmer Assistant CLI v2.1.2")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        repl(provider=DEFAULT_PROVIDER, verbose=False, no_stream=False)

if __name__ == "__main__":
    app()
