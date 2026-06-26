import time
import typer
from typing import Optional
from src.utils.config import DEFAULT_PROVIDER, ConfigError
from src.providers.anthropic_provider import AnthropicProvider
from src.agent.memory import ConversationMemory
from src.agent.core import Agent
from src.cli import display

app = typer.Typer(help="Programmer Assistant CLI Coding Agent")

def get_provider_instance(provider_name: str):
    """Factory to return the selected LLM provider."""
    name_clean = provider_name.lower().strip()
    if name_clean == "anthropic":
        return AnthropicProvider()
    elif name_clean == "gemini":
        from src.providers.gemini_provider import GeminiProvider
        return GeminiProvider()
    elif name_clean in ["openai", "gpt", "gpt-4o"]:
        from src.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    else:
        display.print_error(f"Provider '{provider_name}' is not implemented. Using Anthropic fallback.")
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
        agent = Agent(provider=prov, memory=memory)

        display.print_header(provider, getattr(prov, "model", "unknown"))

        start_time = time.time()
        response_text = agent.run(query, stream=not no_stream)
        duration = time.time() - start_time

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
    try:
        prov = get_provider_instance(provider)
        memory = ConversationMemory()
        agent = Agent(provider=prov, memory=memory)

        display.print_header(provider, getattr(prov, "model", "unknown"))
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

@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Show version information.")
):
    if version:
        typer.echo("Programmer Assistant CLI v2.0.0 (Week 2 Scope)")
        raise typer.Exit()

if __name__ == "__main__":
    app()
