import time
from typing import Any, Dict
from src.providers.base import BaseProvider
from src.agent.memory import ConversationMemory
from src.agent.tools import get_all_tools, execute_tool
from src.cli import display
from src.utils.config import estimate_cost

SYSTEM_PROMPT = """You are Programmer Assistant, an expert coding agent running in a CLI terminal.
You help developers write, debug, review, and understand code.

RULES:
1. Always use read_file tool before answering questions about a specific file.
   Never guess or assume file contents.
2. When creating or modifying code files, always use write_file tool.
3. When testing calculations, logic, or verifying scripts, use run_code tool.
4. Use list_directory to understand project structure before project-level questions.
5. Use git_status to inspect modified files or repository diffs.
6. Use search_web to look up live documentation, library APIs, or real-time information.
7. Write clean, minimal, production-quality code. No unnecessary comments.
8. If the user writes in Hindi, Hinglish, French, or any other language,
   respond in that same language.
9. After every tool call, reason about the result before deciding next action.
10. Never make up file contents, function signatures, or library APIs.
11. Be direct. Skip unnecessary preamble."""

class Agent:
    """Core autonomous coding agent implementing the ReAct tool-use loop."""

    def __init__(self, provider: BaseProvider, memory: ConversationMemory):
        self.provider = provider
        self.memory = memory
        self.tools = get_all_tools()
        self.system = SYSTEM_PROMPT
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def estimated_cost(self) -> float:
        model_name = getattr(self.provider, "model", "claude-3-5-sonnet-20241022")
        return estimate_cost(model_name, self.total_input_tokens, self.total_output_tokens)

    def run(self, user_input: str, stream: bool = False) -> str:
        """Execute the ReAct loop for a user input."""
        self.memory.add("user", user_input)
        messages = self.memory.get()

        display.print_thinking("Thinking...")

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            response = self.provider.complete(
                messages=messages,
                tools=self.tools,
                system=self.system
            )

            self.total_input_tokens += response.input_tokens
            self.total_output_tokens += response.output_tokens

            if response.has_tool_calls:
                self.memory.add_raw(response.raw_assistant_message)
                for tool_call in response.tool_calls:
                    display.print_tool_call(tool_call.name, tool_call.args)

                    start = time.time()
                    result = execute_tool(tool_call.name, tool_call.args)
                    duration = time.time() - start

                    display.print_tool_result(result, duration)

                    self.memory.add_raw(self.provider.format_tool_result_message(tool_call.id, result))

                messages = self.memory.get()
            else:
                final_text = response.text
                self.memory.add("assistant", final_text)
                return final_text

        return "Max tool iterations reached. Please try a more specific question."
