import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
from ..providers.base import BaseProvider, RateLimitError, ProviderResponse
from .memory import ConversationMemory
from .tools import get_all_tools, execute_tool
from ..cli import display
from ..utils.config import estimate_cost

RULES_PROMPT = """RULES:
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
11. Be direct. Skip unnecessary preamble.

THINKING PROTOCOL (MANDATORY):
Before making ANY tool call, you MUST first output at least one sentence of plain
text explaining what you are about to do and why. This reasoning must appear as
regular text BEFORE the tool call — never silent, never skipped.
Good example: "I'll read the file first to understand its current structure."
Bad example: [silent tool call with no prior text]
This applies to every single tool call in every iteration."""

def parse_at_mentions(user_input: str) -> str:
    """Detect @filename mentions, synchronously read files, attach context invisibly, and clean prompt."""
    matches = re.findall(r'(?:^|\s)@\s*([\w\.\-\/\\:]+)', user_input)
    if not matches:
        return user_input

    clean_input = user_input
    attachments = []
    for raw_fpath in sorted(set(matches), key=len, reverse=True):
        fpath = raw_fpath.rstrip('.!,?;:')
        try:
            resolved_path = Path(fpath).resolve()
        except Exception:
            resolved_path = Path(fpath)

        if resolved_path.exists() and resolved_path.is_file():
            from .tools import _validate_workspace_path
            validated = _validate_workspace_path(resolved_path)
            if isinstance(validated, str):
                attachments.append(f"[Warning: Security blocked reading @{fpath}: {validated}]")
                continue

            pattern = r'(?:^|\s)@\s*' + re.escape(raw_fpath) + r'(?=\s|$|[.!,?;:])'
            clean_input = re.sub(pattern, ' ', clean_input).strip()
            try:
                with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                attachments.append(f"[Context attached from @{fpath}: \n{content}\n]")
            except Exception as e:
                attachments.append(f"[Warning: Could not read @{fpath}: {str(e)}]")
        else:
            attachments.append(f"[Warning: Mentioned file @{fpath} does not exist]")

    clean_input = re.sub(r'\s+', ' ', clean_input).strip()
    if not clean_input:
        clean_input = "Please inspect the attached file context."

    if attachments:
        return clean_input + "\n\n" + "\n\n".join(attachments)
    return clean_input

class Agent:
    """Core autonomous coding agent implementing the ReAct tool-use loop."""

    def __init__(self, provider: BaseProvider, memory: Optional[ConversationMemory] = None, max_iterations: int = 10, verbose: bool = True, tools: Optional[list] = None, event_callback: Optional[Any] = None):
        self.provider = provider
        self.memory = memory if memory is not None else ConversationMemory()
        self.tools = tools if tools is not None else get_all_tools()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.event_callback = event_callback
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Smart Startup: Global vs. Project Context check
        cwd = os.getcwd()
        has_git = os.path.exists(os.path.join(cwd, ".git"))
        has_pyproject = os.path.exists(os.path.join(cwd, "pyproject.toml"))
        has_package_json = os.path.exists(os.path.join(cwd, "package.json"))

        if has_git or has_pyproject or has_package_json:
            project_name = os.path.basename(cwd) or "Unknown Project"
            self.mode_str = f"Project Mode ({project_name})"
            mode_prompt = f"You are Nexus-Agent, currently working inside the project directory: {project_name}."
        else:
            self.mode_str = "Global Mode"
            mode_prompt = "You are Nexus-Agent, an expert coding agent running in a CLI terminal in Global Mode."

        self.system = f"{mode_prompt}\nYou help developers write, debug, review, and understand code.\n\n{RULES_PROMPT}"

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def estimated_cost(self) -> float:
        model_name = getattr(self.provider, "model", "claude-3-5-sonnet-20241022")
        return estimate_cost(model_name, self.total_input_tokens, self.total_output_tokens)

    def run(self, user_input: str, stream: bool = False) -> str:
        """Execute the ReAct loop for a user input."""
        processed_input = parse_at_mentions(user_input)
        self.memory.add("user", processed_input)
        messages = self.memory.get()

        status = display.create_status("Thinking...") if self.verbose else None

        iteration = 0
        executed_tools = set()
        try:
            while iteration < self.max_iterations:
                iteration += 1
                display.update_status(status, "Thinking...")

                try:
                    if stream and hasattr(self.provider, "stream"):
                        if status:
                            display.stop_status(status)
                            status = None
                        response = None
                        streamed_text = ""
                        for item in self.provider.stream(messages=messages, tools=self.tools, system=self.system):
                            if isinstance(item, ProviderResponse):
                                response = item
                            elif isinstance(item, str):
                                streamed_text += item
                                if self.event_callback:
                                    self.event_callback({"type": "stream_chunk", "content": item})
                                else:
                                    display.print_stream_chunk(item)
                        if response is None:
                            approx_out = max(1, len(streamed_text) // 4) if streamed_text else 0
                            response = ProviderResponse(text=streamed_text, input_tokens=0, output_tokens=approx_out)
                        elif streamed_text and not response.text:
                            response.text = streamed_text
                    else:
                        response = self.provider.complete(
                            messages=messages,
                            tools=self.tools,
                            system=self.system
                        )
                except RateLimitError as e:
                    display.stop_status(status)
                    status = None
                    display.print_warn(f"{e.provider} rate limit hit. Switching to next provider...")
                    raise

                self.total_input_tokens += response.input_tokens
                self.total_output_tokens += response.output_tokens

                if response.has_tool_calls:
                    self.memory.add_raw(response.raw_assistant_message)

                    # Show THINKING trace: the LLM's reasoning text before tool call
                    if response.text and not stream:
                        if self.event_callback:
                            self.event_callback({"type": "thinking", "content": response.text})
                        if self.verbose:
                            display.stop_status(status)
                            status = None
                            display.print_thinking(response.text)

                    for tool_call in response.tool_calls:
                        # Build status display with full argument visibility (up to 150 chars)
                        args_preview = ", ".join(
                            f'{k}="{v[:120]}..."' if isinstance(v, str) and len(v) > 120 else
                            (f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}")
                            for k, v in tool_call.args.items()
                        )
                        if status is None and self.verbose:
                            status = display.create_status(f"Running: {tool_call.name}...")
                        display.update_status(status, f"Running tool: {tool_call.name}({args_preview[:150]})...")

                        if self.event_callback:
                            self.event_callback({"type": "action", "name": tool_call.name, "args": tool_call.args})

                        if self.verbose:
                            display.stop_status(status)
                            status = None
                            display.print_tool_call(tool_call.name, tool_call.args)

                        # Check for exact duplicate tool calls
                        # We use json.dumps with sorted_keys to ensure deterministic string representation
                        import json
                        tool_sig = f"{tool_call.name}:{json.dumps(tool_call.args, sort_keys=True)}"
                        start = time.time()
                        
                        if tool_sig in executed_tools:
                            result = "[SYSTEM WARNING: Duplicate Tool Call Detected] You have already executed this tool with these exact arguments. Synthesize your answer from existing results, or try a completely different approach."
                        else:
                            executed_tools.add(tool_sig)
                            result = execute_tool(tool_call.name, tool_call.args)
                            
                        duration = time.time() - start

                        if self.event_callback:
                            self.event_callback({"type": "observe", "name": tool_call.name, "duration": round(duration, 2), "result": str(result)})

                        if self.verbose:
                            display.print_tool_result(result, duration, tool_call.name)

                        self.memory.add_raw(self.provider.format_tool_result_message(tool_call.id, result))

                    messages = self.memory.get()

                    # Resume spinner for next iteration
                    if self.verbose and status is None:
                        status = display.create_status("Thinking...")
                else:
                    display.stop_status(status)
                    status = None
                    final_text = response.text
                    if self.event_callback:
                        self.event_callback({"type": "response", "content": final_text, "tokens": self.total_tokens, "cost": self.estimated_cost})
                    
                    if stream and not self.event_callback:
                        if streamed_text:
                            print() # Just insert a newline after streamed chunks
                        else:
                            display.print_response(final_text)
                        
                    self.memory.add("assistant", final_text)
                    return final_text
        finally:
            display.stop_status(status)

        return "Max tool iterations reached. Please try a more specific question."
