import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
from ..providers.base import BaseProvider, RateLimitError, ProviderResponse
from . import context
from .memory import ConversationMemory
from .tools import get_all_tools, execute_tool, execute_list_directory
from .guardrails import LoopGuard, guardrails_enabled
from ..cli import display
from ..utils.config import estimate_cost

RULES_PROMPT = """RULES:
1. Always use read_file tool before answering questions about a specific file.
   Never guess or assume file contents.
2. When creating new files, use write_file. When modifying existing files, always use patch_file
   to replace only the specific targeted code blocks and preserve file integrity.
   Read the file with read_file first; edits to unread or changed files are refused.
   If patch_file says the target was not found, copy the target exactly from the
   closest text it shows and retry once.
3. When testing calculations or standalone algorithms, use run_code tool.
   When verifying bug fixes, test passes, or regressions, use run_tests tool to execute pytest.
4. Use list_directory to understand project structure before project-level questions.
5. Use git_status to inspect modified files or repository diffs.
6. Use search_web to look up live documentation, library APIs, or real-time information.
7. Write clean, minimal, production-quality code. No unnecessary comments.
8. If the user writes in Hindi, Hinglish, French, or any other language,
   respond in that same language.
9. After every tool call, reason about the result before deciding next action.
10. Never make up file contents, function signatures, or library APIs.
11. Be direct. Skip unnecessary preamble.
12. STOP using tools once you have enough information to answer. Give your final
    answer as plain text. Do NOT keep calling the same tool repeatedly.
13. For simple math or factual questions, answer directly WITHOUT using tools.
14. If a duplicate tool call warning appears, immediately synthesize your final
    answer from the results you already have. Do NOT call any more tools.

THINKING PROTOCOL (MANDATORY):
Before making ANY tool call, you MUST first output at least one sentence of plain
text explaining what you are about to do and why. This reasoning must appear as
regular text BEFORE the tool call — never silent, never skipped.
Good example: "I'll read the file first to understand its current structure."
Bad example: [silent tool call with no prior text]
This applies to every single tool call in every iteration."""

# ── Smart Tool Routing for Local Models ──────────────────────────────────

# Keywords/patterns that indicate a query NEEDS tool access
_TOOL_PATTERNS = [
    # File operations
    r'\b(read|open|show|view|display|print)\b.*\b(file|\.py|\.js|\.ts|\.java|\.cpp|\.c\b|\.h\b|\.go|\.rs|\.rb|\.html|\.css|\.json|\.yaml|\.yml|\.toml|\.md|\.txt|\.csv|\.xml|\.sql)',
    r'\b(write|create|generate|make|build)\b.*\b(file|class|function|script|module|program)',
    r'\b(modify|edit|update|change|refactor|fix|patch)\b.*\b(file|code|function|class)',
    r'\b(delete|remove)\b.*\b(file|line|function)',
    r'@\S+',  # @mention file references
    # Code execution
    r'\b(run|execute|test|evaluate)\b.*\b(code|script|function|program|test|tests|pytest|suite)',
    # Git operations
    r'\b(git|commit|diff|branch|merge|stash|rebase|cherry)',
    # Web search
    r'\b(search|look up|find online|google|browse|latest version)',
    # Directory
    r'\b(list|show|explore)\b.*\b(dir|directory|folder|files|structure)',
    # Debug
    r'\b(debug|error|traceback|exception|bug|crash|broken|failing)',
    # Review
    r'\b(review|audit|analyze|check|inspect|scan)\b.*\b(code|file|project|repo)',
    # File path references
    r'[\w./\\]+\.\w{1,5}\b',  # anything that looks like a file path
]

def _is_local_provider(provider: BaseProvider) -> bool:
    """Check if the provider is a local model (needs smart tool routing)."""
    model = getattr(provider, 'model', '') or ''
    provider_name = type(provider).__name__.lower()
    return ('local' in provider_name or
            'liquid' in model.lower() or
            'lfm' in model.lower() or
            hasattr(provider, '_server_proc'))

_EDIT_INTENT = re.compile(r"\b(fix|change|edit|modify|update|replace|rewrite|refactor|patch|rename|implement|add|remove|delete)\b", re.IGNORECASE)
_WRITE_TOOLS = ("patch_file", "write_file")


def _wants_edit(user_input: str) -> bool:
    return bool(_EDIT_INTENT.search(user_input or ""))


def _query_needs_tools(user_input: str) -> bool:
    """Determine if a query likely needs tool access (file ops, code, git, etc.)."""
    q = user_input.lower().strip()
    for pattern in _TOOL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    return False


def parse_at_mentions(user_input: str) -> str:
    """Detect @filename mentions, synchronously read files, attach context invisibly, and clean prompt."""
    matches = re.findall(r'(?:^|\s)@\s*(?:"([^"]+)"|\'([^\']+)\'|([\w\.\-\/\\:]+))', user_input)
    if not matches:
        return user_input

    clean_input = user_input
    attachments = []
    extracted = [m[0] or m[1] or m[2] for m in matches if (m[0] or m[1] or m[2])]
    for raw_fpath in sorted(set(extracted), key=len, reverse=True):
        fpath = raw_fpath.rstrip('.!,?;:')
        try:
            resolved_path = Path(fpath).resolve()
        except Exception:
            resolved_path = Path(fpath)

        if resolved_path.exists() and resolved_path.is_file():
            from .tools import execute_read_file
            content = execute_read_file(str(resolved_path))
            if content.startswith("ERROR:"):
                attachments.append(f"[Warning: Could not attach @{fpath}: {content}]")
                continue
            pattern = r'(?:^|\s)@\s*(?:"' + re.escape(raw_fpath) + r'"|\'' + re.escape(raw_fpath) + r'\'|' + re.escape(raw_fpath) + r')(?=\s|$|[.!,?;:])'
            clean_input = re.sub(pattern, ' ', clean_input).strip()
            attachments.append(f"[Context attached from @{fpath}: \n{content}\n]")

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

    def __init__(self, provider: BaseProvider, memory: Optional[ConversationMemory] = None, max_iterations: int = 10, verbose: bool = True, tools: Optional[list] = None, event_callback: Optional[Any] = None, guardrails: Optional[bool] = None):
        self.provider = provider
        self.guardrails = guardrails_enabled(guardrails)
        self.memory = memory if memory is not None else ConversationMemory()
        self.tools = tools if tools is not None else get_all_tools()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.event_callback = event_callback
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Only local llama providers expose a fixed window we can budget against.
        window = getattr(provider, "context_size", None)
        self.context_window = window if isinstance(window, int) and not isinstance(window, bool) and window > 0 else None
        self.last_trim = {"dropped_messages": 0, "truncated_messages": 0, "tokens": 0}
        self.unresolved_errors: List[str] = []  # tool errors not fixed by the end of the last run

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
        if _is_local_provider(self.provider):
            return 0.0
        model_name = getattr(self.provider, "model", "")
        return estimate_cost(model_name, self.total_input_tokens, self.total_output_tokens)

    def _fit(self, messages, tools, scale: float = 1.0):
        """Trim what is sent to the model so it fits the window. Stored history is untouched."""
        if not self.context_window:
            return messages
        window = int(self.context_window * scale)
        reserve = min(1024, max(256, window // 4))  # room for the answer
        overhead = context.estimate_tokens(self.system) + context.tools_tokens(tools)
        budget = max(256, window - reserve - overhead)
        fitted, stats = context.fit_messages(messages, budget)
        self.last_trim = stats
        if stats["dropped_messages"] or stats["truncated_messages"]:
            display.print_warn(
                f"Context is full: left out {stats['dropped_messages']} older message(s) and cut "
                f"{stats['truncated_messages']} large item(s) for this request. Use /compact to shrink history."
            )
        return fitted

    def context_used(self) -> int:
        """Estimated tokens the next request would use (history + instructions)."""
        return context.history_tokens(self.memory.get()) + context.estimate_tokens(self.system) + context.tools_tokens(self.tools)

    def compact(self, keep_turns: int = 2):
        """Shrink stored history; returns (before_tokens, after_tokens, info)."""
        before = context.history_tokens(self.memory.get())
        new, info = context.compact_messages(self.memory.get(), keep_turns=keep_turns)
        if info["removed_messages"]:
            self.memory.replace(new)
        return before, context.history_tokens(self.memory.get()), info

    def run(self, user_input: str, stream: bool = False) -> str:
        """Execute the ReAct loop for a user input."""
        processed_input = parse_at_mentions(user_input)
        self.memory.add("user", processed_input)
        messages = self.memory.get()

        # Smart Tool Routing: for local models, only pass tools when the query
        # actually needs file/code/git/web operations. This prevents the local
        # model from looping on tools for simple questions.
        use_tools = self.tools
        local_mode = _is_local_provider(self.provider)
        if local_mode and not _query_needs_tools(processed_input):
            use_tools = []  # no tools → model answers directly
            effective_max_iter = 1
        else:
            effective_max_iter = self.max_iterations

        status = display.create_status("Thinking...") if self.verbose else None

        iteration = 0
        executed_tools = set()
        duplicate_counts = {}
        failed_counts: Dict[str, int] = {}
        force_final = False
        tool_errors = []
        unresolved: Dict[str, str] = {}  # key (path or tool name) -> error still not fixed
        window_scale = 1.0
        wrote_ok = False
        guard = LoopGuard(execute_list_directory) if self.guardrails else None
        self.unresolved_errors = []
        try:
            while iteration < effective_max_iter:
                iteration += 1
                display.update_status(status, "Thinking...")
                send_messages = self._fit(messages, use_tools, window_scale)

                try:
                    if stream and hasattr(self.provider, "stream"):
                        if status:
                            display.stop_status(status)
                            status = None
                        response = None
                        streamed_text = ""
                        for item in self.provider.stream(messages=send_messages, tools=use_tools, system=self.system):
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
                            messages=send_messages,
                            tools=use_tools,
                            system=self.system
                        )
                except RateLimitError as e:
                    display.stop_status(status)
                    status = None
                    display.print_warn(f"{e.provider} rate limit hit. Switching to next provider...")
                    raise
                except Exception as e:
                    # One retry with a tighter budget when the server says the prompt did not fit.
                    if self.context_window and window_scale == 1.0 and context.is_context_overflow(e):
                        window_scale = 0.6
                        iteration -= 1
                        display.print_warn("The model rejected the request as too large. Retrying once with a smaller history.")
                        continue
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
                        
                        allowed_names = {tool.name for tool in use_tools}
                        if tool_call.name not in allowed_names:
                            result = f"ERROR: Tool '{tool_call.name}' is not allowed in this session."
                        elif tool_sig in executed_tools:
                            duplicate_counts[tool_sig] = duplicate_counts.get(tool_sig, 0) + 1
                            if duplicate_counts[tool_sig] >= 2:
                                # Hard stop: the model is stuck in a loop.
                                # Strip tools on the next request and force a direct answer.
                                force_final = True
                                result = ("[SYSTEM: TOOL BUDGET EXHAUSTED] You have repeated this exact tool call "
                                          "multiple times. Do NOT call any more tools. Answer the user's original "
                                          "question directly now using the information you already have.")
                            else:
                                result = "[SYSTEM WARNING: Duplicate Tool Call Detected] You have already executed this tool with these exact arguments. Synthesize your answer from existing results, or try a completely different approach."
                        else:
                            executed_tools.add(tool_sig)
                            result = execute_tool(tool_call.name, tool_call.args)
                            if guard:
                                result = guard.after_tool(tool_call.name, tool_call.args, result)

                        error_key = str(tool_call.args.get("path") or tool_call.name)
                        if str(result).startswith("ERROR:"):
                            tool_errors.append(str(result))
                            unresolved[error_key] = str(result)
                            # A failed call may legitimately be retried unchanged (e.g. after
                            # read_file). Only a repeated failure counts as a loop.
                            failed_counts[tool_sig] = failed_counts.get(tool_sig, 0) + 1
                            if failed_counts[tool_sig] < 2:
                                executed_tools.discard(tool_sig)
                        elif not str(result).startswith("[SYSTEM"):
                            unresolved.pop(error_key, None)  # a later success on the same file fixes it
                            if tool_call.name in _WRITE_TOOLS:
                                wrote_ok = True
                                # State changed: re-running run_tests / read_file / git_* is legitimate now.
                                executed_tools.clear()
                        duration = time.time() - start

                        if self.event_callback:
                            self.event_callback({"type": "observe", "name": tool_call.name, "duration": round(duration, 2), "result": str(result)})

                        if self.verbose:
                            display.print_tool_result(result, duration, tool_call.name)

                        self.memory.add_raw(self.provider.format_tool_result_message(tool_call.id, result))

                    if force_final:
                        # Anti-loop: no tools next round, model must answer directly
                        self.memory.add("user", "[SYSTEM] Tool budget exhausted: the same tool call was repeated. "
                                                 "Do not call any more tools. Give your final answer to the user's "
                                                 "original question now.")
                        use_tools = []
                        effective_max_iter = max(effective_max_iter, iteration + 1)

                    messages = self.memory.get()

                    # Resume spinner for next iteration
                    if self.verbose and status is None:
                        status = display.create_status("Thinking...")
                else:
                    display.stop_status(status)
                    status = None
                    final_text = response.text
                    if guard and use_tools:
                        verdict, reviewed = guard.review_final(final_text)
                        if verdict == "retry":
                            self.memory.add("assistant", final_text)
                            self.memory.add("user", reviewed)
                            if self.event_callback:
                                self.event_callback({"type": "guardrail", "content": "blocked unverified success claim"})
                            effective_max_iter = max(effective_max_iter, iteration + 1)
                            messages = self.memory.get()
                            if self.verbose and status is None:
                                status = display.create_status("Thinking...")
                            continue
                        final_text = reviewed
                    self.unresolved_errors = list(unresolved.values())
                    if (not wrote_ok and not self.unresolved_errors and _wants_edit(user_input)
                            and any(getattr(t, "name", None) in _WRITE_TOOLS for t in (use_tools or []))):
                        final_text += ("\n\nWarning: no file was changed. No successful patch_file or write_file call "
                                       "happened in this run, so do not assume the edit was applied.")
                    if self.unresolved_errors:
                        final_text += "\n\nTool errors occurred; do not assume the task succeeded:\n" + "\n".join(dict.fromkeys(self.unresolved_errors))
                    if self.event_callback:
                        self.event_callback({"type": "response", "content": final_text, "tokens": self.total_tokens, "cost": self.estimated_cost})
                    
                    if stream and not self.event_callback:
                        if streamed_text:
                            print()  # newline after streamed chunks
                        else:
                            display.print_response(final_text)

                    self.memory.add("assistant", final_text)
                    return final_text
        finally:
            display.stop_status(status)
            self.unresolved_errors = list(unresolved.values())

        return "Max tool iterations reached. Please try a more specific question."
