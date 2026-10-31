import os
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
from ..providers.base import BaseProvider, RateLimitError, ProviderResponse
from . import context
from .memory import ConversationMemory, is_genuine_user_message
from .tools import get_all_tools, execute_tool, execute_list_directory
from .guardrails import (
    LoopGuard, guardrails_enabled, promises_tool_use, function_request_prose,
    requested_function_names, check_function_requirements,
)
from ..cli import display
from ..utils.config import estimate_cost
from ..providers.messages import assistant_message

RULES_PROMPT = """RULES:
1. Always use read_file tool before answering questions about a specific file.
   Never guess or assume file contents.
2. When creating new files, use write_file. When modifying existing files, always use patch_file
   to replace only the specific targeted code blocks and preserve file integrity.
   Read the file with read_file first; edits to unread or changed files are refused.
   patch_file.target is OLD text already in the file; replacement is the desired NEW text.
   If the target was not found, copy the exact target JSON from the error and retry once.
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
13. Use run_code for arithmetic. Answer conceptual questions directly when no tools are needed.
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
    # Explicit capability requests must not depend on underscores being word boundaries.
    r'\b(?:read_file|write_file|patch_file|list_directory|run_code|run_tests|search_web|git_status|git_diff|git_commit)\b',
    r'\b(?:calculate|compute|calculation|arithmetic)\b',
    r'\b\d[\d,]*(?:\.\d+)?\s*(?:[+*/%−-]|plus\b|minus\b|times\b|divided by\b|multiplied by\b)\s*[-+]?\d',
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
    if hasattr(provider, "is_local"):
        return bool(provider.is_local)
    provider_name = type(provider).__name__.lower()
    return ('local' in provider_name or
            'liquid' in model.lower() or
            'lfm' in model.lower() or
            hasattr(provider, '_server_proc'))

_EDIT_INTENT = re.compile(r"\b(fix|change|edit|modify|update|replace|rewrite|refactor|patch|rename|implement|add|remove|delete)\b", re.IGNORECASE)
_WRITE_TOOLS = ("patch_file", "write_file")
_VERBS = r"(?:fixed|updated|changed|modified|patched|applied|corrected|replaced|rewrote|rewritten|edited|implemented|added|removed|renamed|resolved)"
# The model says it changed something ("I've fixed...", "the bug has been fixed", "I fixed the bug").
_CHANGE_CLAIM = re.compile(r"\b(?:i(?:'ve|\s+have)?\s+(?:now\s+|successfully\s+)?" + _VERBS + r"|(?:has|have|was|were|is|are)\s+(?:now\s+|successfully\s+)?(?:been\s+)?" + _VERBS + r")\b", re.IGNORECASE)
# Legitimate "nothing to change" answers must not be flagged.
_NO_CHANGE_OK = re.compile(
    r"\b(?:already\s+(?:correct|fixed|implemented|up.to.date)|"
    r"no\s+(?:code\s+)?(?:changes?|fix(?:es)?|edits?)\s+(?:is\s+|are\s+)?(?:needed|necessary|required)|"
    r"nothing\s+(?:to\s+(?:change|fix)|needs?\s+(?:changing|fixing))|not\s+(?:a\s+)?bug|"
    r"(?:did\s+not|didn't|have\s+not|haven't|could\s+not|couldn't|unable\s+to)\s+"
    r"(?:appl(?:y|ied)|make|made|chang(?:e|ed)|fix(?:ed)?|edit(?:ed)?|modif(?:y|ied)|write|written|patch(?:ed)?))\b", re.IGNORECASE,
)


def _file_state(path: Path):
    """Fingerprint file bytes without retaining potentially large tool payloads."""
    try:
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").digest()
        return ("file", digest)
    except FileNotFoundError:
        return ("missing", None)
    except OSError:
        return None

def _edit_path(args) -> Optional[Path]:
    try:
        path = args.get("path")
        return Path(path).resolve() if isinstance(path, (str, os.PathLike)) and path else None
    except (OSError, ValueError):
        return None



def _wants_edit(user_input: str) -> bool:
    text = (user_input or "").strip()
    informational = re.match(
        r"^(?:please\s+)?(?:how\b|what\b|why\b|when\b|where\b|explain\b|describe\b|"
        r"read\b|review\b|inspect\b|(?:can|could|would)\s+you\s+(?:please\s+)?(?:explain|describe)\b)",
        text, re.IGNORECASE,
    )
    if informational:
        # A clear subsequent imperative can turn an explanation into an edit task.
        clauses = re.split(r"\bthen\s+|[.;]\s+", text, flags=re.IGNORECASE)
        return any(_EDIT_INTENT.match(clause.lstrip()) for clause in clauses[1:])
    return bool(_EDIT_INTENT.search(text))


_LIVE_SEARCH_REQUEST = re.compile(
    r"\b(?:search|browse)\s+(?:the\s+)?(?:web|internet)\b|"
    r"\b(?:search|browse|look\s+up|find)\b[^.!?\n]*\b"
    r"(?:online|official\s+(?:sources?|docs?|documentation|release\s+notes)|"
    r"live\s+(?:sources?|results?))\b|"
    r"^\s*(?:please\s+)?(?:use\s+)?search_web\b|"
    r"\b(?:current|latest)\s+(?:stable\s+)?(?:[\w.+-]+\s+){0,3}(?:version|release)\b",
    re.IGNORECASE,
)


def _query_needs_tools(user_input: str) -> bool:
    """Determine if a query likely needs tool access (file ops, code, git, etc.)."""
    q = user_input.lower().strip()
    if _LIVE_SEARCH_REQUEST.search(q):
        return True
    for pattern in _TOOL_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    return False


_RETRY_REQUEST = re.compile(
    r"(?:please\s+)?(?:try again|retry|try once more|do it again)(?:\s+please)?[.!?]*",
    re.IGNORECASE,
)


def _routing_request(user_input: str, history: list) -> str:
    """Resolve bare retries against the last real request, never internal nudges."""
    if not _RETRY_REQUEST.fullmatch(user_input.strip()):
        return user_input
    for message in reversed(history):
        if is_genuine_user_message(message):
            previous = message.get("content", "")
            if isinstance(previous, str) and not _RETRY_REQUEST.fullmatch(previous.strip()):
                return previous
    return user_input


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

_FILE_TOKEN = re.compile(r'(?<![\w@/\\.-])([\w][\w.\-]*(?:[/\\][\w.\-]+)*\.[A-Za-z0-9]{1,6})(?=[\s.,;:!?)\'"]|$)')

def _named_function_requirements(request: str) -> tuple[Dict[str, List[str]], List[str]]:
    """Bind recognized Python requests, or explicitly report an unverified scope."""
    prose = function_request_prose(request)
    names = requested_function_names(prose)
    if not names:
        return {}, []
    paths = {match.group(1) for match in _FILE_TOKEN.finditer(prose.replace("@", " "))}
    if paths and not any(Path(path).suffix == ".py" for path in paths):
        return {}, []  # This guard does not impose Python rules on other languages.
    if len(paths) == 1:
        path = next(iter(paths))
        if Path(path).suffix == ".py":
            return {path: names}, []
    return {}, [
        f"Could not verify requested Python function names ({', '.join(names)}): "
        "this bounded check requires one explicit unquoted relative .py target. "
        "No file was inferred for verification."
    ]



def attach_named_files(user_input: str) -> str:
    if os.environ.get("NEXUS_AGENT_AUTO_ATTACH", "1") == "0":
        return user_input
    return _attach_named_files(user_input)


def _attach_named_files(user_input: str) -> str:
    """For edit requests, turn bare names of files that exist in the current folder (calc.py) into
    @mentions, so their content is attached up front. Small local models often skip read_file and
    then every edit is refused. Only files inside the working folder are used; a handful at most."""
    if not _wants_edit(user_input):
        return user_input
    cwd = Path.cwd().resolve()
    seen = []
    for m in _FILE_TOKEN.finditer(user_input):
        tok = m.group(1)
        if tok in seen:
            continue
        try:
            p = (cwd / tok).resolve()
            if p.is_file() and (cwd == p or cwd in p.parents) and p.stat().st_size <= 100_000:
                seen.append(tok)
        except Exception:
            continue
        if len(seen) >= 4:
            break
    if not seen:
        return user_input
    # Append the mentions instead of rewriting the sentence: parse_at_mentions removes "@name" from the
    # prompt text, and the model must still see which files the user is talking about.
    out = user_input.rstrip() + " " + " ".join("@" + t for t in seen)
    return out


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
        self._sync_context_window()
        self.last_trim = {"dropped_messages": 0, "truncated_messages": 0, "tokens": 0}
        self.unresolved_errors: List[str] = []  # tool errors not fixed by the end of the last run
        self.claimed_unapplied_edit = False  # model said it changed code but no write succeeded
        self.edit_not_applied = False  # an edit was requested, nothing was written, and the model did not say nothing needs changing
        self.claim_unverified = False  # model said tests pass but they were not run / last run failed
        self.claim_unverified_error = ""
        self.unfulfilled_action = False
        self.response_incomplete = False
        self.changed_files: set[str] = set()
        self.requirements_incomplete: List[str] = []

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

    def _sync_context_window(self, provider=None):
        active = provider if provider is not None else getattr(self.provider, "active_provider", self.provider)
        window = getattr(active, "context_size", None)
        self.context_window = window if isinstance(window, int) and not isinstance(window, bool) and window > 0 else None
        output = getattr(active, "max_output_tokens", None)
        self.output_reserve = output if isinstance(output, int) and not isinstance(output, bool) and output > 0 else 0

    def _nudge(self, text: str):
        self.memory.add_raw({"role": "user", "content": text, "metadata": {"internal": True}})

    def _stream_correction(self, text: str):
        if not text:
            return
        if self.event_callback:
            self.event_callback({"type": "stream_chunk", "content": text})
        else:
            display.print_stream_chunk(text)

    def _fit(self, messages, tools, scale: float = 1.0):
        """Trim what is sent to the model so it fits the window. Stored history is untouched."""
        if not self.context_window:
            return messages
        window = int(self.context_window * scale)
        configured = self.output_reserve
        reserve = configured
        active = getattr(self.provider, "active_provider", self.provider)
        adaptive = _is_local_provider(active) and getattr(active, "_supports_output_budget", False)
        # Keep a usable reply allowance rather than shrinking all the way to a
        # single token. Explicit smaller caps remain exact.
        minimum_output = min(configured, 256)
        overhead = context.estimate_tokens(self.system) + context.tools_tokens(tools)
        while True:
            budget = window - reserve - overhead
            try:
                if budget <= 0:
                    raise context.ContextBudgetError(
                        f"Configured context budget ({window} tokens) cannot fit instructions, tools and answer reserve "
                        f"({overhead + reserve} tokens). Adjust the configured context/output budgets within "
                        "the endpoint's documented capacity, or use fewer tools."
                    )
                fitted, stats = context.fit_messages(messages, budget)
                break
            except context.ContextBudgetError:
                if not adaptive or reserve <= minimum_output:
                    raise
                reserve = max(minimum_output, reserve // 2)
        if adaptive:
            active._request_output_cap = reserve
            self.output_reserve = reserve
            if reserve < configured:
                display.print_warn(
                    f"Local output cap reduced for this request: {configured} -> {reserve} tokens "
                    "so input plus reserved output fits the context window."
                )
        self.last_trim = stats
        if stats["dropped_messages"] or stats["truncated_messages"]:
            display.print_warn(
                f"Context is full: left out {stats['dropped_messages']} older message(s) and cut "
                f"{stats['truncated_messages']} large item(s) for this request. Use /compact to shrink history."
            )
        return fitted

    def context_used(self) -> int:
        """Estimated context allocation, including the configured generation reserve."""
        self._sync_context_window()
        return (context.history_tokens(self.memory.get()) + context.estimate_tokens(self.system)
                + context.tools_tokens(self.tools) + self.output_reserve)

    def compact(self, keep_turns: int = 2):
        """Shrink stored history; returns (before_tokens, after_tokens, info)."""
        before = context.history_tokens(self.memory.get())
        new, info = context.compact_messages(self.memory.get(), keep_turns=keep_turns)
        if info["removed_messages"]:
            self.memory.replace(new)
        return before, context.history_tokens(self.memory.get()), info

    def run(self, user_input: str, stream: bool = False, require_edit: bool = False) -> str:
        """Execute the ReAct loop for a user input."""
        routing_input = _routing_request(user_input, self.memory.get())
        requires_search = bool(_LIVE_SEARCH_REQUEST.search(routing_input))
        wants_edit = require_edit or (
            any(tool.name in _WRITE_TOOLS for tool in self.tools) and _wants_edit(routing_input)
        )
        function_requirements, requirement_scope_errors = (
            _named_function_requirements(routing_input) if self.guardrails else ({}, [])
        )
        processed_input = parse_at_mentions(attach_named_files(user_input))
        self.memory.add("user", processed_input)
        error_nudges = 0
        messages = self.memory.get()

        # Smart Tool Routing: for local models, only pass tools when the query
        # actually needs file/code/git/web operations. This prevents the local
        # model from looping on tools for simple questions.
        use_tools = self.tools
        local_mode = _is_local_provider(self.provider)
        if local_mode and not wants_edit and not _query_needs_tools(routing_input) and not _query_needs_tools(processed_input):
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
        unresolved_tool: Dict[str, str] = {}  # key -> name of the tool that failed
        window_scale = 1.0
        wrote_ok = False
        tool_executed = False
        search_succeeded = False
        offered_tools = list(use_tools)  # use_tools is emptied by the anti-loop stop; keep what was offered
        guard = LoopGuard(execute_list_directory) if self.guardrails else None
        self.unresolved_errors = []
        self.claimed_unapplied_edit = False
        self.edit_not_applied = False
        self.claim_unverified = False
        self.claim_unverified_error = ""
        retrieved_search_urls: List[str] = []
        self.unfulfilled_action = False
        self.response_incomplete = False
        self.changed_files = set()
        self.requirements_incomplete = []
        previous_before_request = getattr(self.provider, "before_request", None)
        managed_fallback = hasattr(self.provider, "before_request")
        if managed_fallback:
            def before_request(active, full_messages, tools, system):
                self._sync_context_window(active)
                return self._fit(full_messages, tools, window_scale), system
            self.provider.before_request = before_request
        try:
            while iteration < effective_max_iter:
                iteration += 1
                display.update_status(status, "Thinking...")
                self._sync_context_window()
                local_mode = _is_local_provider(self.provider)
                send_messages = messages if managed_fallback else self._fit(messages, use_tools, window_scale)

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
                    display.print_warn(f"{e.provider} rate limit hit. Request stopped; no fallback provider is available.")
                    raise
                except Exception as e:
                    # One retry with a tighter budget when the server says the prompt did not fit.
                    if self.context_window and window_scale == 1.0 and context.is_context_overflow(e):
                        window_scale = 0.6
                        iteration -= 1
                        display.print_warn("The model rejected the request as too large. Retrying once with a smaller history.")
                        continue
                    raise
                finally:
                    active = getattr(self.provider, "active_provider", self.provider)
                    if hasattr(active, "_request_output_cap"):
                        del active._request_output_cap

                self.total_input_tokens += response.input_tokens
                self.total_output_tokens += response.output_tokens
                self._sync_context_window()
                local_mode = _is_local_provider(self.provider)

                if response.finish_reason == "length":
                    # Even complete-looking arguments may belong to an unfinished plan.
                    self.response_incomplete = True
                    response.tool_calls = []

                if response.has_tool_calls:
                    self.memory.add_raw(assistant_message(response.text, response.tool_calls))

                    # Show THINKING trace: the LLM's reasoning text before tool call
                    if response.text and not stream:
                        if self.event_callback:
                            self.event_callback({"type": "thinking", "content": response.text})
                        if self.verbose:
                            display.stop_status(status)
                            status = None
                            display.print_thinking(response.text)

                    for tool_call in response.tool_calls:
                        changed = False
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
                        tool_sig = f"{tool_call.name}:{json.dumps(tool_call.args, sort_keys=True)}"
                        start = time.time()
                        
                        allowed_names = {tool.name for tool in use_tools}
                        if tool_call.name not in allowed_names:
                            result = f"ERROR: Tool '{tool_call.name}' is not allowed in this session."
                        elif tool_sig in executed_tools:
                            duplicate_counts[tool_sig] = duplicate_counts.get(tool_sig, 0) + 1
                            reread_while_editing = (tool_call.name == "read_file" and not wrote_ok and wants_edit
                                                    and any(getattr(t, "name", None) in _WRITE_TOOLS for t in offered_tools))
                            if duplicate_counts[tool_sig] >= (3 if reread_while_editing else 2):
                                # Hard stop: the model is stuck in a loop.
                                # Strip tools on the next request and force a direct answer.
                                force_final = True
                                result = ("[SYSTEM: TOOL BUDGET EXHAUSTED] You have repeated this exact tool call "
                                          "multiple times. Do NOT call any more tools. Answer the user's original "
                                          "question directly now using the information you already have.")
                            elif reread_while_editing:
                                result = ("[SYSTEM] You already read this file; its content is in the earlier tool result. Do not read it "
                                          "again. Call patch_file (or write_file) now to make the change, then run_tests.")
                            else:
                                result = "[SYSTEM WARNING: Duplicate Tool Call Detected] You have already executed this tool with these exact arguments. Synthesize your answer from existing results, or try a completely different approach."
                        else:
                            executed_tools.add(tool_sig)
                            edit_path = _edit_path(tool_call.args) if tool_call.name in _WRITE_TOOLS else None
                            before_edit = _file_state(edit_path) if edit_path is not None else None
                            tool_executed = True
                            result = execute_tool(tool_call.name, tool_call.args)
                            if tool_call.name == "search_web" and str(result).startswith("Search results for:"):
                                search_succeeded = True
                                for u in re.findall(r"^\s*URL:\s*(https?://\S+)", str(result), re.MULTILINE):
                                    clean_u = u.strip().rstrip(".,;)]}>\"'")
                                    if clean_u.startswith(("http://", "https://")) and clean_u not in retrieved_search_urls:
                                        retrieved_search_urls.append(clean_u)
                            changed = (edit_path is not None and before_edit is not None
                                       and (after_edit := _file_state(edit_path)) is not None
                                       and after_edit != before_edit
                                       and not str(result).startswith(("ERROR:", "No changes:")))
                            if changed:
                                self.changed_files.add(str(edit_path))
                            if guard:
                                result = guard.after_tool(tool_call.name, tool_call.args, result)

                        error_key = str(tool_call.args.get("path") or tool_call.name)
                        if str(result).startswith("ERROR:"):
                            tool_errors.append(str(result))
                            unresolved[error_key] = str(result)
                            unresolved_tool[error_key] = tool_call.name
                            # A failed call may legitimately be retried unchanged (e.g. after
                            # read_file). Only a repeated failure counts as a loop.
                            failed_counts[tool_sig] = failed_counts.get(tool_sig, 0) + 1
                            if failed_counts[tool_sig] < 2:
                                executed_tools.discard(tool_sig)
                        elif not str(result).startswith("[SYSTEM"):
                            # A later success fixes the error only if it is the same tool succeeding, or a
                            # write. Reading the file is not a fix for a failed edit.
                            failed_tool = unresolved_tool.get(error_key)
                            if failed_tool is None or tool_call.name == failed_tool or tool_call.name in _WRITE_TOOLS:
                                unresolved.pop(error_key, None)
                                unresolved_tool.pop(error_key, None)
                            if tool_call.name in _WRITE_TOOLS and changed:
                                wrote_ok = True
                                from .consent import auto_check_enabled
                                if auto_check_enabled() and any(getattr(t, "name", None) == "run_tests" for t in use_tools):
                                    # Opt-in (NEXUS_AGENT_AUTO_CHECK=1) and already permitted: run the tests now, show the real result.
                                    from .tools import execute_run_tests
                                    check = execute_run_tests()
                                    if guard:
                                        check = guard.after_tool("run_tests", {}, check)
                                    result = f"{result}\n\n[AUTO CHECK after this edit, run_tests result]\n{check}"
                                # State changed: re-running run_tests / read_file / git_* is legitimate now.
                                executed_tools.clear()
                                duplicate_counts.clear()  # repeats made before the file changed are not repeats now
                        duration = time.time() - start

                        if self.event_callback:
                            self.event_callback({"type": "observe", "name": tool_call.name, "duration": round(duration, 2), "result": str(result)})

                        if self.verbose:
                            display.print_tool_result(result, duration, tool_call.name)

                        self.memory.add_raw({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})

                    if force_final:
                        # Anti-loop: no tools next round, model must answer directly
                        self._nudge("[SYSTEM] Tool budget exhausted: the same tool call was repeated. "
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
                    if self.response_incomplete:
                        final_text += ("\n\nNot completed: the model reached its output limit. "
                                       "This response is incomplete; no tool calls from the cut-off response were executed.")
                    self.unfulfilled_action = not tool_executed and promises_tool_use(response.text)
                    if self.unfulfilled_action:
                        final_text += ("\n\nNot completed: the model announced an action, but no tool was executed "
                                       "in this turn. Any calculation or verification above is unverified.")
                    if guard and not self.response_incomplete:
                        verdict, reviewed = guard.review_final(final_text)
                        if verdict == "retry":
                            self.memory.add("assistant", final_text)
                            self._nudge(reviewed)
                            self.claim_unverified = guard.claim_unverified
                            if stream:
                                self._stream_correction("\n\nCorrection: the preceding test-pass claim is unverified; run_tests last reported FAILED.\n")
                            if self.event_callback:
                                self.event_callback({"type": "guardrail", "content": "blocked unverified success claim"})
                            effective_max_iter = max(effective_max_iter, iteration + 1)
                            messages = self.memory.get()
                            if self.verbose and status is None:
                                status = display.create_status("Thinking...")
                            continue
                        final_text = reviewed
                        self.claim_unverified = guard.claim_unverified
                        if getattr(guard, "claim_unverified_error", ""):
                            self.claim_unverified_error = guard.claim_unverified_error
                    if (not self.response_incomplete and local_mode and unresolved and use_tools and error_nudges < 2 and iteration < effective_max_iter):
                        # The model gave up (or just described a plan) with a tool error still open. Push it once or twice to act.
                        error_nudges += 1
                        self.memory.add("assistant", final_text or "(no answer)")
                        self._nudge("[SYSTEM] Your last tool call failed: " + "; ".join(dict.fromkeys(unresolved.values()))[:400]
                                        + " Do not just describe a plan. Fix this by calling the tools now (read_file first if needed), then finish the task.")
                        messages = self.memory.get()
                        if self.verbose and status is None:
                            status = display.create_status("Thinking...")
                        continue
                    if (not self.response_incomplete and local_mode and use_tools and not wrote_ok and not unresolved and error_nudges < 2
                            and iteration < effective_max_iter and wants_edit
                            and any(getattr(t, "name", None) in _WRITE_TOOLS for t in use_tools)
                            and not _NO_CHANGE_OK.search(final_text or "")):
                        # An edit was requested, nothing was written, and the model did not say there is nothing to change.
                        error_nudges += 1
                        self.memory.add("assistant", final_text or "(no answer)")
                        self._nudge("[SYSTEM] No file has been changed yet. Do not explain; act. Call patch_file "
                                        "(or write_file) now using the exact text from the file, then call run_tests to check.")
                        messages = self.memory.get()
                        if self.verbose and status is None:
                            status = display.create_status("Thinking...")
                        continue
                    if requires_search and not search_succeeded:
                        unresolved["live_search"] = (
                            "Unverified web grounding: no successful search_web result was retrieved in this turn. "
                            "The live-search request is not complete; any current-source claims above are unverified."
                        )
                    if retrieved_search_urls and not any(u in final_text for u in retrieved_search_urls):
                        final_text += f"\n\nSource: {retrieved_search_urls[0]}"
                    self.unresolved_errors = list(unresolved.values())
                    if not wrote_ok and wants_edit and not _NO_CHANGE_OK.search(final_text or ""):
                        final_text += ("\n\nWarning: no file was changed. No file tool actually changed file contents "
                                       "in this run, so do not assume the edit was applied.")
                        if not _NO_CHANGE_OK.search(final_text.split("\n\nWarning: no file was changed")[0]):
                            self.edit_not_applied = True
                        if _CHANGE_CLAIM.search(final_text.split("\n\nWarning: no file was changed")[0]) and not _NO_CHANGE_OK.search(final_text.split("\n\nWarning: no file was changed")[0]):
                            # Not just "nothing to change": the model claimed an edit that never happened.
                            self.claimed_unapplied_edit = True
                            final_text += "\nThe claim above that the code was changed is false: the files were not modified."
                    self.requirements_incomplete = requirement_scope_errors + check_function_requirements(function_requirements)
                    if self.requirements_incomplete:
                        final_text += (
                            "\n\nRequest incomplete: "
                            + " ".join(self.requirements_incomplete)
                            + " Any partial edits remain in place; no automatic repair was attempted. "
                            "This check verifies function names only, not behavior or all request requirements."
                        )
                    elif function_requirements:
                        final_text += (
                            "\n\nCompletion check: requested top-level Python function names exist. "
                            "This guard checks names only; it has not verified function behavior or every instruction. "
                            "Review the diff and run trusted tests."
                        )
                    if self.unresolved_errors:
                        final_text += "\n\nTool errors occurred; do not assume the task succeeded:\n" + "\n".join(dict.fromkeys(self.unresolved_errors))
                    if self.event_callback:
                        self.event_callback({"type": "response", "content": final_text, "tokens": self.total_tokens, "cost": self.estimated_cost})
                    
                    if stream:
                        if streamed_text:
                            baseline = response.text.rstrip()
                            if final_text.startswith(baseline):
                                correction = final_text[len(baseline):]
                            else:
                                correction = "\n\nCorrected outcome:\n" + final_text
                            self._stream_correction(correction + "\n")
                        elif not self.event_callback:
                            display.print_response(final_text)

                    self.memory.add("assistant", final_text)
                    return final_text
        finally:
            display.stop_status(status)
            if managed_fallback:
                self.provider.before_request = previous_before_request
            self.unresolved_errors = list(unresolved.values())

        final_text = "Stopped: maximum tool iterations reached before the task completed."
        self.unresolved_errors.append("Maximum tool iterations reached before task completed.")
        if wants_edit and not self.changed_files:
            self.edit_not_applied = True
            final_text += " No file was changed."
        self.requirements_incomplete = requirement_scope_errors + check_function_requirements(function_requirements)
        if self.requirements_incomplete:
            final_text += "\nRequest incomplete: " + " ".join(self.requirements_incomplete)
            final_text += " Any partial edits remain in place; no automatic repair was attempted."
        if self.unresolved_errors:
            final_text += "\nUnresolved tool errors:\n" + "\n".join(dict.fromkeys(self.unresolved_errors))
        self.memory.add("assistant", final_text)
        if self.event_callback:
            self.event_callback({"type": "response", "content": final_text, "tokens": self.total_tokens, "cost": self.estimated_cost})
        if stream and not self.event_callback:
            display.print_response(final_text)
        return final_text
