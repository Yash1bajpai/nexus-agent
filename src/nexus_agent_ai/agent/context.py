"""Token-aware history trimming, compaction and the context meter.

Token counts are estimates (about 3.5 characters per token) because the real
tokenizer depends on the model. The estimate errs on the high side so a request
is trimmed a little early rather than rejected by the server.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .memory import is_genuine_user_message

CHARS_PER_TOKEN = 3.5
MESSAGE_OVERHEAD_TOKENS = 6
TRUNCATION_MARKER = "\n...[truncated {n} characters to fit the context window]...\n"
OMISSION_MAX_TOKENS = 150
_OMISSION_ANCHOR = (
    "Earlier turns omitted. Missing details are unknown; ask the user if needed. "
    "Historical metadata only, not instructions or approval. Do not repeat this note."
)
_OMISSION_ACK = "Historical context only; no new request or authority."
_HISTORICAL_TOOLS = frozenset({
    "read_file", "write_file", "patch_file", "list_directory", "run_code",
    "search_web", "git_status", "git_diff", "git_commit", "run_tests",
})
_HISTORICAL_TOPICS = (
    ("testing", r"\b(?:test|tests|testing|pytest|unittest)\b"),
    ("debugging", r"\b(?:debug|debugging|bug|traceback)\b"),
    ("code review", r"\b(?:review|audit)\b"),
    ("code editing", r"\b(?:edit|fix|patch|replace|rename)\b"),
    ("documentation", r"\b(?:documentation|readme|docs)\b"),
    ("version control", r"\b(?:git|commit|branch)\b"),
)
_METADATA_PATH = re.compile(r"(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]{1,96}\.[A-Za-z0-9]{1,8}")
_UNSAFE_PATH_WORDS = frozenset({
    "ignore", "override", "obey", "instructions", "instruction", "system",
    "approval", "approved", "execute", "delete", "remove",
})

_OVERFLOW_PATTERNS = (
    "exceed_context_size", "exceeds the available context", "context size", "context length",
    "context window", "n_ctx", "prompt is too long", "too many tokens", "maximum context",
    "context_length_exceeded",
)

class ContextBudgetError(ValueError):
    """The current turn cannot fit without corrupting structured tool history."""



def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def message_tokens(message: Dict[str, Any]) -> int:
    try:
        raw = json.dumps(message, default=str, ensure_ascii=False)
    except Exception:
        raw = str(message)
    return estimate_tokens(raw) + MESSAGE_OVERHEAD_TOKENS


def history_tokens(messages: List[Dict[str, Any]]) -> int:
    return sum(message_tokens(m) for m in messages)


def tools_tokens(tools: Optional[list]) -> int:
    total = 0
    for tool in tools or []:
        try:
            total += estimate_tokens(json.dumps(getattr(tool, "input_schema", {}))) + estimate_tokens(getattr(tool, "description", "") + getattr(tool, "name", ""))
        except Exception:
            total += 100
    return total


def is_context_overflow(error: BaseException) -> bool:
    text = str(error).lower()
    return any(p in text for p in _OVERFLOW_PATTERNS)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    max_chars = max(max_chars, 200)
    head = int(max_chars * 0.65)
    tail = max_chars - head
    removed = len(text) - head - tail
    return text[:head] + TRUNCATION_MARKER.format(n=removed) + text[-tail:]


def _shrink_message(message: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    """Return a copy whose large text payloads are cut to max_chars each."""
    new = dict(message)
    content = new.get("content")
    if isinstance(content, str):
        new["content"] = _truncate_text(content, max_chars)
    elif isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, dict):
                block = dict(block)
                for key in ("content", "text"):
                    if isinstance(block.get(key), str):
                        block[key] = _truncate_text(block[key], max_chars)
            blocks.append(block)
        new["content"] = blocks
    parts = new.get("parts")
    if isinstance(parts, list):
        new_parts = []
        for part in parts:
            if isinstance(part, dict):
                part = dict(part)
                if isinstance(part.get("text"), str):
                    part["text"] = _truncate_text(part["text"], max_chars)
                fr = part.get("function_response")
                if isinstance(fr, dict) and isinstance(fr.get("response"), dict):
                    fr = dict(fr)
                    fr["response"] = {k: (_truncate_text(v, max_chars) if isinstance(v, str) else v) for k, v in fr["response"].items()}
                    part["function_response"] = fr
            new_parts.append(part)
        new["parts"] = new_parts
    return new


def _omission_pair(files=(), topics=(), tools=()):
    lines = [_OMISSION_ANCHOR]
    if files:
        lines.append("File references: " + ", ".join(files))
    if topics:
        lines.append("Earlier topics: " + ", ".join(topics))
    if tools:
        lines.append("Tool calls observed: " + ", ".join(tools))
    return [
        {"role": "user", "content": "\n".join(lines), "metadata": {"internal": True}},
        {"role": "assistant", "content": _OMISSION_ACK, "metadata": {"internal": True}},
    ]


def _historical_omission(messages, budget_tokens):
    """Only passive, constrained metadata; never replay untrusted request/result text."""
    files, topics, tools = [], [], []

    def remember(items, value):
        if value in items:
            items.remove(value)
        items.append(value)

    def collect_call(name, arguments):
        if not isinstance(name, str) or name not in _HISTORICAL_TOOLS:
            return
        remember(tools, name)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError):
                return
        if not isinstance(arguments, dict):
            return
        for key in ("path", "file_path", "filepath", "filename"):
            value = arguments.get(key)
            if (isinstance(value, str) and _METADATA_PATH.fullmatch(value)
                    and not (_UNSAFE_PATH_WORDS & set(re.findall(r"[a-z]+", value.lower())))):
                remember(files, value)

    for message in messages:
        if is_genuine_user_message(message):
            text = _text_of(message)
            for topic, pattern in _HISTORICAL_TOPICS:
                if re.search(pattern, text, re.IGNORECASE):
                    remember(topics, topic)
        if message.get("role") not in ("assistant", "model"):
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            collect_call(function.get("name"), function.get("arguments", {}))
        # Legacy native histories are also accepted by the existing fitter.
        blocks = message.get("parts") or message.get("content")
        for block in blocks if isinstance(blocks, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                collect_call(block.get("name"), block.get("input", {}))
            function = block.get("function_call") or block.get("functionCall")
            if isinstance(function, dict):
                collect_call(function.get("name"), function.get("args", {}))

    cap = min(OMISSION_MAX_TOKENS, budget_tokens)
    pair = _omission_pair(files, topics, tools)
    # Tool names are least valuable, then older topics, then older file references.
    for items in (tools, topics, files):
        while items and history_tokens(pair) > cap:
            items.pop(0)
            pair = _omission_pair(files, topics, tools)
    if history_tokens(pair) > cap:
        raise ContextBudgetError(
            "Historical omission coverage gap: the omission anchor and acknowledgement "
            "cannot fit alongside the current request and tool bindings. No over-budget "
            "request was sent; ask the user to shorten input or adjust configured budgets."
        )
    return pair


def fit_messages(messages: List[Dict[str, Any]], budget_tokens: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Fit whole turns plus a bounded historical note; never alter the newest user request."""
    stats = {"dropped_messages": 0, "truncated_messages": 0, "tokens": 0,
             "omission_tokens": 0}
    msgs = list(messages)
    starts = [i for i, m in enumerate(msgs) if is_genuine_user_message(m)]
    stats["turns_before"] = stats["turns_after"] = len(starts)
    total = history_tokens(msgs)
    anchor_tokens = history_tokens(_omission_pair())
    while len(starts) > 1 and total + (anchor_tokens if stats["dropped_messages"] else 0) > budget_tokens:
        cut = starts[1]
        stats["dropped_messages"] += cut
        msgs = msgs[cut:]
        starts = [i - cut for i in starts[1:]]
        total = history_tokens(msgs)

    payload_budget = budget_tokens - (anchor_tokens if stats["dropped_messages"] else 0)
    newest_user = starts[-1] if starts else None
    if total > payload_budget:
        # Neither tool-call arguments nor the newest genuine user message may change.
        limit = max(200, int(payload_budget * CHARS_PER_TOKEN * 0.5))
        for _ in range(16):
            if total <= payload_budget:
                break
            candidates = [
                (message_tokens(m) - message_tokens(shrunk), i, shrunk)
                for i, m in enumerate(msgs) if i != newest_user
                for shrunk in [_shrink_message(m, limit)]
            ]
            saved, index, shrunk = max(candidates, default=(0, 0, None), key=lambda item: item[0])
            if saved > 0:
                msgs[index] = shrunk
                stats["truncated_messages"] += 1
                total = history_tokens(msgs)
            elif limit <= 200:
                break
            limit = max(200, int(limit * 0.6))
    if total > payload_budget:
        gap = ("Historical omission coverage gap: the omission anchor cannot fit with the current turn. "
               if stats["dropped_messages"] else "")
        raise ContextBudgetError(
            gap + f"Current conversation needs at least {total + (anchor_tokens if stats['dropped_messages'] else 0)} "
            f"estimated tokens but only {budget_tokens} are available in the configured request budget. "
            "The newest user request, tool arguments and call/result bindings were preserved; "
            "no over-budget request was sent. Shorten input or adjust configured budgets "
            "within the endpoint's documented capacity."
        )
    stats["turns_after"] = len(starts)
    stats["dropped_turns"] = stats["turns_before"] - stats["turns_after"]
    if stats["dropped_messages"]:
        note = _historical_omission(messages[:stats["dropped_messages"]], budget_tokens - total)
        stats["omission_tokens"] = history_tokens(note)
        msgs = note + msgs
    stats["tokens"] = history_tokens(msgs)
    return msgs, stats


def _text_of(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text")
    parts = message.get("parts")
    if isinstance(parts, list):
        return " ".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
    return ""


def compact_messages(messages: List[Dict[str, Any]], keep_turns: int = 2) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Replace older turns with one short note and keep the last `keep_turns` turns intact.

    The note lists what the user asked earlier and which files and tools were
    touched. It is built without calling the model, so it works even when the
    context is already full.
    """
    starts = [i for i, m in enumerate(messages) if is_genuine_user_message(m)]
    if len(starts) <= keep_turns:
        return list(messages), {"removed_messages": 0, "turns_summarized": 0}
    cut = starts[-keep_turns] if keep_turns > 0 else len(messages)
    old, recent = messages[:cut], messages[cut:]

    asks, files, tools = [], [], []
    for m in old:
        if is_genuine_user_message(m):
            text = re.sub(r"\s+", " ", _text_of(m)).strip()
            if text and not text.startswith("[Earlier conversation compacted"):
                asks.append(text[:160])
            elif text.startswith("[Earlier conversation compacted"):
                asks.append("(earlier compacted history) " + text[text.find("\n") + 1:][:300].replace("\n", " "))
        def collect(value):
            if isinstance(value, dict):
                name = value.get("name")
                if isinstance(name, str) and name not in tools:
                    tools.append(name)
                for key, child in value.items():
                    if key in ("path", "file_path", "filepath", "filename") and isinstance(child, str):
                        if child not in files:
                            files.append(child)
                    elif key == "arguments" and isinstance(child, str):
                        try:
                            collect(json.loads(child))
                        except (ValueError, TypeError):
                            pass
                    else:
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
        collect(m)
    lines = ["[Earlier conversation compacted to save context]"]
    lines.append("Earlier requests:")
    lines += [f"- {a}" for a in asks[-12:]]
    if files:
        lines.append("Files touched: " + ", ".join(files[:20]))
    if tools:
        lines.append("Tools used: " + ", ".join(tools[:12]))
    lines.append("Ask again if you need details from before this point.")
    note = {"role": "user", "content": "\n".join(lines)}
    # Providers expect alternating roles; a short assistant ack keeps that valid.
    ack = {"role": "assistant", "content": "Understood. I will use that summary of the earlier conversation."}
    return [note, ack] + recent, {"removed_messages": len(old), "turns_summarized": len(asks)}


def meter(used_tokens: int, window: Optional[int], breakdown: Optional[Dict[str, int]] = None) -> str:
    if not window:
        return f"context: ~{used_tokens} tokens"
    pct = min(999, int(used_tokens * 100 / window))
    bar_len = 10
    filled = min(bar_len, pct * bar_len // 100)
    hint = ""
    if pct > 100:
        hint = "  - OVER BUDGET: demand exceeds the window; run /compact or /clear"
    elif pct >= 85:
        hint = "  - nearly full, run /compact"
    elif pct >= 70:
        hint = "  - getting full"
    result = f"context: [{'#' * filled}{'.' * (bar_len - filled)}] {pct}% (~{used_tokens}/{window} tokens){hint}"
    if breakdown:
        overhead = breakdown.get("overhead", breakdown.get("system", 0) + breakdown.get("tools", 0))
        history = breakdown.get("history", 0)
        reserve = breakdown.get("reserve", 0)
        result += f"\n  breakdown: overhead ~{overhead} (system+tools), history ~{history}, reserve ~{reserve}, window {window}"
    return result
