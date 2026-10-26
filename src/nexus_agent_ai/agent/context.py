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


def fit_messages(messages: List[Dict[str, Any]], budget_tokens: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Trim a history to fit a token budget without mutating the stored history.

    1. Drop the oldest whole turns (a turn starts at a real user message, so
       tool call/result pairs stay together). The newest turn is always kept.
    2. If the newest turn alone is too big, cut the largest text payloads
       (file contents, tool output) down, keeping the start and the end.
    """
    stats = {"dropped_messages": 0, "truncated_messages": 0, "tokens": 0}
    msgs = list(messages)
    total = history_tokens(msgs)
    if total <= budget_tokens:
        stats["tokens"] = total
        return msgs, stats

    starts = [i for i, m in enumerate(msgs) if is_genuine_user_message(m)]
    while total > budget_tokens and len(starts) > 1:
        cut = starts[1]
        stats["dropped_messages"] += cut
        msgs = msgs[cut:]
        starts = [i - cut for i in starts[1:]]
        total = history_tokens(msgs)

    if total > budget_tokens:
        # Tool-call arguments are immutable: truncating their JSON could change
        # what the assistant actually executed. Only shrink text payloads.
        limit = max(200, int(budget_tokens * CHARS_PER_TOKEN * 0.5))
        for _ in range(16):
            if total <= budget_tokens:
                break
            candidates = [
                (message_tokens(m) - message_tokens(shrunk), i, shrunk)
                for i, m in enumerate(msgs)
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
    if total > budget_tokens:
        raise ContextBudgetError(
            f"Current conversation needs at least {total} estimated tokens but only "
            f"{budget_tokens} are available. Tool arguments and call/result bindings "
            "were preserved; no over-budget request was sent. Start a new turn or use a larger context window."
        )
    stats["tokens"] = total
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


def meter(used_tokens: int, window: Optional[int]) -> str:
    if not window:
        return f"context: ~{used_tokens} tokens"
    pct = min(999, int(used_tokens * 100 / window))
    bar_len = 10
    filled = min(bar_len, pct * bar_len // 100)
    hint = ""
    if pct >= 85:
        hint = "  - nearly full, run /compact"
    elif pct >= 70:
        hint = "  - getting full"
    return f"context: [{'#' * filled}{'.' * (bar_len - filled)}] {pct}% (~{used_tokens}/{window} tokens){hint}"
