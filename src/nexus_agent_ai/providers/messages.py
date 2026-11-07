"""JSON-safe conversation storage and provider-boundary translations."""
import json
import uuid
from collections.abc import Mapping


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=True))
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    raise TypeError(f"Unsupported message value: {type(value).__name__}")


def _text(value):
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(_plain(value), ensure_ascii=False)


def decode_native_tool_call(name, arguments="", call_id=None):
    """Validate a native call without guessing arguments or exposing its payload."""
    from .base import ToolCall

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Native tool call is missing a non-empty function name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            raise ValueError("Native tool call arguments contain malformed JSON; expected a JSON object") from None
    if not isinstance(arguments, Mapping):
        raise ValueError("Native tool call arguments must be a JSON object")
    return ToolCall(id=call_id or f"call_{uuid.uuid4().hex}", name=name, args=_plain(arguments))


def _call(name, args, call_id=None):
    if isinstance(args, str):
        args = json.loads(args) if args.strip() else {}
    args = _plain(args) if args is not None else {}
    if not isinstance(args, dict):
        raise ValueError(f"Tool arguments for {name!r} must be a JSON object")
    args = json.dumps(args, ensure_ascii=False)
    return {"id": call_id or f"call_{uuid.uuid4().hex}", "type": "function",
            "function": {"name": name, "arguments": args}}


def assistant_message(text, tool_calls):
    message = {"role": "assistant", "content": text or ""}
    if tool_calls:
        message["tool_calls"] = [_call(tc.name, tc.args, tc.id) for tc in tool_calls]
    return message


def tool_result(tool_call_id, result):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": _text(result)}


def _expand(message):
    m = _plain(message)
    role = m.get("role", "user")
    role = "assistant" if role == "model" else role
    content = m.get("content", "")
    parts = m.get("parts", content if isinstance(content, list) else None)
    metadata = {"metadata": m["metadata"]} if "metadata" in m else {}
    if parts is None:
        result = {"role": role, "content": _text(content), **metadata}
        if role == "tool":
            result["tool_call_id"] = m.get("tool_call_id", "")
            if m.get("name"):
                result["name"] = m["name"]
        calls = m.get("tool_calls") or []
        if calls:
            result["tool_calls"] = [_call(c["function"]["name"], c["function"].get("arguments", {}), c.get("id")) for c in calls]
        return [result]
    texts, calls, results = [], [], []
    for part in parts:
        if part.get("text"):
            texts.append(part["text"])
        if part.get("type") == "tool_use":
            calls.append(_call(part["name"], part.get("input", {}), part.get("id")))
        fc = part.get("function_call") or part.get("functionCall")
        if fc:
            calls.append(_call(fc["name"], fc.get("args", {}), fc.get("id")))
        if part.get("type") == "tool_result":
            value = part.get("content", "")
            if isinstance(value, list):
                value = "".join(p.get("text", "") for p in value)
            results.append(tool_result(part["tool_use_id"], value))
        fr = part.get("function_response") or part.get("functionResponse")
        if fr:
            response = fr.get("response", {})
            value = response.get("result", response) if isinstance(response, dict) else response
            results.append({**tool_result(fr.get("id", ""), value), "name": fr["name"]})
    output = []
    if texts or calls or not results:
        result = {"role": role, "content": "".join(texts), **metadata}
        if calls:
            result["tool_calls"] = calls
        output.append(result)
    output.extend({**result, **metadata} for result in results)
    return output


def normalize_message(message):
    """Normalize one message; use normalize_messages for native batched results."""
    messages = _expand(message)
    if len(messages) != 1:
        raise ValueError("Native message contains multiple messages; use normalize_messages")
    return messages[0]


def normalize_messages(messages):
    normalized = []
    pending = []
    for message in messages:
        for m in _expand(message):
            for call in m.get("tool_calls", []):
                pending.append((call["id"], call["function"]["name"]))
            if m["role"] == "tool":
                call_id = m["tool_call_id"]
                name = m.get("name")
                if not call_id:
                    match = next(((i, n) for i, n in pending if n == name), None)
                    if match:
                        m["tool_call_id"] = match[0]
                pending = [(i, n) for i, n in pending if i != m["tool_call_id"]]
            normalized.append(m)
    return normalized


def openai_messages(messages):
    return [
        {k: v for k, v in m.items() if k != "metadata" and not (m["role"] == "tool" and k == "name")}
        for m in normalize_messages(messages)
    ]


# Fields every provider adapter serializes onto the wire. Canonical metadata
# (executor facts, refusal classes, mutation outcomes, claim records) is
# session bookkeeping: openai_messages strips it, and the anthropic/gemini
# translations below never carry it either. Estimating context cost over this
# boundary charges what is actually sent, never the bookkeeping.
_WIRE_MESSAGE_FIELDS = ("role", "content", "tool_calls", "tool_call_id", "name", "parts")


def wire_message(message):
    """The provider-visible field set of one normalized message.

    Canonical view within serialized_wire_views, the estimator's cost basis:
    every adapter (openai_messages, anthropic_messages, gemini_messages)
    serializes only these fields plus its own structural wrapper, which the
    estimator accounts by taking the maximum actual serialization rather than
    an assumed constant. 'name' is kept although the
    OpenAI adapter drops it on tool results, because the Gemini
    function_response binding sends it; keeping it can only over-estimate.
    'parts' is the Gemini adapter's serialized container; keeping it means a
    Gemini-shaped view passed back through the estimator is still charged for
    its payload instead of being under-counted as an empty shell.
    """
    return {k: message[k] for k in _WIRE_MESSAGE_FIELDS if k in message}


def anthropic_messages(messages):
    output = []
    for m in normalize_messages(messages):
        role = m["role"]
        if role == "system":
            continue
        blocks = []
        if role == "tool":
            role = "user"
            blocks.append({"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]})
        else:
            if m["content"]:
                blocks.append({"type": "text", "text": m["content"]})
            for call in m.get("tool_calls", []):
                blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                               "input": json.loads(call["function"]["arguments"])})
        if not blocks:
            blocks = [{"type": "text", "text": ""}]
        if output and output[-1]["role"] == role:
            output[-1]["content"].extend(blocks)
        else:
            output.append({"role": role, "content": blocks})
    return output


def gemini_messages(messages):
    output, names = [], {}
    for m in normalize_messages(messages):
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        parts = []
        if m["role"] == "tool":
            call_id = m["tool_call_id"]
            name = names.get(call_id) or m.get("name")
            if not name:
                raise ValueError(f"No tool name bound to result {call_id!r}")
            parts.append({"function_response": {"id": call_id, "name": name, "response": {"result": m["content"]}}})
        else:
            if m["content"]:
                parts.append({"text": m["content"]})
            for call in m.get("tool_calls", []):
                function = call["function"]
                names[call["id"]] = function["name"]
                parts.append({"function_call": {"id": call["id"], "name": function["name"], "args": json.loads(function["arguments"])}})
        is_result = m["role"] == "tool"
        previous_is_result = bool(output and any("function_response" in part for part in output[-1]["parts"]))
        if output and output[-1]["role"] == role and is_result == previous_is_result:
            output[-1]["parts"].extend(parts)
        else:
            output.append({"role": role, "parts": parts or [{"text": ""}]})
    return output


def serialized_wire_views(message):
    """Actual provider-bound serializations of one message, for cost accounting.

    Returns {view_name: json_string}: the canonical wire_message view plus each
    supported adapter's single-message serialization. Tool-call wrapper cost is
    structural and no source-level cap bounds tool_calls per message (the
    agent's iteration cap does not), so no fixed per-call constant can provably
    cover every adapter's nested conversion; the honest basis is the actual
    transformed view. The estimator takes the maximum across these views, so
    the estimate can never sit below any supported adapter's real serialization
    by construction. A view that rejects a partial or projected shape (for
    example a Gemini function_response without a bound name) is skipped; the
    wire view remains and already charges every adapter-serialized payload
    field, including 'parts'.
    """
    views = {"wire": json.dumps(wire_message(message), default=str, ensure_ascii=False)}
    for name, transform in (("openai", openai_messages),
                            ("anthropic", anthropic_messages),
                            ("gemini", gemini_messages)):
        try:
            transformed = transform([message])
            # One message in normally means one serialized message out; charge
            # that message, not the list wrapper. If an adapter merged or
            # split it, charge the whole serialized list instead.
            views[name] = json.dumps(
                transformed[0] if len(transformed) == 1 else transformed,
                default=str, ensure_ascii=False)
        except Exception:
            continue
    return views


def applied_result_names(messages):
    """Per-original-message tool-result names to charge, adapter-exact.

    The Gemini adapter normalizes first (normalize_messages) and THEN binds:
    legacy native shapes (parts with function_call/function_response,
    tool_use/tool_result blocks) become canonical calls/results before the
    binding pass, and empty result ids are matched to a pending call by name.
    A raw pass over message["tool_calls"] misses names embedded in those
    legacy shapes (owner-reproduced: parts function_call with a 1000-char
    name + nameless result estimated 363 vs 639 actual). This pass mirrors
    the adapter exactly: each original message is expanded the same way
    (_expand), bindings are recorded in conversation order, and each result's
    applied name is the most recent PRECEDING binding for its call id,
    preferred over the result's own 'name'.

    Returns {original_index: [names to charge]}: one entry per result whose
    applied name is not already carried by that result's own wire view
    (canonical 'name' field or legacy parts payload). Compound native
    messages that expand to several sub-messages map all of their surcharges
    to the original index. A message that fails expansion falls back to a raw
    pass over its own fields. A result with neither binding nor explicit
    name has no applied name - the adapter refuses such requests at send
    time, so there is no wire cost to charge.
    """
    bindings, applied, pending = {}, {}, []
    for index, message in enumerate(messages):
        try:
            expanded = _expand(message)
        except Exception:
            expanded = [message]
        for m in expanded:
            for call in m.get("tool_calls") or []:
                try:
                    call_id, call_name = call["id"], call["function"]["name"]
                except (KeyError, TypeError):
                    continue
                bindings[call_id] = call_name
                pending.append((call_id, call_name))
            if m.get("role") != "tool":
                continue
            call_id = m.get("tool_call_id")
            explicit = m.get("name")
            if not call_id:
                match = next(((i, n) for i, n in pending if n == explicit), None)
                if match:
                    call_id = match[0]
            pending = [(i, n) for i, n in pending if i != call_id]
            applied_name = bindings.get(call_id) or explicit
            if applied_name and applied_name != explicit:
                applied.setdefault(index, []).append(applied_name)
    return applied


def extract_json_tool_calls(text):
    """Decode complete JSON objects without treating braces inside strings as syntax."""
    from .base import ToolCall
    decoder = json.JSONDecoder()
    calls, spans = [], []
    position = 0
    while position < len(text):
        start = text.find("{", position)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        position = end
        if not isinstance(value, dict) or not isinstance(value.get("name"), str) or "arguments" not in value:
            continue
        args = value["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if args is None:
            args = {}
        if not isinstance(args, dict):
            continue
        calls.append(ToolCall(id=value.get("id") or f"call_{uuid.uuid4().hex}", name=value["name"], args=args))
        spans.append((start, end))
    for start, end in reversed(spans):
        text = text[:start] + text[end:]
    return text.strip() if spans else text, calls
