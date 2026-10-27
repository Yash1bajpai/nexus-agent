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
