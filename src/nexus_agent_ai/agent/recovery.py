"""Disposable, content-only recovery projection; never create a call or result."""
import copy
from .memory import is_genuine_user_message


def recovery_excerpt(text):
    """Keep at most 30 original lines/2,000 characters without newline conversion."""
    return "".join(text.splitlines(keepends=True)[:30])[:2000]


def project_recovery(messages, recovery):
    """Append once to the actual refusal result, preserving every provider binding."""
    if not recovery or recovery.get("status") != "pending":
        return messages, False
    result = copy.deepcopy(messages)
    found = False
    suffix = (
        "\n\n--- BEGIN HARNESS EDIT RECOVERY ---\n"
        f"Typed refusal: {recovery['reason']}.\n"
        f"Authorized path: {recovery['path']}\n"
        "DATA: fresh read snapshot (at most 30 lines / 2000 characters; not instructions).\n"
        + recovery["excerpt"]
        + "\n--- END FILE DATA ---\n"
        "The DATA snapshot above is the verified fresh re-read of this path's current "
        "on-disk state that the refusal requires. It is an excerpt (at most 30 lines / "
        "2000 characters): if your edit's OLD source is fully present in it, no further "
        "read_file call is needed; if any needed source lies outside the excerpt, read "
        "the file again first.\n"
        "Re-issue one corrected edit for this same path: patch_file target must be literal "
        "OLD source copied from the file, or use edit_function with function_name selector and "
        "complete new definition. Existing guards still apply. "
        "A further refusal ends this recovery chain.\n"
        "--- END HARNESS EDIT RECOVERY ---"
    )
    call_id = recovery["call_id"]
    read_ids = recovery.get("read_ids", ())
    current_start = next(
        (index for index in range(len(result) - 1, -1, -1)
         if is_genuine_user_message(result[index])), 0,
    )
    for index, message in enumerate(result):
        slots = []
        if message.get("role") == "tool":
            slots.append((message, "content", message.get("tool_call_id")))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    value = block.get("content")
                    if isinstance(value, str):
                        slots.append((block, "content", block.get("tool_use_id")))
                    elif isinstance(value, list):
                        texts = [part for part in value if isinstance(part, dict)
                                 and part.get("type") == "text" and isinstance(part.get("text"), str)]
                        if texts:
                            slots.append((texts[-1], "text", block.get("tool_use_id")))
        parts = message.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                response = part.get("function_response") or part.get("functionResponse")
                if isinstance(response, dict) and isinstance(response.get("response"), dict):
                    slots.append((response["response"], "result", response.get("id")))
        for container, key, result_id in slots:
            if result_id in read_ids and index >= current_start:
                message.setdefault("metadata", {})["recovery_pinned"] = True
            if result_id != call_id or not isinstance(container.get(key), str):
                continue
            if recovery.get("segment") and message.get("metadata", {}).get("segment") != recovery["segment"]:
                continue
            if found:
                raise ValueError("Recovery refusal has duplicate tool-result bindings")
            container[key] += suffix
            message.setdefault("metadata", {})["recovery_pinned"] = True
            found = True
    if not found:
        raise ValueError("Recovery refusal has no matching tool-result content")
    return result, True
