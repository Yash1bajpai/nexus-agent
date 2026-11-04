"""Pure B evidence checks. Effects, assistant claims and environment stay separate."""
import json
import re

PASS, FAIL, REVIEW = "PASS", "FAIL", "REVIEW"
ENVIRONMENT_BLOCKED, NOT_RUN = "ENVIRONMENT_BLOCKED", "NOT_RUN"
HARNESS_ERROR, INCOMPLETE = "HARNESS_ERROR", "INCOMPLETE"
ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text):
    return ANSI_ESCAPE.sub("", text or "")


def _axis(status, *evidence):
    return {"status": status, "evidence": list(evidence)}


def _row(name, status, evidence, reply=REVIEW):
    return {"name": name, "task": _axis(status, evidence), "reply": _axis(reply, "Independent claim review" if reply == REVIEW else evidence)}


def _plain(message):
    raw = message.get("message_json")
    try:
        payload = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        payload = {}
    return dict(payload if isinstance(payload, dict) else {}, **{k: v for k, v in message.items() if v is not None})


def tool_evidence(messages, name, path=None):
    """Only successful results paired to the exact canonical function call count."""
    items = [_plain(m) for m in messages]
    calls = {}
    for m in items:
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {})
            if fn.get("name") != name or not call.get("id"):
                continue
            try:
                args = fn.get("arguments", {})
                args = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                continue
            if path is None or (isinstance(args, dict) and str(args.get("path", "")).replace("\\", "/").split("/")[-1] == path):
                calls[call["id"]] = call
    results = [str(m.get("content", "")) for m in items if m.get("role") == "tool" and m.get("tool_call_id") in calls]
    return results


def _successful(results):
    return [r for r in results if r.strip() and not re.match(r"\s*ERROR\b", r, re.I)]


def parse_tty_probe(text):
    m = re.search(r"__NEXUS_TTY_PROBE__:\s*isatty=(\w+)\s*win_console=(\w+)\s*interactive=(\w+)", strip_ansi(text))
    return {"observed": bool(m), **{k: bool(m and m.group(i).lower() == "true") for i, k in enumerate(("isatty", "win_console", "interactive"), 1)}}


def parse_context_output(text):
    text = " ".join(strip_ansi(text).split())
    patterns = {
        "window": r"Configured context budget:\s*(\d+)", "config_ceiling": r"configured output ceiling:\s*(\d+)",
        "stored_demand_total": r"Stored demand:\s*~(\d+)/", "stored_window": r"Stored demand:\s*~\d+/(\d+)",
        "system": r"system\s*~(\d+)", "tools": r"tools\s*~(\d+)", "overhead": r"system\+tools overhead\s*~(\d+)",
        "history": r"full stored history\s*~(\d+)", "reserve": r"output reserve\s*~(\d+)",
        "projection": r"Next request projection:\s*~(\d+)/", "projection_window": r"Next request projection:\s*~\d+/(\d+)",
        "fitted": r"fitted history\s*~(\d+)", "omission": r"omission note\+ack\s*~(\d+)", "free": r"Free for new input:\s*~(\d+)",
    }
    return {k: int(m.group(1)) for k, pattern in patterns.items() if (m := re.search(pattern, text))}


def validate_context_metrics(m):
    required = {"window", "config_ceiling", "stored_demand_total", "system", "tools", "overhead", "history", "reserve", "projection", "fitted", "omission", "free"}
    if missing := required - m.keys():
        return {"status": HARNESS_ERROR, "errors": ["Missing context fields: " + ", ".join(sorted(missing))]}
    checks = {
        "overhead": m["overhead"] == m["system"] + m["tools"],
        "stored demand": m["stored_demand_total"] == m["overhead"] + m["history"] + m["reserve"],
        "projection (omission already included)": m["projection"] == m["overhead"] + m["fitted"] + m["reserve"],
        "free input": m["free"] == m["window"] - m["projection"],
        "budget": m["window"] == 4096 and m["config_ceiling"] == 512 and 0 < m["reserve"] <= 512,
        "denominators": m.get("stored_window", 4096) == m.get("projection_window", 4096) == 4096,
        "omission bounds": 0 <= m["omission"] <= m["fitted"],
        "fitted request bounds": m["projection"] <= 4096 and m["free"] >= 0,
    }
    errors = [name + " inconsistent" for name, valid in checks.items() if not valid]
    return {"status": FAIL if errors else PASS, "errors": errors}


def parse_compact_output(text):
    m = re.search(r"Compacted\s*(\d+)\s*earlier turn\(s\):\s*history\s*~(\d+)\s*->\s*~(\d+)\s*tokens", " ".join(strip_ansi(text).split()))
    n, before, after = map(int, m.groups()) if m else (0, 0, 0)
    return {"compacted": n > 0 and before > after, "turns_summarized": n, "before_tokens": before, "after_tokens": after}


def extract_assistant_turn_answer(messages):
    answers = [m["content"].strip() for m in messages if m.get("role") == "assistant" and not m.get("tool_calls") and isinstance(m.get("content"), str) and m["content"].strip()]
    return answers[-1] if answers else None


def assess_task_readme(answer, tool_messages, user_message, expected_prompt):
    backspace = bool(user_message and user_message.get("content", "").strip() == expected_prompt)
    answer_ok = all(s in (answer or "").lower() for s in ("harbor desk", "mira chen"))
    read_ok = bool(_successful(tool_evidence(tool_messages, "read_file", "readme.txt")))
    row = _row("row-01-readme", PASS if answer_ok and read_ok and backspace else FAIL, {"read_bound": read_ok, "answer_correct": answer_ok}, PASS if answer_ok and read_ok else REVIEW)
    row["interactive"] = _axis(PASS if backspace else HARNESS_ERROR, {"backspace_exact": backspace})
    return row


def assess_task_greeting(greeting_bytes, tool_messages):
    mutation = any(_successful(tool_evidence(tool_messages, name, "greeting.txt")) for name in ("write_file", "patch_file"))
    return _row("row-02-greeting", PASS if mutation and greeting_bytes == b"Hello Harbor Desk\n" else FAIL, {"mutation_bound": mutation, "exact_bytes": greeting_bytes == b"Hello Harbor Desk\n"})


def assess_task_calculator_review(answer, before_src, after_src, tool_messages, is_slash=False):
    name = "row-04-slash-review" if is_slash else "row-03-calculator-review"
    if before_src != after_src or any(tool_evidence(tool_messages, n) for n in ("write_file", "patch_file")):
        return _row(name, FAIL, "Read-only review changed a file or executed a mutation")
    if not is_slash and not _successful(tool_evidence(tool_messages, "read_file", "calculator.py")):
        return _row(name, FAIL, "No bound successful read of calculator.py")
    text = " ".join((answer or "").lower().split())
    if "available repl commands" in text:
        return _row(name, FAIL, "Help is not a bug review")
    explicit = re.search(r"\bmultiply\b[^.!?]{0,140}\b(?:adds|adding|addition|a\s*\+\s*b|sum)\b[^.!?]{0,100}\b(?:instead|rather|should|incorrect|wrong|bug)", text)
    explicit = explicit or re.search(r"\bmultiply\b[^.!?]{0,100}\b(?:incorrectly|wrongly|bug)\b[^.!?]{0,100}\b(?:adds|adding|addition|a\s*\+\s*b|sum)\b", text)
    valid = bool(explicit) and not re.search(r"\b(?:no bug|not a bug)\b", text)
    return _row(name, PASS if valid else REVIEW, "Multiply addition bug explicitly identified" if valid else "Ambiguous bug explanation; inspect assistant text", PASS if valid else REVIEW)


def assess_task_calculator_fix(behavior_verified, add_preserved, before_src, after_src, tool_messages):
    mutation = any(_successful(tool_evidence(tool_messages, name, "calculator.py")) for name in ("write_file", "patch_file"))
    valid = mutation and before_src != after_src and behavior_verified and add_preserved
    return _row("row-05-calculator-fix", PASS if valid else FAIL, {"mutation_bound": mutation, "behavior_verified": behavior_verified, "add_preserved": add_preserved})


_RELEASE_URL = re.compile(r"https?://github\.com/(?:ggerganov|ggml-org)/llama\.cpp/releases(?:/tag/[^\s\]\)<>\"']+)?", re.I)
_TAG = re.compile(r"\b(?:b\d{3,}|v\d+\.\d+(?:\.\d+)?)\b", re.I)


def assess_task_web_search(answer, tool_messages):
    results = tool_evidence(tool_messages, "search_web")
    successful = _successful(results)
    errors = [r for r in results if re.match(r"\s*ERROR\b", r, re.I)]
    outage = any(re.search(r"http\s*[45]\d{2}|connection.*(?:failed|refused|reset)|timed?\s*out|timeout|network|rate.limit|getaddrinfo", r, re.I) for r in errors)
    text = answer or ""
    answer_urls = {m.rstrip(".,;").lower() for m in _RELEASE_URL.findall(text)}
    answer_tags = {m.lower() for m in _TAG.findall(text)}
    evidence = "\n".join(successful)
    evidence_urls = {m.rstrip(".,;").lower() for m in _RELEASE_URL.findall(evidence)}
    evidence_tags = {m.lower() for m in _TAG.findall(evidence)}

    evidence_record = {
        "search_evidence": successful,
        "outages": errors,
        "assistant_answer": text,
        "answer_urls": sorted(answer_urls),
        "answer_tags": sorted(answer_tags),
        "evidence_urls": sorted(evidence_urls),
        "evidence_tags": sorted(evidence_tags),
    }

    if outage:
        claimed_release = bool(answer_urls and answer_tags)
        honest = bool(re.search(r"cannot|can't|unable|unavailable|failed|could not|error", text, re.I)) and not claimed_release
        row = _row("row-06-web-search", ENVIRONMENT_BLOCKED if not claimed_release else FAIL, evidence_record, PASS if honest else REVIEW if not claimed_release else FAIL)
    elif not successful:
        row = _row("row-06-web-search", FAIL, evidence_record, REVIEW)
    else:
        # Semantic checks all REVIEW for independent reading (preferred minimal option).
        # No PASS from independent tag/url intersections.
        row = _row("row-06-web-search", REVIEW, evidence_record, REVIEW)

    row["network"] = _axis(ENVIRONMENT_BLOCKED if outage and not successful else PASS if successful else REVIEW, "Only observed tool results establish network state")
    return row


CANONICAL_BRIEF_PROMPT = "Read brief.txt and summarize it in four short bullet points."
_BULLET_ITEM_RE = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+(.*)$")


def _norm_tokens(text):
    return " ".join(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def extract_bullet_items(text):
    if not text:
        return []
    items = []
    current = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                items.append(" ".join(current).strip())
                current = []
            continue
        m = _BULLET_ITEM_RE.match(line)
        if m:
            if current:
                items.append(" ".join(current).strip())
            current = [m.group(1).strip()]
        elif current:
            current.append(line.strip())
    if current:
        items.append(" ".join(current).strip())
    return items


def assess_task_brief(answer, tool_messages):
    messages = [_plain(message) for message in tool_messages]
    read_results = tool_evidence(messages, "read_file", "brief.txt")
    executed = [message for message in messages if message.get("role") != "tool"
                or (message.get("metadata", {}).get("executed") is True
                    and not message.get("metadata", {}).get("refusal_class"))]
    successful = _successful(tool_evidence(executed, "read_file", "brief.txt"))
    read_ok = bool(successful)

    items = extract_bullet_items(answer or "")
    exact_four = len(items) == 4
    nontrivial = all(len(re.findall(r"[a-zA-Z]+", it)) >= 2 for it in items)
    not_prompt_echo = _norm_tokens(" ".join(items)) != _norm_tokens(CANONICAL_BRIEF_PROMPT)

    items_ok = exact_four and nontrivial and not_prompt_echo
    valid = read_ok and items_ok

    evidence_record = {
        "read_evidence": read_results,
        "assistant_answer": answer or "",
        "bullet_items": items,
        "read_bound": read_ok,
        "four_substantive_items": items_ok,
    }

    status = REVIEW if valid else FAIL
    return _row("compact-turn-02-brief", status, evidence_record, REVIEW)


def assess_compact_recall(answer, expected_codename="Amber-Kite-731"):
    text = (answer or "").lower()
    if expected_codename.lower() in text:
        # Seed substring presence requires independent reading; uncertainty cannot PASS.
        return _row("compact-row-08-recall", REVIEW, "Earlier seed mentioned in assistant answer; requires independent reading", REVIEW)
    honest = bool(re.search(r"don't (?:know|recall)|do not (?:know|recall)|cannot remember|can't remember|not remember|unknown|forgot", text))
    return _row("compact-row-08-recall", FAIL, "Earlier seed not recalled", PASS if honest else REVIEW)

def assess_compact_slash(compact_parsed, session_messages, seed_prefix="[Earlier conversation compacted"):
    notes = [m.get("content", "") for m in session_messages if m.get("role") == "user" and seed_prefix.lower() in str(m.get("content", "")).lower()]
    valid = compact_parsed.get("compacted") and any("Amber-Kite-731" in n for n in notes)
    return _row("compact-row-06-slash-compact", PASS if valid else FAIL, {"transition": compact_parsed, "seed_in_note": any("Amber-Kite-731" in n for n in notes)})


def assess_context_command(metrics, step_label):
    validation = validate_context_metrics(metrics)
    return _row("compact-context-" + step_label, validation["status"], validation["errors"] or metrics, validation["status"])


def assess_cleanup(owned_metadata, killed_pids, remaining_llama, confirmed, errors):
    valid = bool(confirmed and owned_metadata and not remaining_llama and not errors)
    return _axis(PASS if valid else INCOMPLETE, {"owned": len(owned_metadata), "remaining_llama": remaining_llama, "errors": errors})


REQUIRED_TASKS = ("row-01-readme", "row-02-greeting", "row-03-calculator-review", "row-04-slash-review", "row-05-calculator-fix", "row-06-web-search", "compact-turn-02-brief", "compact-context-before", "compact-row-06-slash-compact", "compact-context-after", "compact-row-08-recall")


def summarize_axes(rows, cleanup_axis, tty_axis, model_initial_ok, model_final_ok, budget_expired):
    indexed = {r["name"]: r for r in rows}
    statuses = [indexed.get(n, {}).get("task", {}).get("status", NOT_RUN) for n in REQUIRED_TASKS]
    pending = any(s in (NOT_RUN, INCOMPLETE, HARNESS_ERROR) for s in statuses) or budget_expired
    task = FAIL if FAIL in statuses else INCOMPLETE if pending else ENVIRONMENT_BLOCKED if ENVIRONMENT_BLOCKED in statuses else REVIEW if REVIEW in statuses else PASS
    reply_statuses = [r.get("reply", {}).get("status", REVIEW) for r in rows if r.get("name") in REQUIRED_TASKS]
    needs_review = any(s != PASS for s in reply_statuses) or len(reply_statuses) != len(REQUIRED_TASKS)
    exit_rows = [r.get("executor", {}).get("status", NOT_RUN) for r in rows if "exit" in r.get("name", "")]
    execution = PASS if len(exit_rows) == 2 and all(s == PASS for s in exit_rows) else INCOMPLETE
    automatic = task == PASS and execution == PASS and cleanup_axis.get("status") == PASS and tty_axis.get("status") == PASS and model_initial_ok and model_final_ok and not budget_expired
    return {"qualified": bool(automatic and not needs_review), "automated_core_pass": bool(automatic), "independent_review_required": needs_review,
            "axes": {"task": task, "reply": REVIEW if needs_review else PASS, "executor": execution,
                     "interactive": tty_axis.get("status", INCOMPLETE), "cleanup": cleanup_axis.get("status", INCOMPLETE),
                     "model_sha": PASS if model_initial_ok and model_final_ok else INCOMPLETE, "budget": INCOMPLETE if budget_expired else PASS}}
