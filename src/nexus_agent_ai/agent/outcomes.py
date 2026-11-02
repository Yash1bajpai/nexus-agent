"""Bounded executor observations and disposable, deterministic claim views.

An unchanged snapshot is evidence about an executor action, not file semantics.
Only an explicit actor's direct mutation assertion can contradict a failed action;
state descriptions remain intact and receive an executor-only historical note.
"""
import copy
import json
import os
import re

from .memory import is_genuine_user_message

MAX_OUTCOME_FILES = 8
MAX_OUTCOME_METADATA_CHARS = 2048
MAX_CLAUSES_PER_TURN = 3
_VERBS = r"(?:fixed|updated|changed|modified|patched|applied|corrected|replaced|rewrote|rewritten|edited|implemented|added|removed|renamed|resolved|created|written|wrote|saved|generated)"
_ADVERBS = r"(?:(?:now|successfully|just)\s+)*"
_ACTOR = re.compile(r"^\s*(?:[-*]\s+)?I(?:'ve|\s+have)?\s+" + _ADVERBS + _VERBS + r"\b\s+(?P<files>.+?)\s*[.!?;]?\s*$", re.IGNORECASE)
_PASSIVE = re.compile(r"^\s*(?:[-*]\s+)?[^\n]+?\s+(?:was|were|is|are|has been|have been)\s+" + _ADVERBS + _VERBS + r"\b", re.IGNORECASE)
_BARE = re.compile(r"^\s*(?:[-*]\s+)?(?:here(?:'s| is)\s+)?(?:your\s+|the\s+)?" + _ADVERBS + _VERBS + r"\b", re.IGNORECASE)
_SEPARATOR = re.compile(r"\band\b|\bthe\s+(?:file|script|module)\b|[\s,`'\"]+", re.IGNORECASE)
_CLAUSES = re.compile(r".+?(?:[.!?](?=\s|$)|;|\n|$)", re.DOTALL)
_CODE = re.compile(r"(```.*?```|~~~.*?~~~)", re.DOTALL)
_CONJUNCTION = re.compile(r"\s+(?:and|but)\s+(?=(?:I\b|" + _VERBS + r"\b))", re.IGNORECASE)


def _path_key(path):
    # Lexical only: never inspect the current filesystem to judge history.
    return os.path.normcase(os.path.abspath(path))


def _valid_event(event):
    if not isinstance(event, dict):
        return False
    limits = {"event": 96, "turn": 96, "segment": 96, "path": 256, "action": 64}
    return (all(isinstance(event.get(key), str) and 0 < len(event[key]) <= limit
                for key, limit in limits.items())
            and event.get("status") in ("changed", "unchanged", "unknown")
            and isinstance(event.get("success"), bool)
            and ("failed" not in event or isinstance(event["failed"], bool))
            and all(key not in event or isinstance(event[key], bool)
                    for key in ("segment_mutated", "segment_complete", "segment_failed_only"))
            and ("refusal" not in event or isinstance(event["refusal"], str) and 0 < len(event["refusal"]) <= 64))


def mutation_metadata(events):
    """Copy the newest bounded actual executor events, with no legacy status map."""
    if not isinstance(events, list):
        raise TypeError("mutation_metadata requires a list of executor events")
    records, paths, seen, omitted = [], set(), set(), set()
    for event in reversed(events):
        if not _valid_event(event) or event["event"] in seen:
            continue
        seen.add(event["event"])
        identity = (event["turn"], event["segment"])
        path = _path_key(event["path"])
        if path not in paths and len(paths) >= MAX_OUTCOME_FILES:
            omitted.add(identity)
            continue
        record = {key: event[key] for key in ("event", "turn", "segment", "path", "action", "status", "success")}
        for key in ("failed", "refusal", "segment_mutated", "segment_failed_only"):
            if key in event:
                record[key] = event[key]
        # Reserve the longer false value so clipping cannot overflow the cap
        # when it invalidates a previously selected segment's completeness.
        record["segment_complete"] = False
        candidate = {"mutation_outcomes": [record, *records]}
        if len(json.dumps(candidate, ensure_ascii=False)) <= MAX_OUTCOME_METADATA_CHARS:
            records.insert(0, record)
            paths.add(path)
        else:
            omitted.add(identity)
    complete = {(event["turn"], event["segment"]): True for event in records}
    for event in events:
        if _valid_event(event) and event.get("segment_complete") is False:
            complete[(event["turn"], event["segment"])] = False
    for record in records:
        identity = (record["turn"], record["segment"])
        record["segment_complete"] = complete[identity] and identity not in omitted
    return {"mutation_outcomes": records}


def _events(messages):
    events, seen, incomplete, mutated, nonfailure = [], set(), set(), set(), set()
    for message in messages:
        if message.get("role") not in ("assistant", "tool"):
            continue
        metadata = message.get("metadata", {})
        records = metadata.get("mutation_outcomes", [])
        if not isinstance(records, list):
            continue
        for event in mutation_metadata(records)["mutation_outcomes"]:
            identity = (event["turn"], event["segment"])
            if event.get("segment_complete") is False:
                incomplete.add(identity)
            if event.get("segment_mutated") is True:
                mutated.add((*identity, _path_key(event["path"])))
            if event.get("segment_failed_only") is False:
                nonfailure.add((*identity, _path_key(event["path"])))
            if event["event"] not in seen:
                events.append(event)
                seen.add(event["event"])
    for event in events:
        identity = (event["turn"], event["segment"])
        if identity in incomplete:
            event["segment_complete"] = False
        if (*identity, _path_key(event["path"])) in mutated:
            event["segment_mutated"] = True
        if (*identity, _path_key(event["path"])) in nonfailure:
            event["segment_failed_only"] = False
    return events


def _references(paths):
    aliases = {}
    for path in paths:
        try:
            relative = os.path.relpath(path)
        except ValueError:
            relative = path
        for alias in (path, relative, os.path.basename(path)):
            aliases.setdefault(alias, set()).add(path)
    unique = {alias: next(iter(targets)) for alias, targets in aliases.items() if len(targets) == 1}
    if not unique:
        return None, unique
    # These are escaped executor-touched paths, never arbitrary filename patterns.
    pattern = re.compile(r"(?<![\w./\\-])(?:" + "|".join(re.escape(alias) for alias in sorted(unique, key=len, reverse=True)) + r")(?![\w/\\-]|\.[\w])")
    return pattern, unique


def _clause_parts(text):
    """Yield prose clauses with source offsets; fenced code remains untouched."""
    offset = 0
    for index, piece in enumerate(_CODE.split(text)):
        if index % 2 == 0:
            for clause in _CLAUSES.finditer(piece):
                first = clause.start()
                for conjunction in _CONJUNCTION.finditer(clause.group()):
                    end = clause.start() + conjunction.start()
                    yield offset + first, offset + end, piece[first:end]
                    first = clause.start() + conjunction.end()
                yield offset + first, offset + clause.end(), piece[first:clause.end()]
        offset += len(piece)


def _claim(clause, pattern, aliases):
    """Return detected paths and direct actor-object spans, not semantic judgments."""
    if pattern is None:
        return set(), []
    if re.match(r"\s*(?:[\"']|for example\b|example\b)", clause, re.IGNORECASE):
        return set(), []
    actor = _ACTOR.fullmatch(clause)
    passive = _PASSIVE.match(clause)
    bare = _BARE.match(clause)
    if not (actor or passive or bare):
        return set(), []
    references = list(pattern.finditer(clause))
    if not references:
        return set(), []
    if actor:
        names = actor.group("files").rstrip(".!?; \r\n")
        if not _SEPARATOR.sub("", pattern.sub("", names)):
            return {aliases[ref.group()] for ref in references}, references
    # Detect explicit post-verb destinations and noun-modifier / passive shapes,
    # but never use those shapes to infer that a semantic description is false.
    detected = set()
    for reference in references:
        before = clause[:reference.start()]
        if ((passive and reference.end() <= passive.end())
                or re.search(r"\b(?:to|in)\s+[`'\"]?$", before, re.IGNORECASE)
                or (bare and not _SEPARATOR.sub("", clause[bare.end():reference.start()]))):
            detected.add(aliases[reference.group()])
    return detected, []


def _failed_paths(events):
    grouped = {}
    for event in events:
        grouped.setdefault(_path_key(event["path"]), []).append(event)
    return {path for path, records in grouped.items()
            if all(event["status"] == "unchanged" and not event["success"]
                   and event.get("failed") is True
                   and event.get("segment_complete") is not False
                   and event.get("segment_failed_only") is not False
                   and event.get("segment_mutated") is not True for event in records)}


def _executor_note(paths, events):
    facts = []
    for path in sorted(paths):
        identifiers = ", ".join(event["event"] for event in events if _path_key(event["path"]) == path)
        facts.append(f"no executor write to {json.dumps(path, ensure_ascii=False)} was observed in failed segment event(s) {identifiers}")
    return "Historical executor record only, not a new request or approval: " + "; ".join(facts) + "."


def fitted_claim_view(messages):
    """Copy audit history; bind corrections to exact executor segment identities."""
    output = []
    starts = [i for i, message in enumerate(messages) if is_genuine_user_message(message)]
    boundaries = sorted(set([0, *starts, len(messages)]))
    for first, last in zip(boundaries, boundaries[1:]):
        turn = copy.deepcopy(messages[first:last])
        events = _events(turn)
        grouped = {}
        for event in events:
            grouped.setdefault((event["turn"], event["segment"]), []).append(event)
        notes, matched = {}, 0
        for index, message in enumerate(turn):
            if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
                continue
            metadata = message.get("metadata", {})
            identity = (metadata.get("turn"), metadata.get("segment"))
            evidence = grouped.get(identity, [])
            failed = _failed_paths(evidence)
            if not failed or matched >= MAX_CLAUSES_PER_TURN:
                continue
            pattern, aliases = _references({_path_key(event["path"]) for event in evidence})
            replacements, redactions = [], []
            for start, end, clause in _clause_parts(message["content"]):
                paths, direct = _claim(clause, pattern, aliases)
                if not paths:
                    continue
                if matched >= MAX_CLAUSES_PER_TURN:
                    break
                matched += 1
                contradicted = paths & failed
                if not contradicted:
                    continue
                notes.setdefault(identity, set()).update(contradicted)
                if not direct:
                    continue
                for path in sorted(contradicted):
                    redactions.append({"turn": identity[0], "file": path, "trigger": "failed_segment_actor_mutation"})
                if all(aliases[ref.group()] in failed for ref in direct):
                    replacements.append((start, end, "[Unsupported executor-action claim omitted.]" + ("\n" if clause.endswith("\n") else "")))
                else:
                    for reference in direct:
                        if aliases[reference.group()] in failed:
                            replacements.append((start + reference.start(), start + reference.end(), "[unsupported executor-action reference omitted]"))
            for start, end, replacement in reversed(replacements):
                message["content"] = message["content"][:start] + replacement + message["content"][end:]
            if redactions:
                message["metadata"] = {**metadata, "claim_redactions": redactions[:MAX_CLAUSES_PER_TURN * MAX_OUTCOME_FILES]}
        # Corrections come after all results for their segment, never between
        # assistant tool calls and their paired tool messages.
        insertions = {}
        for identity, paths in notes.items():
            positions = [index for index, message in enumerate(turn)
                         if (message.get("metadata", {}).get("turn"), message.get("metadata", {}).get("segment")) == identity]
            if not positions:
                continue
            after = max(positions)
            while after + 1 < len(turn) and not turn[after + 1].get("metadata", {}).get("segment"):
                after += 1
            insertions.setdefault(after, []).append({
                "role": "user", "content": _executor_note(paths, grouped[identity]),
                "metadata": {"internal": True, "outcome_correction": True, "turn": identity[0], "segment": identity[1]},
            })
        for index, message in enumerate(turn):
            output.append(message)
            output.extend(insertions.get(index, []))
    return output


def current_claim_correction(text, messages, *, session_events=None):
    """Return an executor-only suffix without rewriting a current read-only reply."""
    events = _events(messages) if session_events is None else session_events
    events = mutation_metadata(events)["mutation_outcomes"]
    pattern, aliases = _references({_path_key(event["path"]) for event in events})
    touched, matched = set(), 0
    for _, _, clause in _clause_parts(text):
        paths, _ = _claim(clause, pattern, aliases)
        if not paths:
            continue
        matched += 1
        touched.update(paths)
        if matched >= MAX_CLAUSES_PER_TURN:
            break
    suffixes, audit = [], []
    for path in sorted(touched)[:MAX_CLAUSES_PER_TURN]:
        records = [event for event in events if _path_key(event["path"]) == path]
        last_success = max((index for index, event in enumerate(records) if event["success"]), default=-1)
        failed_segments = {(event["turn"], event["segment"]) for event in records[last_success + 1:]
                           if event.get("failed") is True}
        failed = [event for event in records
                  if (event["turn"], event["segment"]) in failed_segments]
        if not failed or any(event["status"] != "unchanged" or event["success"]
                             or event.get("failed") is not True
                             or event.get("segment_failed_only") is False
                             or event.get("segment_complete") is False
                             or event.get("segment_mutated") is True for event in failed):
            continue
        successful = [event for event in records if event["success"] and event["status"] == "changed"]
        identifier = successful[-1]["event"] if successful else "none recorded"
        suffixes.append(f"Executor record: no successful mutation of {json.dumps(path, ensure_ascii=False)} was observed in the failed edit segment(s); last successful mutation: {identifier}.")
        audit.append({"turn": failed[-1]["turn"], "file": path, "trigger": "current_executor_record"})
    return ("\n\n" + "\n".join(suffixes) if suffixes else ""), audit
