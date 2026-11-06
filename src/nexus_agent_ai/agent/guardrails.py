"""Loop guardrails: check the model's work against real tool results.

Small models hallucinate file names, claim tests pass when they failed, and
describe git output they never read. These checks run inside the agent loop,
check tool evidence and explicitly requested Python bindings, and never change files.

Turn off with NEXUS_GUARDRAILS=0 (for measuring the model alone).
"""
import ast
from dataclasses import dataclass
import hashlib
import json
import keyword
import difflib
import math
import os
import posixpath
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .refusals import RunnerStatus, replace_result_text, runner_status

MAX_TEST_RETRIES = 2

_TEST_CLAIM = re.compile(
    r"\b(?:"
    r"(?:i\s+)?(?:ran|run|executed|conducted)\s+(?:a\s+|the\s+)?(?:tests?|test\s+suite|pytest)\b|"
    r"(?:ran|run|executed)\s+(?:a\s+|the\s+)?tests?\s+to\s+(?:confirm|verify|check|ensure)\b|"
    r"(?:tests?|test\s+suite|pytest|run_tests)\s+(?:(?:all|now|both)\s+)?(?:(?:has|have)\s+)?(?:pass(?:es|ed|ing)?|succeed(?:s|ed)?|"
    r"(?:is|are)\s+(?:green|passing)|(?:is|are)\s+now\s+passing)\b|"
    r"\ball\s+(?:tests?\s+)?(?:pass(?:es|ed)?|green)\b|"
    r"\b(?:all\s+)?assertions?\s+(?:(?:has|have)\s+)?pass(?:es|ed)?\b|"
    r"\b(?:the\s+)?tests?\s+completed\s+successfully\b|"
    r"(?:tests?|test\s+suite|pytest)\s+(?:(?:has|have)\s+)?(?:runs?|ran)\s+"
    r"(?:correctly|properly|successfully|without\s+errors?)\b|"
    r"(?:everything|all)\s+passes\b|"
    r"tested\s+(?:the\s+)?(?:changes?|fix|code|file|implementation)\b|"
    r"tested\s+and\s+(?:confirmed|verified)\b|"
    r"(?:confirmed|verified)\s+(?:the\s+)?(?:fix|changes?|code)\s+(?:by\s+running|with|via)\s+(?:a\s+|the\s+)?tests?"
    r")\b",
    re.IGNORECASE,
)
_INLINE_TEST_SCOPE = re.compile(r"\b(?:inline|detached|snippets?|run_code|scratchpad|assertions?)\b", re.IGNORECASE)
_SNIPPET_SUCCESS_CLAIM = re.compile(
    r"\b(?:(?P<quantifier>all|both|the|a)\s+)?(?:code\s+)?"
    r"(?P<subject>snippets?|run_code|(?:inline|detached)\s+(?:code|scripts?))\s+"
    r"(?:(?:has|have)\s+)?(?:completed|finished|succeeded|"
    r"(?:executed|ran)\s+(?:successfully|without\s+errors?))\b",
    re.IGNORECASE,
)
_FILE_VERIFY_CLAIM = re.compile(
    r"\b(?:"
    r"(?:i\s+)?(?:checked|verified|inspected|examined)\s+(?:the\s+)?(?:file|changes?|diff|modifications?|patch)\b|"
    r"(?:i\s+)?(?:checked|verified|inspected|examined)\s+[\w.-]+\s+and\b|"
    r"(?:checked|verified|inspected)\s+(?:the\s+)?(?:file|changes?)\s+and\s+(?:the\s+)?fix\s+works\b|"
    r"(?:the\s+)?fix\s+works\s+now\b|"
    r"verified\s+(?:that\s+)?(?:the\s+)?fix\s+works\b"
    r")\b",
    re.IGNORECASE,
)
_NEGATION_OR_PROPOSAL = re.compile(
    r"\b(?:"
    r"no|none|neither|not|never|cannot|can't|couldn't|didn't|haven't|hasn't|don't|doesn't|"
    r"aren't|isn't|wasn't|weren't|unable|unverified|unconfirmed|unrun|without|whether|if|"
    r"may|might|should|would|"
    r"i\s+can|we\s+can|you\s+can|can\s+(?:be\s+)?(?:run|verified|checked|tested|executed)|"
    r"i\s+will|i'll|we\s+will|we'll|please|let\s+me"
    r")\b",
    re.IGNORECASE,
)
_PATH_ARG_TOOLS = ("read_file", "patch_file", "write_file", "edit_function", "add_function")


_FAILURE_CLASSES = (
    ("sandbox-access-denied", r"Security Sandbox Access Denied:"),
    ("sensitive-file-access-denied", r"Security Blocked:"),
    ("unread-file", r"Read .+ with read_file before editing it,"),
    ("stale-file", r".+ changed on disk since you last read it\."),
    ("missing-file", r"File not found:|Target test path does not exist:"),
    ("missing-directory", r"Directory not found or not a directory:"),
    ("not-a-file", r"Path is a directory, not a file:"),
    ("not-a-directory", r".+ is a file, not a directory\."),
    ("empty-target", r"Target string to replace cannot be empty\."),
    ("target-mismatch", r"Target content not found in "),
    ("ambiguous-target", r"Target content found \d+ times in "),
    ("syntax-validation", r"Python syntax validation failed|Code has a syntax error:"),
    ("removal-not-approved", r"This (?:overwrite of|patch to) "),
    ("permission-denied", r"Permission denied"),
    ("execution-not-approved", r"Project test execution is not approved\."),
    ("execution-disabled", r"Executing existing files "),
    ("commit-disabled", r"Git commit is disabled "),
    ("invalid-arguments", r"Invalid arguments for tool "),
    ("unknown-tool", r"Unknown tool "),
    ("read-failed", r"Could not read file:"),
    ("write-failed", r"Could not write file:"),
    ("patch-failed", r"Could not patch file:"),
    ("directory-list-failed", r"Could not list directory:"),
    ("execution-timeout", r"Code execution timed out|Pytest execution timed out"),
    ("execution-failed", r"Code execution failed:|Execution failed for tool "),
    ("unsupported-language", r"Running language "),
    ("pytest-unavailable", r"pytest is not installed "),
    ("invalid-test-arguments", r"Invalid or unsafe -k |Invalid --maxfail |Unsupported or unsafe pytest argument:"),
    ("pytest-failed", r"Failed to run pytest:"),
    ("git-status-failed", r"Could not get git status"),
    ("git-diff-failed", r"Could not read git diff"),
    ("git-commit-failed", r"git commit failed:|Could not commit:"),
    ("empty-commit-message", r"Commit message cannot be empty\."),
    ("no-staged-changes", r"No staged changes found to commit\."),
)


class IdenticalFailureTracker:
    """Bounded, turn-local evidence of repeated failures on verified unchanged state.

    Observe actual tool results before guard wrappers add hints. File states are
    supplied by the caller; this tracker never reads or resolves a target path.
    Only SHA256 identities and fixed error labels are retained, not tool payloads.
    """

    def __init__(self, max_entries: int = 128):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._failures = {}
        self.last_error_class: Optional[str] = None

    def begin_turn(self) -> None:
        self._failures.clear()
        self.last_error_class = None

    @staticmethod
    def _error_identity(result: str) -> Tuple[str, str]:
        # Inspect a bounded header only, never supplemental file DATA or hints.
        prefix = result[6:518].lstrip(" \t").split("\n", 1)[0].split("\r", 1)[0]
        prefix = re.split(
            r"\b(?:DATA|HINTS?):", prefix, maxsplit=1, flags=re.IGNORECASE,
        )[0].rstrip()
        for label, pattern in _FAILURE_CLASSES:
            if re.match(pattern, prefix):
                return label, label
        search = re.match(r"search_web \[(empty|irrelevant|network|unavailable|blocked|timeout|error)\]:", prefix)
        if search:
            label = "search-" + search.group(1)
            return label, label
        # Unrecognized errors remain distinct without exposing their text. The
        # bounded digest also prevents arbitrary error payloads growing state.
        return "unclassified-tool-error", hashlib.sha256(prefix.encode("utf-8")).hexdigest()

    @staticmethod
    def _verified_state(state) -> bool:
        if not isinstance(state, tuple) or len(state) != 2:
            return False
        kind, digest = state
        if kind in ("missing", "no-target"):
            return digest is None
        return (
            kind == "file" and (
                (isinstance(digest, bytes) and len(digest) == 32)
                or (
                    isinstance(digest, str)
                    and re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None
                )
            )
        )

    def observe(self, name: str, args: dict, result: str, before_state, after_state) -> bool:
        """Return true on the second identical ERROR with unchanged known state.

        Different failures may intervene. Any successful actual call or explicit
        turn boundary clears all evidence. Unknown/changed state invalidates old
        evidence for these arguments rather than assuming a file stayed unchanged.
        """
        if not result.startswith("ERROR:"):
            self.begin_turn()
            return False
        self.last_error_class, error_identity = self._error_identity(result)
        normalized = args
        if name == "patch_file" and "allow_multiple" not in args:
            normalized = dict(args, allow_multiple=False)
        try:
            canonical = json.dumps(
                normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            # Non-JSON arguments cannot establish a canonical identity.
            self._failures.clear()
            return False
        args_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if (
            not self._verified_state(before_state)
            or not self._verified_state(after_state)
            or before_state != after_state
        ):
            self._failures = {
                key: value for key, value in self._failures.items()
                if key[:2] != (name, args_hash)
            }
            return False
        key = (name, args_hash, self.last_error_class, error_identity, before_state)
        if key in self._failures:
            return True
        if len(self._failures) >= self.max_entries:
            del self._failures[next(iter(self._failures))]
        self._failures[key] = None
        return False


def _clean_for_claims(text: str) -> str:
    text = re.sub(r"```.*?```|~~~.*?~~~|`[^`]*`", "", text or "", flags=re.DOTALL)
    text = re.sub(r"\"[^\"\n]*\"|“[^”\n]*”|(?<!\w)'[^'\n]*'(?!\w)", "", text)
    return text


def _test_claim_clauses(text: str):
    # Quoted contractions are data too. Normalize only this disposable matching
    # view; file-verification grammar and canonical model bytes stay unchanged.
    quoted = (text or "").replace("\u2019", "'").replace("\u2018", "'")
    quoted = re.sub(r"(?<!\w)'(?:[^'\n]|(?<=\w)'(?=\w))*'(?!\w)", "", quoted)
    for clause in re.split(r"[\n.!?;]+", _clean_for_claims(quoted)):
        if re.match(r"\s*(?:for\s+example|example|sample\s+(?:reply|answer))\s*[:,]", clause, re.IGNORECASE):
            continue
        if not clause.lstrip().startswith(">") and not _NEGATION_OR_PROPOSAL.search(clause):
            yield clause


def claims_tests_pass(text: str) -> bool:
    """Recognize bounded affirmative test claims, never execution evidence."""
    return any(_TEST_CLAIM.search(clause) for clause in _test_claim_clauses(text))


def claims_file_verification(text: str) -> bool:
    """Recognize file-verification assertions, not proposals or negative admissions."""
    clean = _clean_for_claims(text)
    for clause in re.split(r"[\n.!?;]+", clean):
        if clause.lstrip().startswith(">"):
            continue
        if _FILE_VERIFY_CLAIM.search(clause) and not _NEGATION_OR_PROPOSAL.search(clause):
            return True
    return False


_TOOL_ACTION = r"(?:run_code|search_web|read_file|run_tests|patch_file|write_file|edit_function|add_function|list_directory|git_status|git_diff)"
_ACTION_PROMISE = re.compile(
    r"\bi(?:'ll|\s+will|\s+am\s+going\s+to)\s+(?:now\s+)?"
    r"(?:verify|run(?!\s+through\b)|execute|search|check|read|patch|edit|test|"
    r"use\s+(?:the\s+)?" + _TOOL_ACTION + r"|" + _TOOL_ACTION + r")\b|"
    r"\b" + _TOOL_ACTION + r"\s+"
    r"(?:initiated|started|running)\b",
    re.IGNORECASE,
)


_FILE_NAME_RE = re.compile(
    r"[\w./\\-]+\.(?:py|js|ts|java|c|cpp|h|go|rs|rb|html|css|json|ya?ml|toml|md|txt|csv|xml|sql|sh)",
    re.IGNORECASE,
)
# Honest admission that no read happened; such answers are not claims.
_READ_ADMISSION = re.compile(
    r"\b(?:could\s+not|couldn't|cannot|can't|did\s+not|didn't|unable\s+to|failed\s+to|"
    r"haven't|have\s+not|not\s+able\s+to)\s+(?:read|open|access|review|inspect|examine)\b",
    re.IGNORECASE,
)
# Weak content signals: require a fenced block nearby to count as a contents claim.
_CONTENT_CLAIM = re.compile(
    r"\bi(?:'ll|\s+will|\s+have|'ve)?\s*(?:now\s+|first\s+|just\s+)*"
    r"(?:read|reviewed|checked|inspected|examined|looked\s+(?:at|through|over))\b|"
    r"\bthe\s+(?:file|code|contents?)\s+(?:contains?|shows?|is|are|has|have|looks?)\b|"
    r"\bno\s+(?:obvious\s+)?bugs?\b|\bbugs?\s+(?:found|identified)\b|"
    r"\bcurrent\s+(?:contents?|structure|implementation|version)\b|"
    r"\bincorrect\s+behavior\b|\bnothing\s+wrong\b",
    re.IGNORECASE,
)
# Strong claim: a first-person statement that the read/review was PERFORMED.
# Unambiguous enough to flag without a fenced block.
_PERFORMED_CLAIM = re.compile(
    r"\bi\s+(?:read|reviewed|checked|inspected|examined|looked\s+(?:at|through|over))\b|"
    r"\bi(?:'ve|\s+have)\s+(?:now\s+|just\s+|already\s+)*"
    r"(?:read|reviewed|checked|inspected|examined|looked\s+(?:at|through|over))\b",
    re.IGNORECASE,
)
# Proposed-new-file / advice phrasing: the file is being drafted, not reviewed.
_PROPOSAL_SIGNAL = re.compile(
    r"\b(?:create|creates|creating|write|writing|save|generates?|generating|new\s+file|"
    r"you\s+(?:can|could|should|might))\b|"
    r"\bhere(?:'s|\s+is)\s+(?:a|an)\s+(?:example|template|sample|draft)\b",
    re.IGNORECASE,
)
def tool_result_succeeded(result):
    """True when a tool result is an actual success, by status metadata first.

    A successful read of a file whose bytes begin with "ERROR" must count as a
    read; only a bare string without status metadata falls back to the prefix.
    """
    if getattr(result, "refusal_class", None) is not None:
        return False
    executed = getattr(result, "executed", None)
    if executed is not None:
        return bool(executed)
    return not str(result).lstrip().startswith("ERROR")


def _norm_path(value):
    # Platform-appropriate identity: os.path.normcase lowercases only where the
    # filesystem itself is case-insensitive, so case-distinct names never merge
    # on case-sensitive platforms. normcase also converts "/" to "\\" on Windows,
    # so separators are normalized AGAIN after it: one representation everywhere.
    normed = posixpath.normpath(str(value).replace("\\", "/").strip())
    return os.path.normcase(normed).replace("\\", "/").rstrip("/")


def _basename(path):
    return path.rsplit("/", 1)[-1]


def _file_identity(path):
    """Content fingerprint in the executor's own _file_state representation.

    Equal bytes hash equal on every supported Python, so an identity captured
    by the tool loop (hashlib.file_digest) matches one recomputed here.
    """
    try:
        return ("file", hashlib.sha256(Path(path).read_bytes()).digest())
    except FileNotFoundError:
        return ("missing", None)
    except OSError:
        return None


def known_file_entry(path, *, base_dir=None, bound_state=None):
    """Provenance record for one successfully read/written file.

    ref: normalized path as recorded (for claim matching). abs: the resolved
    on-disk target; relative paths resolve against base_dir, which the
    registration site anchors to the same root the tools resolve against
    (_validate_workspace_path: the process cwd at execution time), and resolve()
    mirrors the tool layer. bound_state: the executor's own content identity
    ("file", sha256 digest) captured inside its before/after observation window
    - for a read, the identities must be equal across the read, otherwise the
    observation is unbound (None); for a write, the post-write identity.
    bound_state=None means historical trusted provenance with NO freshness
    claim: such records can never suppress a current-disk warning.
    """
    ref = _norm_path(path)
    raw = str(path)
    target = Path(raw)
    if not target.is_absolute():
        target = (Path(base_dir) if base_dir else Path.cwd()) / raw
    target = target.resolve()  # mirror the tool layer's _validate_workspace_path
    return {"ref": ref, "abs": str(target), "state": bound_state}


def _entry_fresh(entry):
    """True: the live file's content identity equals the bound observation.
    False: changed, missing, or unreadable. None: no bound observation on
    record, so freshness is unverifiable - this NEVER suppresses a warning."""
    state = entry.get("state")
    if not (isinstance(state, tuple) and len(state) == 2 and state[0] == "file"):
        return None
    current = _file_identity(entry["abs"])
    return current == state


def _provenance_entry(ref_raw, entries, workspace_root=None):
    """The entry identifying the same file as ref_raw, else None.

    Identity is exact normalized equality, or canonical equality with the
    workspace-resolved target (so a qualified relative reference matches an
    absolute executor record of the same file). Reading calculator.py says
    nothing about app/calculator.py. A bare name matches a known file's
    basename only when exactly one known file has it.
    """
    ref = _norm_path(ref_raw)
    candidates = {ref}
    if workspace_root and "/" in ref and not posixpath.isabs(ref) and ":" not in ref.split("/", 1)[0]:
        try:
            candidates.add(_norm_path(str((Path(workspace_root) / str(ref_raw)).resolve())))
        except (OSError, ValueError):
            pass
    for entry in entries:
        if entry["ref"] in candidates:
            return entry
        abs_norm = _norm_path(entry["abs"]) if entry.get("abs") else None
        if abs_norm and abs_norm in candidates:
            return entry
    if "/" in ref:
        return None
    basename_candidates = [e for e in entries if _basename(e["ref"]) == ref]
    return basename_candidates[0] if len(basename_candidates) == 1 else None


_FENCE_SPAN_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_SPAN_RE = re.compile(r"`[^`\n]*`")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _segment_spans(text, fence_spans):
    """Sentence-like segments of the non-fence text, original offsets kept.

    Fenced source blocks are hard boundaries: quoted or source data never
    exempts a claim and never counts as one. Inline-code spans are masked from
    signal matching but keep their enclosing sentence, so an inline-formatted
    filename (``I'll create `config.py` for you:``) still gets its exemptions.
    """
    segments = []
    pos = 0
    for cs, ce in sorted(fence_spans) + [(len(text), len(text))]:
        base = pos
        for m in _SENTENCE_SPLIT.finditer(text[pos:cs]):
            if base < pos + m.start():
                segments.append((base, pos + m.start()))
            base = pos + m.end()
        if base < cs:
            segments.append((base, cs))
        pos = ce
    return segments


class FileClaimReport(tuple):
    """(unverified, stale, unverifiable): files with unsupported claims.

    unverified: no successful disk read/write of the file is on record in the
    current context. stale: a bound read/write is on record but the file on
    disk has changed or become unavailable since. unverifiable: a historical
    trusted record exists but carries no bound observation (or its source left
    the context), so freshness could not be verified - it is NOT a claim that
    the file changed.
    """
    __slots__ = ()

    def __new__(cls, unverified, stale, unverifiable):
        return tuple.__new__(cls, (list(unverified), list(stale), list(unverifiable)))

    @property
    def unverified(self):
        return self[0]

    @property
    def stale(self):
        return self[1]

    @property
    def unverifiable(self):
        return self[2]


def unverified_file_content_claims(reply, *, user_input="", known_files=(), workspace_root=None):
    """Heuristic (not general protection): files a reply presents contents or a
    performed review for without a verified disk source.

    Provenance is per-file and comes ONLY from trusted records: successful
    read/write tool calls and real @file attachments, as known_file_entry
    records. Request text grants nothing: a pasted fence, the words "content
    of", or user-typed attachment markers are not disk reads. Only positively
    established freshness (a bound observation whose target still matches the
    disk) suppresses a current-disk warning; historical records without a bound
    observation are classified separately as unverifiable.

    A claim needs a performed-review or contents signal in the +-300-char
    window with code spans masked out. Exemptions (honest no-read admissions,
    proposed-new-file/advice phrasing) apply only within the claim's own
    sentence segment and never override a performed-review claim in that same
    segment, so advice cannot cancel an independent fabricated review.
    user_input is accepted for call-site compatibility and grants no provenance.
    """
    if not reply:
        return FileClaimReport([], [], [])
    fence_spans = [(m.start(), m.end()) for m in _FENCE_SPAN_RE.finditer(reply)]
    mask_spans = fence_spans + [(m.start(), m.end()) for m in _INLINE_SPAN_RE.finditer(reply)]
    masked = list(reply)
    for cs, ce in mask_spans:
        for i in range(cs, ce):
            masked[i] = " "
    masked = "".join(masked)
    segments = _segment_spans(reply, fence_spans)
    has_fence = bool(fence_spans)

    entries = []
    for item in known_files or ():
        if isinstance(item, dict):
            entry = dict(item)
            entry["ref"] = _norm_path(entry.get("ref") or entry.get("abs") or "")
            entries.append(entry)
        else:
            entries.append({"ref": _norm_path(item), "abs": None, "mtime_ns": None, "size": None})

    def _segment_text(pos):
        for cs, ce in fence_spans:
            if cs <= pos < ce:
                return ""
        for ss, se in segments:
            if ss <= pos < se:
                return masked[ss:se]
        return ""

    unverified, stale, unverifiable, seen = [], [], [], set()
    for match in _FILE_NAME_RE.finditer(reply):
        ref = _norm_path(match.group(0))
        if not ref or ref in seen:
            continue
        window = masked[max(0, match.start() - 300):match.end() + 300]
        if not (_PERFORMED_CLAIM.search(window) or (has_fence and _CONTENT_CLAIM.search(window))):
            continue
        segment = _segment_text(match.start())
        if (not _PERFORMED_CLAIM.search(segment)
                and (_READ_ADMISSION.search(segment) or _PROPOSAL_SIGNAL.search(segment))):
            continue
        seen.add(ref)
        entry = _provenance_entry(match.group(0), entries, workspace_root=workspace_root)
        fresh = _entry_fresh(entry) if entry is not None else None
        if entry is None:
            unverified.append(ref)
        elif fresh is False:
            stale.append(ref)
        elif fresh is None:
            unverifiable.append(ref)
        # Only positively established freshness (fresh is True) stays silent.
        if len(unverified) + len(stale) + len(unverifiable) >= 3:
            break
    return FileClaimReport(unverified, stale, unverifiable)


def promises_tool_use(text: str) -> bool:
    """Recognize unfulfilled action announcements, not examples or admissions."""
    text = re.sub(r"```.*?```|`[^`]*`", "", text or "", flags=re.DOTALL)
    text = re.sub(r"\"[^\"\n]*\"|“[^”\n]*”|(?<!\w)'[^'\n]*'(?!\w)", "", text)
    for clause in re.split(r"[\n.!?;]+", text):
        # Negation must apply to the action itself, not an unrelated constraint
        # such as "I'll verify without changing files".
        if not clause.lstrip().startswith(">") and _ACTION_PROMISE.search(clause):
            return True
    return False


def function_request_prose(request: str) -> str:
    """Exclude examples/quotations before extracting bounded edit requirements."""
    # Bare retries may route through a stored prompt that already has attachments.
    # Only its request prefix is eligible; attached source is never an instruction.
    request = (request or "").split("\n\n[Context attached from @", 1)[0]
    text = re.sub(r"```.*?```|~~~.*?~~~|`[^`]*`", "", request, flags=re.DOTALL)
    text = re.sub(r"\"[^\"\n]*\"|“[^”\n]*”|(?<!\w)'[^'\n]*'(?!\w)", "", text)
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))


_PYTHON_NAME = r"[A-Za-z_][A-Za-z_0-9]*"
_FUNCTION_ACTION = re.compile(
    r"(?:^\s*|[,;.!]\s*|\b(?:then|and)\s+)(?:please\s+)?"
    r"(?:(?:can|could|would)\s+you\s+(?:please\s+)?)?(?:"
    r"rename\s+(?:the\s+)?(?:" + _PYTHON_NAME + r"\s+function|function\s+" + _PYTHON_NAME
    + r")\s+to\s+(?P<rename>" + _PYTHON_NAME + r")\b|"
    r"(?:add|create|define)\s+(?:(?:a|the)\s+)?(?:"
    r"(?P<add>" + _PYTHON_NAME + r")\s*\([^()\n]*\)(?:\s+function\b|"
    r"(?=\s*(?:[,;.!?]|\b(?:to|in|that|which|and|then|with)\b|$)))|"
    r"(?P<named>" + _PYTHON_NAME + r")\s+function\b|"
    r"function\s+(?P<define>" + _PYTHON_NAME + r")\b"
    r"(?=\s*(?:\(|[,;.!?]|\b(?:to|in|that|which|and|then|with)\b|$))))",
    re.IGNORECASE,
)
_FUNCTION_NOUN_FILLERS = frozenset({
    "a", "an", "the", "function", "to", "in", "for", "with", "and", "or",
    "that", "which", "please", "named", "called", "new",
})


def requested_function_names(prose: str) -> List[str]:
    """Recognize bounded direct rename/add/create/define requests, not semantics."""
    intent = re.sub(r"^\s*(?:in|for)\s+[^,;\n]+,\s*", "", prose, flags=re.IGNORECASE)
    if re.match(
        r"^\s*(?:please\s+)?(?:how\b|what\b|why\b|when\b|where\b|explain\b|describe\b|"
        r"(?:can|could|would)\s+you\s+(?:please\s+)?(?:explain|describe)\b)",
        intent, re.IGNORECASE,
    ):
        return []
    names = []
    for match in _FUNCTION_ACTION.finditer(prose):
        name = (
            match.group("rename") or match.group("add")
            or match.group("named") or match.group("define")
        )
        # Articles/linking words are not missing names. Parentheses and rename
        # targets make identifiers explicit; keep their existing name semantics.
        if not (match.group("rename") or match.group("add")) and name.casefold() in _FUNCTION_NOUN_FILLERS:
            continue
        if name.isidentifier() and not keyword.iskeyword(name) and name not in names:
            names.append(name)
    return names


def check_function_requirements(requirements: Dict[str, List[str]]) -> List[str]:
    """Check actual workspace source without executing it or trusting tool claims."""
    from .tools import execute_read_file

    incomplete = []
    for path, names in requirements.items():
        source = execute_read_file(path)
        if source.startswith("ERROR:"):
            incomplete.append(
                f"{path}: could not check requested top-level Python functions "
                f"({', '.join(names)}): {source}"
            )
            continue
        try:
            module = ast.parse(source, filename=path)
        except (SyntaxError, ValueError, RecursionError) as exc:
            incomplete.append(
                f"{path}: could not check requested top-level Python functions "
                f"({', '.join(names)}): source could not be parsed ({type(exc).__name__})."
            )
            continue
        present = {
            node.name for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = [name for name in names if name not in present]
        if missing:
            incomplete.append(f"{path}: missing top-level Python function(s): {', '.join(missing)}.")
    return incomplete


@dataclass(frozen=True)
class LiteralAssignmentRequirement:
    """A single explicit request, not a claim about runtime program behavior."""

    path: str
    name: str
    literal: str
    value: object


_LITERAL_ASSIGNMENT = re.compile(
    r"(?:please[ \t]+)?(?:change|set)[ \t]+(?P<name>" + _PYTHON_NAME + r")"
    r"[ \t]+to[ \t]+(?P<literal>[^\r\n]+?)[ \t]+in[ \t]+"
    r"(?P<path>[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.py)[.!]?",
    re.IGNORECASE,
)


def _literal_value(node: ast.AST) -> object:
    """Accept only unambiguous scalars, lists, tuples, and dictionaries."""
    if isinstance(node, ast.Constant):
        value = node.value
        if type(value) not in (type(None), bool, int, float, str, bytes):
            raise ValueError("unsupported literal")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite literal")
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        if not isinstance(node.operand, ast.Constant) or type(node.operand.value) not in (int, float):
            raise ValueError("unsupported signed literal")
        value = ast.literal_eval(node)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite literal")
        return value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_literal_value(item) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        result = {}
        for key, item in zip(node.keys, node.values):
            if key is None:
                raise ValueError("dictionary unpacking")
            key_value = _literal_value(key)
            if key_value in result:
                raise ValueError("ambiguous duplicate dictionary key")
            result[key_value] = _literal_value(item)
        return result
    raise ValueError("unsupported literal")


def _literal_identity(value: object) -> tuple:
    """Compare nested literal types too: Python's True == 1 is not sufficient."""
    if isinstance(value, dict):
        contents = frozenset(
            (_literal_identity(key), _literal_identity(item)) for key, item in value.items()
        )
    elif isinstance(value, (list, tuple)):
        contents = tuple(_literal_identity(item) for item in value)
    else:
        contents = value
    return type(value), contents


def requested_literal_assignment(request: str) -> Optional[LiteralAssignmentRequirement]:
    """Recognize only a complete, unquoted, single-file literal edit instruction."""
    # Bare retries route through the stored request, which may include auto-attached source.
    request = (request or "").split("\n\n[Context attached from @", 1)[0]
    match = _LITERAL_ASSIGNMENT.fullmatch((request or "").strip())
    if match is None:
        return None
    name, path, literal = match.group("name", "path", "literal")
    if keyword.iskeyword(name) or ".." in path.split("/"):
        return None
    try:
        expression = ast.parse(literal, mode="eval")
        value = _literal_value(expression.body)
    except (SyntaxError, ValueError, TypeError, RecursionError, OverflowError):
        return None
    return LiteralAssignmentRequirement(path, name, literal, value)


class _BindingWrites(ast.NodeVisitor):
    """Find syntactic module writes, excluding function and class-local bindings."""

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self.found = False

    def visit_Name(self, node: ast.Name) -> None:
        if self.enabled and node.id == self.name and isinstance(node.ctx, (ast.Store, ast.Del)):
            self.found = True

    def _visit_mutation(self, node: ast.AST) -> None:
        root = node
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if (self.enabled and isinstance(node.ctx, (ast.Store, ast.Del))
                and isinstance(root, ast.Name) and root.id == self.name):
            self.found = True
        self.generic_visit(node)

    visit_Attribute = _visit_mutation
    visit_Subscript = _visit_mutation

    def visit_Import(self, node: ast.Import) -> None:
        if self.enabled and any(
            (alias.asname or alias.name.split(".")[0]) == self.name for alias in node.names
        ):
            self.found = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.enabled and any(
            alias.name == "*" or (alias.asname or alias.name) == self.name for alias in node.names
        ):
            self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.enabled and node.name == self.name:
            self.found = True
        # Defaults, decorators, and annotations run outside the function body.
        for field, value in ast.iter_fields(node):
            if field != "body":
                self._visit_field(value)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit(node.args)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.enabled and node.name == self.name:
            self.found = True
        for field, value in ast.iter_fields(node):
            if field != "body":
                self._visit_field(value)
        # A class body executes, but only explicit global bindings escape its scope.
        globals_visitor = _ScopeGlobals()
        for statement in node.body:
            globals_visitor.visit(statement)
        body_writes = _BindingWrites(self.name, self.name in globals_visitor.names)
        for statement in node.body:
            body_writes.visit(statement)
        self.found = self.found or body_writes.found

    def _visit_field(self, value: object) -> None:
        if isinstance(value, ast.AST):
            self.visit(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    self.visit(item)

    def _visit_comprehension(self, node: ast.AST) -> None:
        # Iteration targets are comprehension-local; walrus expressions are not.
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for field in ("elt", "key", "value"):
            value = getattr(node, field, None)
            if value is not None:
                self.visit(value)

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if self.enabled and node.name == self.name:
            self.found = True
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if self.enabled and node.name == self.name:
            self.found = True
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if self.enabled and node.rest == self.name:
            self.found = True
        self.generic_visit(node)


class _ScopeGlobals(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Global(self, node: ast.Global) -> None:
        self.names.update(node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef


def check_literal_assignment_requirement(
    requirement: Optional[LiteralAssignmentRequirement],
) -> List[str]:
    """Check the last supported syntactic module binding without executing source."""
    if requirement is None:
        return []
    from .tools import execute_read_file

    path, name = requirement.path, requirement.name
    source = execute_read_file(path)
    if source.startswith("ERROR:"):
        return [f"{path}: requested assignment for {name} is unverified: {source}"]
    try:
        module = ast.parse(source, filename=path)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return [
            f"{path}: requested assignment for {name} is unverified: "
            f"source could not be parsed ({type(exc).__name__})."
        ]

    state, final_value = "missing", None
    for statement in module.body:
        value = None
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == name):
            value = statement.value
        elif (isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name) and statement.target.id == name):
            annotation_writes = _BindingWrites(name)
            annotation_writes.visit(statement.annotation)
            if not annotation_writes.found:
                value = statement.value
        if value is not None:
            try:
                final_value = _literal_value(value)
                state = "literal"
                continue
            except (ValueError, TypeError, RecursionError, OverflowError):
                pass
        writes = _BindingWrites(name)
        writes.visit(statement)
        if writes.found:
            state = "unverified"

    if state == "missing":
        return [f"{path}: missing module-level assignment for {name} to {requirement.literal}."]
    if state == "unverified":
        return [
            f"{path}: final module-level binding of {name} is unverified "
            "(unsupported or conditional write)."
        ]
    if _literal_identity(final_value) != _literal_identity(requirement.value):
        return [
            f"{path}: final module-level assignment for {name} does not match "
            f"requested literal {requirement.literal}."
        ]
    return []



def guardrails_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("NEXUS_GUARDRAILS", "1").strip().lower() not in ("0", "false", "no", "off")


def _close_names(missing: str, names: List[str]) -> List[str]:
    stem = Path(missing).stem.lower()
    ext = Path(missing).suffix.lower()
    scored = []
    for n in names:
        nstem, next_ = Path(n).stem.lower(), Path(n).suffix.lower()
        ratio = difflib.SequenceMatcher(None, Path(missing).name.lower(), n.lower()).ratio()
        shares = bool(stem) and len(stem) >= 3 and (nstem.startswith(stem[:4]) or stem.startswith(nstem[:4]))
        if ratio >= 0.6 or (shares and (not ext or ext == next_)):
            scored.append((ratio, n))
    return [n for _, n in sorted(scored, reverse=True)[:3]]


class LoopGuard:
    """Per-run state. Create one per Agent.run()."""

    def __init__(self, list_directory) -> None:
        self._list_directory = list_directory
        self.last_tests: Optional[str] = None  # Live RunnerStatus value; None before any test attempt.
        self.last_tests_output = ""
        self.claim_unverified = False
        self.claim_unverified_error = ""
        self.edited_since_tests = False
        self.last_code_status = RunnerStatus.UNKNOWN
        self.code_calls = 0
        self.all_code_completed = True
        self.edited_paths: set[str] = set()
        self.post_edit_read_paths: set[str] = set()
        self.read_paths: set[str] = set()
        self.git_outputs: List[str] = []
        self.test_retries = 0
        self.notes: List[str] = []

    # -- after every tool call -------------------------------------------------
    def after_tool(self, name: str, args: Dict, result: str) -> str:
        text = str(result)
        if name == "run_tests":
            self.last_tests = runner_status(result, "run_tests").value
            self.last_tests_output = text
            self.edited_since_tests = False
        elif name == "run_code":
            self.last_code_status = runner_status(result, "run_code")
            self.code_calls += 1
            self.all_code_completed &= self.last_code_status is RunnerStatus.COMPLETED
        elif name in ("patch_file", "write_file", "edit_function", "add_function") and not text.startswith(("ERROR:", "No changes:")):
            self.edited_since_tests = True
            raw_path = str(args.get("path", "")).strip()
            if raw_path:
                norm_p = os.path.normcase(os.path.normpath(raw_path))
                self.edited_paths.add(norm_p)
                self.post_edit_read_paths.discard(norm_p)
        elif name == "read_file" and tool_result_succeeded(result):
            raw_path = str(args.get("path", "")).strip()
            if raw_path:
                norm_p = os.path.normcase(os.path.normpath(raw_path))
                self.read_paths.add(norm_p)
                if norm_p in self.edited_paths:
                    self.post_edit_read_paths.add(norm_p)
        elif name in ("git_status", "git_diff") and not text.startswith("ERROR:"):
            self.git_outputs.append(text)

        if name in _PATH_ARG_TOOLS and text.startswith("ERROR: File not found"):
            text += self._missing_file_help(str(args.get("path", "")))
        return replace_result_text(result, text)

    def _missing_file_help(self, path: str) -> str:
        parent = str(Path(path).parent) if path else "."
        listing = self._list_directory(parent if parent else ".")
        if listing.startswith("ERROR:"):
            listing = self._list_directory(".")
            parent = "."
        names = re.findall(r"\[FILE\] (.+)", listing)
        close = _close_names(path, names)
        self.notes.append(f"missing-file listing for {path}")
        out = f"\n\n[SYSTEM] '{path}' does not exist. Real contents of '{parent}':\n{listing}"
        if close:
            out += "\nClosest existing file(s): " + ", ".join(close) + ". Read the right one before continuing."
        else:
            out += "\nNo similar file exists. Use only names from this listing; if the file is not here, say it does not exist."
        return out

    # -- before the final answer is accepted -----------------------------------
    def review_final(self, text: str) -> Tuple[str, str]:
        """Return (verdict, text). verdict: 'ok' | 'retry' (text is the nudge to send) ."""
        self.claim_unverified = False
        self.claim_unverified_error = ""
        clauses = list(_test_claim_clauses(text))
        test_clauses = [clause for clause in clauses if _TEST_CLAIM.search(clause)]
        claims_tests = bool(test_clauses)
        unsupported_scope = any(_INLINE_TEST_SCOPE.search(clause) for clause in test_clauses)
        snippet_unverified = False
        for clause in clauses:
            for match in _SNIPPET_SUCCESS_CLAIM.finditer(clause):
                plural = (match.group("subject").lower().endswith("s")
                          or (match.group("quantifier") or "").lower() in ("all", "both"))
                if (not self.code_calls or self.last_code_status is not RunnerStatus.COMPLETED
                        or (plural and not self.all_code_completed)):
                    snippet_unverified = True
        claims_file = claims_file_verification(text)

        if claims_tests and self.last_tests == "failed":
            self.claim_unverified = True
            self.claim_unverified_error = "The model claimed tests pass, but run_tests reported FAILED."
            if self.test_retries < MAX_TEST_RETRIES:
                self.test_retries += 1
                self.notes.append("false success claim blocked")
                return "retry", (
                    "[SYSTEM] Your last run_tests result was FAILED, so tests do not pass. "
                    "Fix the code with patch_file or write_file, run run_tests again, and only report success if it says PASSED. "
                    "Last result:\n" + self.last_tests_output[-800:]
                )
            return "ok", text + "\n\nCorrection: run_tests last reported FAILED; no fresh passing run_tests result was recorded.\n" + self.last_tests_output[-600:]

        extra = ""
        test_unverified = False
        if claims_tests:
            if unsupported_scope:
                test_unverified = True
                self.claim_unverified = True
                extra += (
                    "\n\nNote: the inline/assertion claim is unverified. Detached code completion "
                    "does not prove assertions were reached, and run_tests does not establish "
                    "a detached snippet's result or fix correctness."
                )
            elif self.last_tests == "passed" and not self.edited_since_tests:
                pass
            elif self.last_tests == "passed" and self.edited_since_tests:
                test_unverified = True
                self.claim_unverified = True
                extra += "\n\nNote: files changed after the last test run, so rerun tests to confirm."
            else:
                test_unverified = True
                self.claim_unverified = True
                extra += (
                    "\n\nNote: no fresh source-verified passing run_tests result backs this test "
                    "claim, so it is unverified. Detached code completion is not project-test "
                    "or reached-assertion proof."
                )
        if snippet_unverified:
            test_unverified = True
            self.claim_unverified = True
            extra += (
                "\n\nNote: the snippet-completion claim is unverified by the private runner record. "
                "A completed invocation is not inferred from output text; a latest result "
                "does not establish that all earlier snippets completed."
            )

        file_unverified = False
        if claims_file:
            unverified_paths = self.edited_paths - self.post_edit_read_paths
            if unverified_paths:
                file_unverified = True
                self.claim_unverified = True
                extra += "\n\nNote: the modified file was not inspected after being edited, so the claim that changes were verified is unverified."
            elif not self.edited_paths and not self.read_paths:
                file_unverified = True
                self.claim_unverified = True
                extra += "\n\nNote: no file was inspected, so the verification claim is unverified."

        if test_unverified and file_unverified:
            self.claim_unverified_error = "The model made unverified test and verification claims without backing tool evidence."
        elif test_unverified:
            self.claim_unverified_error = "The model made a test/assertion or snippet-completion claim without matching fresh source evidence."
        elif file_unverified:
            self.claim_unverified_error = "The model claimed the changes were verified, but the modified file was not inspected after being edited."
        if self.git_outputs and not self._mentions_git_facts(text):
            self.notes.append("git answer grounded")
            extra += "\n\nActual git output:\n" + self.git_outputs[-1][:1500]
        return "ok", (text.rstrip() + extra) if extra else text

    def _mentions_git_facts(self, text: str) -> bool:
        out = self.git_outputs[-1]
        names = set(re.findall(r"[\w./-]+\.\w{1,5}", out))
        if "Working tree clean" in out or "No changes to commit" in out:
            return bool(re.search(r"\b(clean|no\s+(?:uncommitted\s+)?changes|nothing)\b", text, re.IGNORECASE))
        low = text.lower()
        return any(n.lower() in low for n in names if not n.startswith("a/") and not n.startswith("b/"))
