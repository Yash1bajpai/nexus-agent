"""Loop guardrails: check the model's work against real tool results.

Small models hallucinate file names, claim tests pass when they failed, and
describe git output they never read. These checks run inside the agent loop,
check tool evidence and explicitly requested Python function names, and never change files.

Turn off with NEXUS_GUARDRAILS=0 (for measuring the model alone).
"""
import ast
import keyword
import difflib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MAX_TEST_RETRIES = 2

_TEST_CLAIM = re.compile(
    r"\b(?:"
    r"(?:i\s+)?(?:ran|run|executed|conducted)\s+(?:a\s+|the\s+)?(?:tests?|test\s+suite|pytest)\b|"
    r"(?:ran|run|executed)\s+(?:a\s+|the\s+)?tests?\s+to\s+(?:confirm|verify|check|ensure)\b|"
    r"(?:tests?|test\s+suite|pytest)\s+(?:(?:all|now|both)\s+)?(?:(?:has|have)\s+)?(?:pass(?:es|ed|ing)?|succeed(?:s|ed)?|"
    r"(?:is|are)\s+(?:green|passing)|(?:is|are)\s+now\s+passing)\b|"
    r"\ball\s+(?:tests?\s+)?(?:pass(?:es|ed)?|green)\b|"
    r"(?:everything|all)\s+passes\b|"
    r"tested\s+(?:the\s+)?(?:changes?|fix|code|file|implementation)\b|"
    r"tested\s+and\s+(?:confirmed|verified)\b|"
    r"(?:confirmed|verified)\s+(?:the\s+)?(?:fix|changes?|code)\s+(?:by\s+running|with|via)\s+(?:a\s+|the\s+)?tests?"
    r")\b",
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
_PATH_ARG_TOOLS = ("read_file", "patch_file", "write_file")


def _clean_for_claims(text: str) -> str:
    text = re.sub(r"```.*?```|~~~.*?~~~|`[^`]*`", "", text or "", flags=re.DOTALL)
    text = re.sub(r"\"[^\"\n]*\"|“[^”\n]*”|(?<!\w)'[^'\n]*'(?!\w)", "", text)
    return text


def claims_tests_pass(text: str) -> bool:
    """Recognize test assertions, not quoted examples, proposals, or negative admissions."""
    clean = _clean_for_claims(text)
    for clause in re.split(r"[\n.!?;]+", clean):
        if clause.lstrip().startswith(">"):
            continue
        if _TEST_CLAIM.search(clause) and not _NEGATION_OR_PROPOSAL.search(clause):
            return True
    return False


def claims_file_verification(text: str) -> bool:
    """Recognize file-verification assertions, not proposals or negative admissions."""
    clean = _clean_for_claims(text)
    for clause in re.split(r"[\n.!?;]+", clean):
        if clause.lstrip().startswith(">"):
            continue
        if _FILE_VERIFY_CLAIM.search(clause) and not _NEGATION_OR_PROPOSAL.search(clause):
            return True
    return False


_TOOL_ACTION = r"(?:run_code|search_web|read_file|run_tests|patch_file|write_file|list_directory|git_status|git_diff)"
_ACTION_PROMISE = re.compile(
    r"\bi(?:'ll|\s+will|\s+am\s+going\s+to)\s+(?:now\s+)?"
    r"(?:verify|run(?!\s+through\b)|execute|search|check|read|patch|edit|test|"
    r"use\s+(?:the\s+)?" + _TOOL_ACTION + r"|" + _TOOL_ACTION + r")\b|"
    r"\b" + _TOOL_ACTION + r"\s+"
    r"(?:initiated|started|running)\b",
    re.IGNORECASE,
)


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
    r"(?:^|[,;.!]\s*|\b(?:then|and)\s+)(?:please\s+)?(?:"
    r"rename\s+(?:the\s+)?(?:" + _PYTHON_NAME + r"\s+function|function\s+" + _PYTHON_NAME
    + r")\s+to\s+(?P<rename>" + _PYTHON_NAME + r")\b|"
    r"(?:add|create|define)\s+(?:(?:a|the)\s+)?(?:"
    r"(?P<add>" + _PYTHON_NAME + r")\s*\([^()\n]*\)\s+function\b|"
    r"function\s+(?P<define>" + _PYTHON_NAME + r")\b"
    r"(?=\s*(?:\(|[,;.!?]|\b(?:that|which|and|then|with)\b|$))))",
    re.IGNORECASE,
)


def requested_function_names(prose: str) -> List[str]:
    """Recognize explicit imperative rename/add/create/define forms, not semantics."""
    intent = re.sub(r"^\s*(?:in|for)\s+[^,;\n]+,\s*", "", prose, flags=re.IGNORECASE)
    if re.match(
        r"^\s*(?:please\s+)?(?:how\b|what\b|why\b|when\b|where\b|explain\b|describe\b|"
        r"(?:can|could|would)\s+you\s+(?:please\s+)?(?:explain|describe)\b)",
        intent, re.IGNORECASE,
    ):
        return []
    names = []
    for match in _FUNCTION_ACTION.finditer(prose):
        name = match.group("rename") or match.group("add") or match.group("define")
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
        self.last_tests: Optional[str] = None  # "passed" | "failed" | None
        self.last_tests_output = ""
        self.claim_unverified = False
        self.claim_unverified_error = ""
        self.edited_since_tests = False
        self.ran_code_success = False
        self.edited_since_code = False
        self.edited_paths: set[str] = set()
        self.post_edit_read_paths: set[str] = set()
        self.read_paths: set[str] = set()
        self.git_outputs: List[str] = []
        self.test_retries = 0
        self.notes: List[str] = []

    # -- after every tool call -------------------------------------------------
    def after_tool(self, name: str, args: Dict, result: str) -> str:
        text = str(result)
        if name == "run_tests" and not text.startswith("ERROR:"):
            self.last_tests = "passed" if "Pytest Results: PASSED" in text else "failed"
            self.last_tests_output = text
            self.edited_since_tests = False
        elif name == "run_tests":
            if "not approved" in text or "declined" in text or "pytest is not installed" in text:
                self.last_tests, self.last_tests_output = "unavailable", text  # could not run: not a test failure
            else:
                self.last_tests, self.last_tests_output = "failed", text
        elif name == "run_code" and not text.startswith("ERROR:"):
            self.ran_code_success = True
            self.edited_since_code = False
        elif name in ("patch_file", "write_file") and not text.startswith(("ERROR:", "No changes:")):
            self.edited_since_tests = True
            self.edited_since_code = True
            raw_path = str(args.get("path", "")).strip()
            if raw_path:
                norm_p = os.path.normcase(os.path.normpath(raw_path))
                self.edited_paths.add(norm_p)
                self.post_edit_read_paths.discard(norm_p)
        elif name == "read_file" and not text.startswith("ERROR:"):
            raw_path = str(args.get("path", "")).strip()
            if raw_path:
                norm_p = os.path.normcase(os.path.normpath(raw_path))
                self.read_paths.add(norm_p)
                if norm_p in self.edited_paths:
                    self.post_edit_read_paths.add(norm_p)
        elif name in ("git_status", "git_diff") and not text.startswith("ERROR:"):
            self.git_outputs.append(text)

        if name in _PATH_ARG_TOOLS and text.startswith("ERROR: File not found"):
            return text + self._missing_file_help(str(args.get("path", "")))
        return text

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
        claims_tests = claims_tests_pass(text)
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
            return "ok", text + "\n\nCorrection: run_tests last reported FAILED. The tests do not pass.\n" + self.last_tests_output[-600:]

        extra = ""
        test_unverified = False
        if claims_tests:
            if self.last_tests == "passed" and not self.edited_since_tests:
                pass
            elif self.ran_code_success and not self.edited_since_code:
                pass
            elif (self.last_tests == "passed" and self.edited_since_tests) or (self.ran_code_success and self.edited_since_code):
                test_unverified = True
                self.claim_unverified = True
                extra += "\n\nNote: files changed after the last test run, so rerun tests to confirm."
            else:
                test_unverified = True
                self.claim_unverified = True
                extra += (
                    "\n\nNote: the tests were not run, so the claim that they pass or confirm the fix is unverified."
                    if self.last_tests is None else
                    "\n\nNote: the tests could not be run (permission not given), so the claim that they pass is unverified."
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
            self.claim_unverified_error = "The model claimed the fix was tested, but no passing test run backs that up."
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
