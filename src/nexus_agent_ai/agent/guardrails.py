"""Loop guardrails: check the model's work against real tool results.

Small models hallucinate file names, claim tests pass when they failed, and
describe git output they never read. These checks run inside the agent loop,
use only data the tools already returned, and never change files.

Turn off with NEXUS_GUARDRAILS=0 (for measuring the model alone).
"""
import difflib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MAX_TEST_RETRIES = 2

_SUCCESS_CLAIM = re.compile(
    r"\b(?:tests?|test\s+suite|pytest)\s+(?:(?:all|now)\s+)?"
    r"(?:(?:has|have)\s+)?(?:pass(?:es|ed|ing)?|succeed(?:s|ed)?|"
    r"(?:is|are)\s+(?:green|passing)|(?:is|are)\s+now\s+passing)\b|"
    r"\ball\s+(?:tests?\s+)?(?:pass(?:es|ed)?|green)\b",
    re.IGNORECASE,
)
_NEGATED_CLAIM = re.compile(
    r"\b(?:no|none|neither|not|never|cannot|can't|couldn't|didn't|haven't|hasn't|don't|doesn't|"
    r"aren't|isn't|wasn't|weren't|unable|unverified|unconfirmed|unrun|without|whether|if|may|might|should|would)\b",
    re.IGNORECASE,
)
_PATH_ARG_TOOLS = ("read_file", "patch_file", "write_file")


def claims_tests_pass(text: str) -> bool:
    """Recognize assertions, not quoted examples, plans, or negative admissions."""
    text = re.sub(r"```.*?```|`[^`]*`", "", text or "", flags=re.DOTALL)
    text = re.sub(r"\"[^\"\n]*\"|“[^”\n]*”|(?<!\w)'[^'\n]*'(?!\w)", "", text)
    for clause in re.split(r"[\n.!?;]+", text):
        if clause.lstrip().startswith(">"):
            continue
        if _SUCCESS_CLAIM.search(clause) and not _NEGATED_CLAIM.search(clause):
            return True
    return False


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
        self.edited_since_tests = False
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
        elif name in ("patch_file", "write_file") and not text.startswith(("ERROR:", "No changes:")):
            self.edited_since_tests = True
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
        claims_success = claims_tests_pass(text)
        if claims_success and self.last_tests == "failed":
            self.claim_unverified = True
            if self.test_retries < MAX_TEST_RETRIES:
                self.test_retries += 1
                self.notes.append("false success claim blocked")
                return "retry", (
                    "[SYSTEM] Your last run_tests result was FAILED, so tests do not pass. "
                    "Fix the code with patch_file or write_file, run run_tests again, and only report success if it says PASSED. "
                    "Last result:\n" + self.last_tests_output[-800:]
                )
            self.claim_unverified = True
            return "ok", text + "\n\nCorrection: run_tests last reported FAILED. The tests do not pass.\n" + self.last_tests_output[-600:]
        extra = ""
        if claims_success and self.last_tests in (None, "unavailable"):
            self.claim_unverified = True
            extra = ("\n\nNote: the tests were not run, so the claim that they pass is unverified."
                     if self.last_tests is None else
                     "\n\nNote: the tests could not be run (permission not given), so the claim that they pass is unverified.")
        elif claims_success and self.edited_since_tests and self.last_tests == "passed":
            self.claim_unverified = True
            extra = "\n\nNote: files changed after the last run_tests, so rerun it to confirm."
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
