"""Loop guardrails: check the model's work against real tool results.

Small models hallucinate file names, claim tests pass when they failed, and
describe git output they never read. These checks run inside the agent loop,
check tool evidence and explicitly requested Python bindings, and never change files.

Turn off with NEXUS_GUARDRAILS=0 (for measuring the model alone).
"""
import ast
from dataclasses import dataclass
import keyword
import difflib
import math
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
