"""Regression tests for Windows and macOS manual qualification workflows."""
import ast
from pathlib import Path
import re
import pytest
WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def _load_workflow(name: str) -> str:
    path = WORKFLOWS_DIR / name
    assert path.is_file(), f"Workflow file not found: {path}"
    return path.read_text(encoding="utf-8")


def test_windows_workflow_structure_and_steps():
    raw = _load_workflow("windows.yml")
    assert re.search(r"first-run:\s*.*?\bruns-on:\s*windows-latest", raw, re.DOTALL)
    assert re.search(r"first-run:\s*.*?github\.event_name\s*==\s*'workflow_dispatch'", raw, re.DOTALL)

    step_names = re.findall(r"-\s*name:\s*(.+)", raw)
    assert any("S1 -" in name for name in step_names)
    assert any("S2 -" in name for name in step_names)
    assert any("S3 -" in name for name in step_names)
    assert any("S5 -" in name for name in step_names)
    assert any("S7 -" in name for name in step_names)

    # S1 assertion check
    assert "12541447" in raw
    assert "4829102 + 7712345" in raw

    # S3 final answer check
    assert "$finalSources" in raw
    assert "S3 final user-facing answer did not log an external HTTP(S) source URL" in raw

    # S5 review check
    assert "review_target.py" in raw
    assert "CAPABILITY-MISS" in raw

    # S7 debug check
    assert "buggy.py" in raw
    assert "ZeroDivisionError" in raw
    assert "PARTIAL" in raw


def test_macos_workflow_structure_and_steps():
    raw = _load_workflow("macos.yml")
    assert re.search(r"first-run:\s*.*?\bruns-on:\s*macos-latest", raw, re.DOTALL)
    assert re.search(r"first-run:\s*.*?github\.event_name\s*==\s*'workflow_dispatch'", raw, re.DOTALL)

    step_names = re.findall(r"-\s*name:\s*(.+)", raw)
    assert any("S1 -" in name for name in step_names)
    assert any("S2 -" in name for name in step_names)
    assert any("S3 -" in name for name in step_names)

    # S1 assertion check
    assert "12541447" in raw
    assert "4829102 + 7712345" in raw

    # S2 check
    assert "s2target.py" in raw
    assert "S2: PASS - EDITED" in raw
    assert "S2: PASS - HONEST FAILURE" in raw

    # S3 check
    assert "$finalSources" in raw
    assert "macos-first-run-qualification" in raw


@pytest.mark.parametrize("exit_code,output,expected_status,should_raise", [
    (0, "The function add performs subtraction (a - b) instead of addition.", "PASS", False),
    (0, "Bug found: minus operator used in add function.", "PASS", False),
    (0, "The function looks fine and code style is clean.", "CAPABILITY-MISS", False),
    (1, "Error: fatal crash", "", True),
])
def test_s5_review_acceptance_and_capability_miss_logic(exit_code, output, expected_status, should_raise):
    def eval_s5(code, text):
        if code != 0:
            raise RuntimeError(f"S5 review failed with exit code {code}")
        if re.search(r"(?i)\b(?:subtract(?:ion|s|ed|ing)?|minus|a\s*-\s*b)\b", text):
            return "S5: PASS (exit 0, review identified subtraction mismatch)"
        return "S5: CAPABILITY-MISS (exit 0, review did not mention add/subtract mismatch)"

    if should_raise:
        with pytest.raises(RuntimeError):
            eval_s5(exit_code, output)
    else:
        result = eval_s5(exit_code, output)
        assert expected_status in result


@pytest.mark.parametrize("exit_code,code,output,expected_status,should_raise", [
    # Fixed with zero-guard
    (0, "def divide(a, b):\n    if b == 0:\n        return 0\n    return a / b",
     "Fixed the bug.", "PASS (exit 0", False),
    (0, "def divide(a, b):\n    return a / b if b != 0 else 0",
     "Updated divide safely.", "PASS (exit 0", False),
    (0, "def divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return 0",
     "Handled ZeroDivisionError.", "PASS (exit 0", False),

    # Fixed but claimed verification without executing run_code / run_tests -> PARTIAL
    (0, "def divide(a, b):\n    if b == 0:\n        return 0\n    return a / b",
     "The file has been updated safely. I verified the changes and ran a test to confirm the fix.",
     "PARTIAL", False),
    # Fixed and executed run_tests -> PASS
    (0, "def divide(a, b):\n    if b == 0:\n        return 0\n    return a / b",
     "The file has been updated safely. [ACTION] run_tests()\nAll tests pass.",
     "PASS (exit 0", False),

    # Exit 0 but NOT guarded -> raises
    (0, "def divide(a, b):\n    return a / b", "All done.", "", True),

    # Honest failure with standard footer -> PASS - HONEST FAILURE
    (1, "def divide(a, b):\n    return a / b",
     "[ERROR] Request incomplete. Tokens: 100 (local) | Est. cost: $0.0000 | Time: 1.0s",
     "PASS - HONEST FAILURE", False),

    # Fatal crash / malformed outcome -> raises
    (1, "def divide(a, b):\n    return a / b", "Fatal execution error: segmentation fault", "", True),
    (2, "", "Unknown exit code", "", True),
])
def test_s7_debug_acceptance_partial_and_honest_failure_logic(exit_code, code, output, expected_status, should_raise):
    def eval_s7(code_exit, py_src, chat_text):
        try:
            tree = ast.parse(py_src)
        except Exception:
            tree = ast.Module(body=[], type_ignores=[])
        has_if = any(isinstance(n, (ast.If, ast.IfExp)) for n in ast.walk(tree))
        has_try = any(isinstance(n, ast.Try) for n in ast.walk(tree))
        has_zero = any(isinstance(n, ast.Constant) and n.value == 0 for n in ast.walk(tree))
        guarded = bool(tree.body) and ((has_if and (has_zero or "b" in py_src)) or (has_try and "ZeroDivisionError" in py_src))

        if code_exit == 0:
            if not guarded:
                raise RuntimeError("S7 falsely reported success without a valid zero-division guard in buggy.py")
            claims_verif = bool(re.search(r"(?i)\b(?:verified|ran\s+(?:a\s+)?tests?|checked\s+the\s+file)\b", chat_text))
            ran_tests = bool(re.search(r"\[ACTION\]\s+(?:run_code|run_tests)\b", chat_text))
            if claims_verif and not ran_tests:
                return "S7: PARTIAL (exit 0, zero-guard added, but claimed verification without executing run_code or run_tests)"
            return "S7: PASS (exit 0, AST-confirmed zero-guard)"
        else:
            if code_exit != 1 or not re.search(r"(?s)Tokens:.*Time:", chat_text) or "[ERROR]" not in chat_text or "Fatal execution error:" in chat_text:
                raise RuntimeError(f"S7 failed to run normally; exit {code_exit} is not an honest agent outcome")
            return f"S7: PASS - HONEST FAILURE (agent exit {code_exit}; not accepted as completed)"

    if should_raise:
        with pytest.raises(RuntimeError):
            eval_s7(exit_code, code, output)
    else:
        result = eval_s7(exit_code, code, output)
        assert expected_status in result


def test_s3_final_answer_url_extraction_logic():
    def extract_final_sources(chat_log):
        final_text = chat_log
        m_panel = re.search(r"(?s)╭─+\s*Response\s*─+╮\s*(?P<ans>.*?)\s*╰─+", final_text)
        if m_panel:
            final_text = m_panel.group("ans")
        else:
            parts = re.split(r"(?m)^\s*\[OBSERVE\].*$", final_text)
            if len(parts) > 1:
                final_text = parts[-1]
            m_foot = re.search(r"(?s)(?P<ans>.*?)\s*Tokens:\s*", final_text)
            if m_foot:
                final_text = m_foot.group("ans")
        return [
            m.strip().rstrip(".,;)]}>\"'") for m in re.findall(r"https?://[^\s<>\"'|]+", final_text)
            if not m.startswith("http://127.0.0.1") and "localhost" not in m
        ]

    # Valid final answer with source URL
    streamed_log = (
        "  [ACTION] search_web(query=\"latest python\")\n"
        "  [OBSERVE] Done (0.5s) -> Search results for: 'latest python' | URL: https://python.org ...\n\n"
        "The latest stable Python is 3.12.3.\n\n"
        "Source: https://www.python.org/downloads/latest/\n\n"
        "  Tokens: N/A (local) | Time: 1.0s\n"
    )
    sources = extract_final_sources(streamed_log)
    assert "https://www.python.org/downloads/latest/" in sources

    # Panel formatted log
    panel_log = (
        "  [ACTION] search_web(query=\"latest python\")\n"
        "  [OBSERVE] Done (0.5s) -> Search results ...\n\n"
        "╭────────────────────────────────── Response ──────────────────────────────────╮\n"
        "│ The latest stable Python is 3.12.3.                                          │\n"
        "│ Source: https://www.python.org/downloads/latest/                             │\n"
        "╰──────────────────────────────────────────────────────────────────────────────╯\n"
        "  Tokens: N/A (local) | Time: 1.0s\n"
    )
    sources_panel = extract_final_sources(panel_log)
    assert "https://www.python.org/downloads/latest/" in sources_panel

    # Observation has URL, but final answer does NOT
    missing_url_log = (
        "  [ACTION] search_web(query=\"latest python\")\n"
        "  [OBSERVE] Done (0.5s) -> Search results for: 'latest python' | URL: https://python.org ...\n\n"
        "The latest stable Python is 3.12.3.\n\n"
        "  Tokens: N/A (local) | Time: 1.0s\n"
    )
    assert len(extract_final_sources(missing_url_log)) == 0
