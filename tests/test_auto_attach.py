from nexus_agent_ai.agent.core import attach_named_files


def test_edit_request_attaches_existing_files(tmp_path, monkeypatch):
    (tmp_path / "calc.py").write_text("x = 1\n")
    (tmp_path / "test_calc.py").write_text("y = 2\n")
    monkeypatch.chdir(tmp_path)
    out = attach_named_files("Fix the bug in calc.py so test_calc.py passes.")
    assert "@calc.py" in out and "@test_calc.py" in out
    assert "@@" not in out


def test_missing_and_outside_files_not_attached(tmp_path, monkeypatch):
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")
    (tmp_path / "secret.py").write_text("s")
    out = attach_named_files("Fix the bug in nothere.py and ../secret.py")
    assert "@" not in out


def test_questions_are_untouched(tmp_path, monkeypatch):
    (tmp_path / "calc.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    assert attach_named_files("What does calc.py do?") == "What does calc.py do?"
