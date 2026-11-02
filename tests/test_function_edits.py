"""Byte-preserving Python function operations and their conservative boundaries."""
import ast
import hashlib
import os
from pathlib import Path

import pytest

from nexus_agent_ai.agent import tools as T
from nexus_agent_ai.agent.refusals import RefusalClass as R, RefusalText


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    T.reset_file_tracking()
    yield
    T.reset_file_tracking()


def ready(content, name="module.py"):
    path = Path(name)
    path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    T.execute_read_file(str(path))
    return path


def change(path, body, *, name="f", cls=None, add=False, state=None):
    if state is None:
        state = ("file", hashlib.sha256(path.read_bytes()).digest())
    return T.execute_tool("add_function" if add else "edit_function", {
        "path": str(path), "function_name": name, "new_body": body,
        **({"class_name": cls} if cls is not None else {}),
    }, expected_state=state)


def refused(result, kind):
    assert isinstance(result, RefusalText), result
    assert result.refusal_class is kind
    assert result.startswith("ERROR:")


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_edit_preserves_exact_prefix_suffix_comments_unicode_and_backup(newline):
    prefix = '# café\u2028 still one physical line' + newline + '# f documentation' + newline
    old = 'def f(x: str = "é") -> str:' + newline + '    return "old"'
    suffix = '  # keep inline' + newline + newline + '# neighboring documentation' + newline + 'def neighbor():' + newline + '    return "unchanged"' + newline
    original = (prefix + old + suffix).encode()
    path = ready(original)
    body = 'def f(x: str = "é") -> str:\n    return "新"'
    result = change(path, body)
    assert result.startswith("Successfully edited"), result
    assert path.read_bytes() == prefix.encode() + body.encode() + suffix.encode()
    assert path.with_name("module.py.bak").read_bytes() == original


def test_method_decorators_async_and_multiline_string_values_are_preserved():
    path = ready('class C:\n    @wrap("é")\n    async def f(self, x=default()) -> annotation():\n        return "old"\n\n    # next docblock\n    def g(self):\n        pass\n')
    fragment = '@wrap("é")\nasync def f(self, x=default()) -> annotation():\n    text = """first\n  literal indentation\nlast"""\n    return text'
    result = change(path, fragment, cls="C")
    assert result.startswith("Successfully edited"), result
    node = ast.parse(path.read_text()).body[0].body[0]
    assert node.body[0].value.value == "first\n  literal indentation\nlast"
    assert b'\n\n    # next docblock\n    def g(self):\n        pass\n' in path.read_bytes()


def test_defaults_annotations_and_decorators_are_never_evaluated():
    marker = Path("executed")
    header = '@open("executed", "w")\ndef f(x=open("executed", "w")) -> open("executed", "w"):'
    path = ready(header + '\n    return 1\n')
    assert change(path, header + '\n    return 2').startswith("Successfully edited")
    assert not marker.exists()
    assert change(path, '@open("executed", "w")\ndef g(x=open("executed", "w")) -> open("executed", "w"):\n    pass', name="g", add=True).startswith("Successfully added")
    assert not marker.exists()


@pytest.mark.parametrize("fragment", [
    'def f(x=2):\n    return 2',
    'def f(y=1):\n    return 2',
    'def f(x: int=1):\n    return 2',
    'def f(x=1) -> int:\n    return 2',
    'async def f(x=1):\n    return 2',
    '@other\ndef f(x=1):\n    return 2',
])
def test_signature_and_async_changes_refuse_even_with_user_rewrite_authority(fragment):
    path = ready('def f(x=1):\n    return 1\n')
    T.set_user_intent("Rewrite module.py and replace f function")
    before = path.read_bytes()
    refused(change(path, fragment), R.CONTENT_LOSS)
    assert path.read_bytes() == before
    assert not path.with_suffix(".py.bak").exists()


@pytest.mark.parametrize("decorators", ["@b\n@a", "@a\n@b( )", "@a"])
def test_decorator_order_and_exact_expression_source_are_immutable(decorators):
    path = ready('@a\n@b()\ndef f():\n    pass\n')
    refused(change(path, decorators + '\ndef f():\n    return 1'), R.CONTENT_LOSS)


def test_function_type_comment_is_preserved():
    path = ready('def f(x):  # type: (int) -> int\n    return x\n')
    refused(change(path, 'def f(x):\n    return x + 1'), R.CONTENT_LOSS)
    assert change(path, 'def f(x):  # type: (int) -> int\n    return x + 1').startswith("Successfully edited")


@pytest.mark.parametrize("fragment", [
    'import os\ndef f():\n    pass',
    'def f():\n    import os\n    pass',
    'def f():\n    from pathlib import Path\n    pass',
    '@wrap\n# exterior\ndef f():\n    pass',
    'def f():\n    pass\nx = 1',
    'def f():\n    pass\ndef g():\n    pass',
    '# exterior\ndef f():\n    pass',
    'def f():\n    pass\n# exterior',
    'return 1',
    'def g():\n    pass',
    'class f:\n    pass',
])
def test_new_body_is_exactly_one_named_complete_definition(fragment):
    path = ready('def f():\n    return 1\n')
    before = path.read_bytes()
    refused(change(path, fragment), R.TARGETING)
    assert path.read_bytes() == before


def test_verbatim_python_syntax_diagnostic():
    path = ready('def f():\n    pass\n')
    broken = 'def f(:\n    pass'
    try:
        ast.parse(broken, filename="<new_body>", type_comments=True)
    except SyntaxError as exc:
        diagnostic = str(exc)
    result = change(path, broken)
    refused(result, R.SYNTAX)
    assert diagnostic in result


@pytest.mark.parametrize("source", [
    'def f():\n    pass\ndef f():\n    pass\n',
    'if True:\n    def f():\n        pass\n',
    'def outer():\n    def f():\n        pass\n',
    'class A:\n    def f(self):\n        pass\nclass B:\n    def f(self):\n        pass\n',
    'def f():\n    pass\nclass C:\n    def f(self):\n        pass\n',
])
def test_bare_targets_never_choose_nested_conditional_or_duplicate_spans(source):
    path = ready(source)
    refused(change(path, 'def f():\n    return 2'), R.AMBIGUITY)
    assert path.read_text() == source


@pytest.mark.parametrize("source", [
    'class C:\n    def f(self):\n        pass\n    def f(self):\n        pass\n',
    'class C:\n    if True:\n        def f(self):\n            pass\n',
    'class C:\n    pass\nclass C:\n    def f(self):\n        pass\n',
    'if True:\n    class C:\n        def f(self):\n            pass\n',
])
def test_class_targets_are_unique_direct_unconditional_methods(source):
    path = ready(source)
    refused(change(path, 'def f(self):\n    return 2', cls="C"), R.AMBIGUITY)


def test_class_name_selects_exact_direct_method():
    path = ready('class A:\n    def f(self):\n        return 1\nclass B:\n    def f(self):\n        return 2\n')
    original = path.read_bytes()
    assert change(path, 'def f(self):\n    return 3', cls="A").startswith("Successfully edited")
    assert path.read_bytes() == original.replace(b'return 1', b'return 3')


@pytest.mark.parametrize("original", [b'# module\r\nx = 1\r\n', b'x = 1', b''])
def test_add_module_keeps_every_existing_byte(original):
    path = ready(original)
    body = 'def f():\n    return "é"'
    assert change(path, body, add=True).startswith("Successfully added")
    separator = b'\n' if original and not original.endswith(b'\n') else b''
    assert path.read_bytes() == original + separator + body.encode() + b'\n'


def test_add_class_at_ast_end_keeps_neighbor_comments_and_all_other_bytes():
    prefix = b'class C:\r\n    x = 1  # keep\r\n'
    suffix = b'    # trailing class comment\r\n\r\n# next docblock\r\nclass D:\r\n    pass\r\n'
    path = ready(prefix + suffix)
    assert change(path, 'def f(self):\n    return 2', add=True, cls="C").startswith("Successfully added")
    assert path.read_bytes() == prefix + b'    def f(self):\n        return 2\n' + suffix


@pytest.mark.parametrize("source", [
    'def f():\n    pass\n', 'f = 1\n', 'import math as f\n',
    'if True:\n    def f():\n        pass\n', 'class f:\n    pass\n',
])
def test_add_rejects_preexisting_scope_bindings(source):
    path = ready(source)
    refused(change(path, 'def f():\n    pass', add=True), R.AMBIGUITY)
    assert path.read_text() == source


def test_add_does_not_confuse_other_scopes_with_target_scope():
    path = ready('class C:\n    def f(self):\n        pass\n')
    assert change(path, 'def f():\n    pass', add=True).startswith("Successfully added")


def test_read_guard_returns_typed_data_without_edit_then_allows_retry():
    path = Path("module.py")
    original = b'def f():\n    return 1\n'
    path.write_bytes(original)
    result = change(path, 'def f():\n    return 2')
    refused(result, R.BLIND_EDIT)
    assert "DATA: RAW file content" in result
    assert path.read_bytes() == original
    assert change(path, 'def f():\n    return 2').startswith("Successfully edited")


def test_same_size_same_mtime_external_change_is_not_read_authority():
    path = ready('def f():\n    return 1\n')
    before = path.stat()
    path.write_bytes(b'def f():\n    return 9\n')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    refused(change(path, 'def f():\n    return 2'), R.STALE_STATE)
    assert b'return 9' in path.read_bytes()


def test_fresh_private_snapshot_is_checked_against_actual_read_bytes():
    path = ready('def f():\n    return 1\n')
    snapshot = ("file", hashlib.sha256(path.read_bytes()).digest())
    path.write_bytes(b'def f():\n    return 9\n')
    T.execute_read_file(str(path))
    refused(change(path, 'def f():\n    return 2', state=snapshot), R.STALE_STATE)


@pytest.mark.parametrize("state", [("file", b"short"), ("missing", None), ("file", "a" * 64), ["file", b"a" * 32]])
def test_invalid_private_snapshot_is_not_accepted(state):
    path = ready('def f():\n    pass\n')
    refused(change(path, 'def f():\n    return 2', state=state), R.STALE_STATE)


@pytest.mark.parametrize("content", [b'# coding: latin-1\ndef f():\n    return "\xe9"\n', b'\xef\xbb\xbfdef f():\n    pass\n', b'def f():\r    pass\r'])
def test_unsupported_encoding_and_line_layout_refuse(content):
    path = ready(content)
    refused(change(path, 'def f():\n    return 2'), R.TARGETING)
    assert path.read_bytes() == content


def test_unsupported_extension_and_missing_file():
    path = ready('def f():\n    pass\n', "module.txt")
    refused(change(path, 'def f():\n    pass'), R.TARGETING)
    result = T.execute_add_function("absent.py", "f", 'def f():\n    pass')
    refused(result, R.NOT_FOUND)


def test_inline_class_and_tab_indentation_refuse_instead_of_guessing():
    path = ready('class C: pass\n')
    refused(change(path, 'def f(self):\n    pass', add=True, cls="C"), R.AMBIGUITY)
    path = ready('class C:\n\tdef f(self):\n\t\tpass\n')
    refused(change(path, 'def f(self):\n    return 2', cls="C"), R.AMBIGUITY)


def test_no_change_does_not_create_backup():
    path = ready('def f():\n    pass\n')
    assert change(path, 'def f():\n    pass').startswith("No changes:")
    assert not Path("module.py.bak").exists()


def test_atomic_prepublication_snapshot_rejects_external_change(monkeypatch):
    path = ready('def f():\n    return 1\n')
    real_update = T._atomic_file_update

    def concurrent_update(p, content, **kwargs):
        p.write_bytes(b'def f():\n    return 9\n')
        return real_update(p, content, **kwargs)

    monkeypatch.setattr(T, "_atomic_file_update", concurrent_update)
    refused(change(path, 'def f():\n    return 2'), R.STALE_STATE)
    assert b'return 9' in path.read_bytes()
    assert not Path("module.py.bak").exists()


def test_old_patch_and_write_crlf_normalization_remains_unchanged():
    path = ready(b'def f():\r\n    return 1\r\n')
    assert T.execute_patch_file(str(path), 'return 1', 'return 2').startswith("Successfully patched")
    assert path.read_bytes() == b'def f():\r\n    return 2\r\n'
    assert T.execute_write_file(str(path), 'def f():\n    return 3\n').startswith("Successfully wrote")
    assert path.read_bytes() == b'def f():\r\n    return 3\r\n'


def test_tool_visibility_and_private_schema():
    names = {t.name for t in T.get_all_tools()}
    readonly = {t.name for t in T.get_readonly_tools()}
    assert {"edit_function", "add_function"} <= names
    assert not {"edit_function", "add_function"} & readonly
    for tool in (T.EDIT_FUNCTION_TOOL, T.ADD_FUNCTION_TOOL):
        assert tool.input_schema["required"] == ["path", "function_name", "new_body"]
        assert "expected_state" not in tool.input_schema["properties"]


def test_missing_private_snapshot_fails_closed_after_recorded_read():
    path = ready('def f():\n    return 1\n')
    result = T.execute_tool("edit_function", {
        "path": str(path), "function_name": "f",
        "new_body": 'def f():\n    return 2',
    })
    refused(result, R.STALE_STATE)
    assert path.read_bytes() == b'def f():\n    return 1\n'


def test_decorator_parentheses_are_exact_header_source_not_just_ast_equivalence():
    path = ready('@a\ndef f():\n    return 1\n')
    refused(change(path, '@(a)\ndef f():\n    return 2'), R.CONTENT_LOSS)


def test_parenthesized_multiline_module_decorator_has_proven_at_token_start():
    prefix = b'# preserve preceding decorator documentation\n'
    header = '@(\n    wrap("é")\n)\ndef f():'
    path = ready(prefix + (header + '\n    return 1\n').encode())
    assert change(path, header + '\n    return 2').startswith("Successfully edited")
    assert path.read_bytes() == prefix + (header + '\n    return 2\n').encode()


def test_method_multiline_decorator_reindent_cannot_silently_change_header_bytes():
    path = ready('class C:\n    @wrap(\n        "x"\n    )\n    def f(self):\n        return 1\n')
    original = path.read_bytes()
    fragment = '@wrap(\n        "x"\n    )\ndef f(self):\n    return 2'
    refused(change(path, fragment, cls="C"), R.CONTENT_LOSS)
    assert path.read_bytes() == original


def test_add_class_method_preserves_multiline_literal_values_and_suffix():
    path = ready('class C:\n    pass\n\n# neighbor docblock\nVALUE = 1\n')
    fragment = 'def f(self):\n    return """first\n  literal\nlast"""'
    assert change(path, fragment, cls="C", add=True).startswith("Successfully added")
    tree = ast.parse(path.read_text())
    assert tree.body[0].body[-1].body[0].value.value == "first\n  literal\nlast"
    assert path.read_bytes().endswith(b'\n\n# neighbor docblock\nVALUE = 1\n')


def test_missing_function_and_class_are_typed_not_found():
    path = ready('def g():\n    pass\n')
    refused(change(path, 'def f():\n    pass'), R.NOT_FOUND)
    refused(change(path, 'def f(self):\n    pass', cls="Missing"), R.NOT_FOUND)


def test_new_operations_reuse_sensitive_path_and_unsafe_backup_guards(tmp_path):
    path = ready('def f():\n    return 1\n', "private_secret.py")
    refused(change(path, 'def f():\n    return 2'), R.SENSITIVE_PATH)
    path = ready('def f():\n    return 1\n')
    outside = tmp_path / "unrelated.txt"
    outside.write_bytes(b"untouched\n")
    backup = path.with_name("module.py.bak")
    try:
        backup.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Filesystem cannot create symlinks: {exc}")
    refused(change(path, 'def f():\n    return 2'), R.PERMISSION)
    assert path.read_bytes() == b'def f():\n    return 1\n'
    assert outside.read_bytes() == b"untouched\n"


def test_definition_body_comments_are_allowed_without_consuming_neighbor_docblock():
    suffix = b'\n\n# neighbor documentation\ndef g():\n    pass\n'
    path = ready(b'def f():\n    return 1' + suffix)
    fragment = 'def f():\n    # body explanation\n    return 2\n    # trailing body explanation'
    assert change(path, fragment).startswith("Successfully edited")
    assert path.read_bytes() == fragment.encode() + suffix


@pytest.mark.parametrize("source", [
    'try:\n    pass\nexcept Exception as f:\n    pass\n',
    'match value:\n    case {"key": f}:\n        pass\n',
    'match value:\n    case [*f]:\n        pass\n',
    'match value:\n    case {"key": x, **f}:\n        pass\n',
])
def test_add_rejects_nonassignment_scope_bindings(source):
    path = ready(source)
    refused(change(path, 'def f():\n    pass', add=True), R.AMBIGUITY)
