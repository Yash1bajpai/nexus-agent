"""Deterministic scanner tests; no filesystem, process, or model is needed."""

import hashlib
from io import BytesIO
from itertools import product
import tracemalloc

import pytest

from nexus_agent_ai.agent.text_count_limits import (
    MAX_LITERAL_BYTES,
    MAX_SCAN_BYTES,
    READ_CHUNK_BYTES,
)
from nexus_agent_ai.agent.text_count_scan import scan_stream


def _scan(text, literal, whole_word=True, line=None, chunk_size=READ_CHUNK_BYTES):
    return scan_stream(BytesIO(text.encode("utf-8")), literal, whole_word, line,
                       chunk_size=chunk_size)


def _assert_complete(result, raw, count, total, lines, minimum, maximum):
    assert result == {
        "status": "complete",
        "reason": None,
        "count": count,
        "total_count": total,
        "physical_lines": lines,
        "per_line_count": {"min": minimum, "max": maximum},
        "source": {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "partial": False,
        },
    }


def _assert_incomplete(result, reason, raw, *, lines=None, partial=True):
    assert result == {
        "status": "incomplete",
        "reason": reason,
        "count": None,
        "total_count": None,
        "physical_lines": lines,
        "per_line_count": {"min": None, "max": None},
        "source": {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "partial": partial,
        },
    }


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 7, READ_CHUNK_BYTES])
@pytest.mark.parametrize("text,lines", [
    ("", 0),
    ("\ufeff", 0),
    ("\n", 1),
    ("\r\n", 1),
    ("\n\n", 2),
    ("\r", 1),
    ("\r\r", 1),
    ("word", 1),
    ("word\n", 1),
    ("word\r\n", 1),
    ("word\n\n", 2),
    ("\nword", 2),
    ("word\rword", 1),
    ("word\r\nword\n\r", 3),
])
def test_physical_lines_and_exact_source(text, lines, chunk_size):
    result = _scan(text, "absent", chunk_size=chunk_size)
    bound = 0 if lines else None
    _assert_complete(result, text.encode("utf-8"), 0, 0, lines, bound, bound)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 9, READ_CHUNK_BYTES])
def test_unicode_bom_literal_and_boundary_splits(chunk_size):
    text = "\ufeff猫 猫猫 _猫 猫_ é猫 猫é 猫\r\n猫\r猫\n猫"
    result = _scan(text, "猫", chunk_size=chunk_size)
    _assert_complete(result, text.encode("utf-8"), 5, 5, 3, 1, 2)
    substring = _scan(text, "猫", False, chunk_size=chunk_size)
    _assert_complete(substring, text.encode("utf-8"), 11, 11, 3, 1, 8)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5])
def test_only_initial_bom_is_an_encoding_marker(chunk_size):
    text = "\ufeff\ufeff\n\ufeff"
    result = _scan(text, "\ufeff", False, chunk_size=chunk_size)
    _assert_complete(result, text.encode("utf-8"), 2, 2, 2, 1, 1)
    _assert_complete(_scan("\ufeff", "\ufeff", False, chunk_size=chunk_size),
                     b"\xef\xbb\xbf", 0, 0, 0, None, None)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("text,literal,whole_word,expected", [
    ("aaaaa", "aa", False, 2),
    ("abababa", "aba", False, 2),
    ("aaaaa", "aa", True, 0),
    ("aa aa", "aa", True, 2),
    ("a!!!", "!!", True, 1),  # Left-rejected overlap does not consume the span.
    ("!!a", "!!", True, 0),  # Right-rejected candidate is not a final count.
    ("!!!!", "!!", True, 2),
    ("x! !x ! !", "!", True, 2),
    ("_cat cat_ cat", "cat", True, 1),
    ("écat cat猫 cat\u0301", "cat", True, 1),
    ("Cat cat CAT", "cat", True, 1),
    ("é e\u0301 é", "é", True, 2),
    ("é e\u0301 é", "e\u0301", True, 1),
    ("a.b aXb [x] .* ^$", "a.b", True, 1),
    ("a.b aXb [x] .* ^$", "[x]", True, 1),
    ("a.b aXb [x] .* ^$", ".*", True, 1),
    ("a.b aXb [x] .* ^$", "^$", True, 1),
    ("a\rb\r\na b", "b", True, 2),
    ("a\nb", "ab", False, 0),
    ("x\u0085x\u2028x\u2029x", "x", True, 4),
    ("a\x00b \x00", "\x00", False, 2),
    ("\t \t", "\t", True, 2),
])
def test_literal_word_and_nonoverlap_semantics(text, literal, whole_word,
                                             expected, chunk_size):
    result = _scan(text, literal, whole_word, chunk_size=chunk_size)
    assert result["status"] == "complete"
    assert result["count"] == expected
    assert result["total_count"] == expected


def _reference_count(text, literal, whole_word):
    """Small independent lookaround search oracle, deliberately not KMP."""
    count = 0
    offset = 0
    while True:
        start = text.find(literal, offset)
        if start < 0:
            return count
        end = start + len(literal)
        left_word = start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_")
        right_word = end < len(text) and (text[end].isalnum() or text[end] == "_")
        if not whole_word or not (left_word or right_word):
            count += 1
            offset = end
        else:
            offset = start + 1


@pytest.mark.parametrize("whole_word", [False, True])
def test_small_exhaustive_overlap_oracle(whole_word):
    for size in range(5):
        for characters in product("a!_", repeat=size):
            text = "".join(characters)
            for literal in ("a", "aa", "!!", "a!", "!a", "a!a", "!a!", "a_a"):
                expected = _reference_count(text, literal, whole_word)
                result = _scan(text, literal, whole_word, chunk_size=1)
                assert result["count"] == expected, (text, literal, whole_word)


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 13, READ_CHUNK_BYTES])
def test_selected_line_still_scans_whole_file_and_minmax(chunk_size):
    text = "hit hit\r\n\nmiss\rhit\nhit hit hit\n"
    raw = text.encode("utf-8")
    for line, expected in [(None, 6), (1, 2), (2, 0), (3, 1), (4, 3)]:
        result = _scan(text, "hit", line=line, chunk_size=chunk_size)
        _assert_complete(result, raw, expected, 6, 4, 0, 3)
    missing = _scan(text, "hit", line=5, chunk_size=chunk_size)
    _assert_incomplete(missing, "missing_line", raw, lines=4, partial=False)


@pytest.mark.parametrize("text,line,lines", [
    ("", 1, 0), ("\ufeff", 1, 0), ("\n", 2, 1), ("x\n", 2, 1),
])
def test_missing_line_is_not_a_zero(text, line, lines):
    result = _scan(text, "x", line=line, chunk_size=1)
    _assert_incomplete(result, "missing_line", text.encode("utf-8"),
                       lines=lines, partial=False)


def test_equal_line_counts_and_empty_line_zero():
    _assert_complete(_scan("x\nx\nx\n", "x"), b"x\nx\nx\n", 3, 3, 3, 1, 1)
    _assert_complete(_scan("\n", "x", line=1), b"\n", 0, 0, 1, 0, 0)


@pytest.mark.parametrize("raw", [
    b"\xff", b"\x80", b"\xc0\xaf", b"\xed\xa0\x80", b"\xf4\x90\x80\x80",
    b"\xef", b"\xef\xbb", b"\xe2\x82", b"hit\n\xff", b"hit\n\xe2\x82",
])
@pytest.mark.parametrize("chunk_size", [1, READ_CHUNK_BYTES])
def test_strict_decode_errors_null_all_counts(raw, chunk_size):
    stream = BytesIO(raw)
    result = scan_stream(stream, "hit", line=1, chunk_size=chunk_size)
    _assert_incomplete(result, "decode_error", raw[:stream.tell()])


def test_decode_failure_digest_is_exact_read_prefix_not_whole_source():
    raw = b"hit\n\xffUNREAD"
    stream = BytesIO(raw)
    result = scan_stream(stream, "hit", chunk_size=5)
    assert stream.tell() == 5
    _assert_incomplete(result, "decode_error", raw[:5])


def test_requested_line_does_not_hide_later_decode_failure():
    raw = b"hit\nhit\xff"
    result = scan_stream(BytesIO(raw), "hit", line=1, chunk_size=1)
    _assert_incomplete(result, "decode_error", raw)


@pytest.mark.parametrize("chunk_size", [1, 2, 4, READ_CHUNK_BYTES])
def test_byte_cap_exact_eof_and_one_byte_sentinel(chunk_size):
    raw = b"hit\nhit"
    result = scan_stream(BytesIO(raw), "hit", chunk_size=chunk_size, max_bytes=len(raw))
    _assert_complete(result, raw, 2, 2, 2, 1, 1)
    stream = BytesIO(raw + b"!UNREAD")
    result = scan_stream(stream, "hit", line=1,
                         chunk_size=chunk_size, max_bytes=len(raw))
    assert stream.tell() == len(raw) + 1
    _assert_incomplete(result, "size_limit", raw + b"!")


def test_zero_cap_and_size_before_decode_precedence():
    _assert_complete(scan_stream(BytesIO(b""), "x", max_bytes=0),
                     b"", 0, 0, 0, None, None)
    _assert_incomplete(scan_stream(BytesIO(b"\xffmore"), "x", max_bytes=0),
                       "size_limit", b"\xff")


class _RepeatedBytesStream:
    """Generate a huge line without storing it or recording per-read history."""

    def __init__(self, size):
        self.remaining = size
        self.max_read = 0
        self.reads = 0

    def read(self, size):
        assert size > 0
        self.max_read = max(self.max_read, size)
        self.reads += 1
        amount = min(size, self.remaining)
        self.remaining -= amount
        return b"a" * amount


def test_fixed_default_cap_and_streaming_reads():
    stream = _RepeatedBytesStream(MAX_SCAN_BYTES + 100)
    result = scan_stream(stream, "a", False)
    assert result["reason"] == "size_limit"
    assert result["source"]["bytes"] == MAX_SCAN_BYTES + 1
    assert result["count"] is None
    assert result["total_count"] is None
    assert result["physical_lines"] is None
    assert result["per_line_count"] == {"min": None, "max": None}
    assert result["source"]["sha256"] == hashlib.sha256(b"a" * (MAX_SCAN_BYTES + 1)).hexdigest()
    assert result["source"]["partial"] is True
    assert stream.remaining == 99
    assert stream.max_read <= READ_CHUNK_BYTES


def test_huge_line_uses_bounded_memory_not_a_line_or_token_buffer():
    size = 2 * 1024 * 1024
    stream = _RepeatedBytesStream(size)
    literal = "a" * (MAX_LITERAL_BYTES - 1) + "b"
    tracemalloc.start()
    try:
        result = scan_stream(stream, literal, max_bytes=size)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["status"] == "complete"
    assert result["count"] == 0
    assert result["total_count"] == 0
    assert result["physical_lines"] == 1
    assert result["per_line_count"] == {"min": 0, "max": 0}
    assert stream.remaining == 0
    assert stream.reads > 100
    assert stream.max_read <= READ_CHUNK_BYTES
    assert peak < 192 * 1024


class _NeverRead:
    def read(self, size):
        raise AssertionError("invalid arguments must not read")


@pytest.mark.parametrize("arguments", [
    {"literal": ""}, {"literal": None}, {"literal": 1},
    {"literal": "\n"}, {"literal": "x\ry"}, {"literal": "\ud800"},
    {"literal": "a" * (MAX_LITERAL_BYTES + 1)},
    {"literal": "é" * (MAX_LITERAL_BYTES // 2 + 1)},
    {"whole_word": 1}, {"whole_word": None},
    {"line": 0}, {"line": -1}, {"line": True}, {"line": 1.0},
    {"chunk_size": 0}, {"chunk_size": -1}, {"chunk_size": True},
    {"max_bytes": -1}, {"max_bytes": False}, {"max_bytes": 1.0},
])
def test_invalid_arguments_fail_before_read(arguments):
    query = {"literal": "x", **arguments}
    with pytest.raises(ValueError):
        scan_stream(_NeverRead(), **query)


def test_exact_utf8_literal_limit_is_accepted():
    literal = "é" * (MAX_LITERAL_BYTES // 2)
    result = _scan(literal, literal, chunk_size=1)
    assert result["count"] == 1
    assert result["status"] == "complete"


def test_stream_io_error_propagates_for_worker_classification():
    class BrokenStream:
        def read(self, size):
            raise OSError("private stream failure")

    with pytest.raises(OSError):
        scan_stream(BrokenStream(), "x")
