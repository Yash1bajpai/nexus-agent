"""Pure bounded-memory physical text facts over an already-open binary stream.

This module neither opens paths nor grants file-read/edit proof. The caller owns
handle identity checks, process deadlines, and classification of I/O failures.
"""

import codecs
from collections import deque
import hashlib

from .text_count_limits import MAX_LITERAL_BYTES, MAX_SCAN_BYTES, READ_CHUNK_BYTES


def _facts(reason, read_bytes, digest, partial, *, physical_lines=None,
           count=None, total_count=None, minimum=None, maximum=None):
    return {
        "status": "complete" if reason is None else "incomplete",
        "reason": reason,
        "count": count,
        "total_count": total_count,
        "physical_lines": physical_lines,
        "per_line_count": {"min": minimum, "max": maximum},
        "source": {
            "bytes": read_bytes,
            "sha256": digest.hexdigest(),
            "partial": partial,
        },
    }


def scan_stream(stream, literal, whole_word=True, line=None, *,
                chunk_size=READ_CHUNK_BYTES, max_bytes=MAX_SCAN_BYTES):
    """Count a raw Unicode literal using strict streaming UTF-8 and KMP.

    ``stream`` supplies bounded binary ``read(n)`` calls; exceptions propagate to
    the worker. Invalid arguments raise ValueError before any read. Runtime
    callers use the fixed defaults; private bounds permit deterministic tests.

    Only an initial BOM is removed from text, never from the digest. LF and CRLF
    close physical lines; lone CR is content. Empty input has no lines and a
    final LF creates no phantom line. Whole-word boundaries mean adjacent
    Unicode isalnum()/underscore codepoints, even for nonword literals. Accepted
    matches do not overlap; rejected boundary candidates leave KMP overlaps
    available. There is no normalization, case folding, or regex interpretation.

    All count fields are null on failure, including a missing requested line.
    EOF missing-line facts retain the known physical line count and full digest.
    Decode/cap facts expose only a partial digest of exactly the bytes read.
    At most max_bytes + 1 bytes are read: the extra byte detects, not truncates,
    an oversized source. Handle stability remains the worker's responsibility.

    State is O(len(literal) + chunk_size): a KMP prefix table, a boolean boundary
    window, one bounded decoded chunk, constant counters, and codec/hash state.
    No line, token, source buffer, or per-line count array is retained.
    """
    if (not isinstance(literal, str) or not literal
            or len(literal) > MAX_LITERAL_BYTES
            or "\r" in literal or "\n" in literal):
        raise ValueError("invalid literal")
    if len(literal.encode("utf-8", errors="strict")) > MAX_LITERAL_BYTES:
        raise ValueError("literal limit")
    if not isinstance(whole_word, bool):
        raise ValueError("invalid whole_word")
    if line is not None and (type(line) is not int or line < 1):
        raise ValueError("invalid line")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("invalid chunk_size")
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("invalid max_bytes")

    length = len(literal)
    prefix = [0] * length
    matched = 0
    for index in range(1, length):
        while matched and literal[index] != literal[matched]:
            matched = prefix[matched - 1]
        if literal[index] == literal[matched]:
            matched += 1
        prefix[index] = matched

    # A full candidate ending now needs the word flag immediately before its
    # start. Keeping booleans, rather than text, also handles nonword literals.
    boundary_window = deque(maxlen=length + 1) if whole_word else None
    # utf-8-sig accepts a truncated BOM prefix at EOF on supported Python builds.
    # Decode strictly first; strip only a complete initial marker.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    initial_character = True
    digest = hashlib.sha256()
    read_bytes = 0
    matched = 0
    pending_right = False
    line_open = False
    physical_lines = 0
    line_count = 0
    total_count = 0
    selected_count = None
    minimum = None
    maximum = None

    while True:
        chunk = stream.read(min(chunk_size, max_bytes - read_bytes + 1))
        eof = not chunk
        if chunk:
            read_bytes += len(chunk)
            digest.update(chunk)
            if read_bytes > max_bytes:
                return _facts("size_limit", read_bytes, digest, True)
        try:
            text = decoder.decode(chunk, final=eof)
        except UnicodeDecodeError:
            return _facts("decode_error", read_bytes, digest, True)
        if initial_character and text:
            initial_character = False
            if text[0] == "\ufeff":
                text = text[1:]

        for character in text:
            word = character == "_" or character.isalnum()
            if pending_right:
                if not word:
                    line_count += 1
                    total_count += 1
                    # Only an accepted candidate consumes its span. The current
                    # character remains eligible to start the next match.
                    matched = 0
                pending_right = False

            if character == "\n":
                physical_lines += 1
                if physical_lines == line:
                    selected_count = line_count
                minimum = line_count if minimum is None else min(minimum, line_count)
                maximum = line_count if maximum is None else max(maximum, line_count)
                line_count = 0
                line_open = False
                matched = 0
                if whole_word:
                    boundary_window.clear()
                continue

            line_open = True
            if whole_word:
                boundary_window.append(word)
            while matched and character != literal[matched]:
                matched = prefix[matched - 1]
            if character == literal[matched]:
                matched += 1
            if matched == length:
                if whole_word:
                    if len(boundary_window) <= length or not boundary_window[0]:
                        pending_right = True
                    # A left/right rejection must not discard later overlapping
                    # candidates, unlike an accepted nonoverlapping match.
                    matched = prefix[matched - 1]
                else:
                    line_count += 1
                    total_count += 1
                    matched = 0

        if eof:
            break

    if pending_right:
        line_count += 1
        total_count += 1
    if line_open:
        physical_lines += 1
        if physical_lines == line:
            selected_count = line_count
        minimum = line_count if minimum is None else min(minimum, line_count)
        maximum = line_count if maximum is None else max(maximum, line_count)
    if line is not None and selected_count is None:
        return _facts("missing_line", read_bytes, digest, False,
                      physical_lines=physical_lines)
    return _facts(None, read_bytes, digest, False,
                  physical_lines=physical_lines,
                  count=total_count if line is None else selected_count,
                  total_count=total_count, minimum=minimum, maximum=maximum)
