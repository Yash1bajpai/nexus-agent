"""One answer to "is a person typing at this terminal?"."""
import os
import sys


def _is_windows_console(stream) -> bool:
    """True when a Windows stream is a real console.

    On Windows `isatty()` can also be True for the NUL device (`< NUL`), so
    ask the console API directly.
    """
    try:
        import ctypes
        import msvcrt
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)))
    except Exception:
        return False


def stdin_is_interactive() -> bool:
    stream = sys.stdin
    try:
        if stream is None or not stream.isatty():
            return False
    except Exception:
        return False
    if os.name == "nt":
        return _is_windows_console(stream)
    return True
