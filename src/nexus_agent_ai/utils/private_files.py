"""Owner-only permissions for persistent secrets and conversation history."""
import os
from pathlib import Path


def restrict_permissions(path: Path, mode: int) -> None:
    """Replace access permissions, failing closed if privacy cannot be enforced.

    On Windows replace the protected DACL atomically; chmod cannot restrict
    Windows ACLs, and resetting inherited ACLs first can expose private data.
    """
    if os.name != "nt":
        os.chmod(path, mode)
        return
    _restrict_windows(path)


def _restrict_windows(path: Path) -> None:

    import csv
    import ctypes
    import io
    import subprocess
    from ctypes import wintypes

    identity = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True,
        text=True, check=True, timeout=20,
    )
    sid = next(csv.reader(io.StringIO(identity.stdout)))[1]
    if not sid.startswith("S-1-") or any(c not in "S0123456789-" for c in sid):
        raise OSError("Could not resolve current Windows user SID")
    inheritance = "OICI" if path.is_dir() else ""
    descriptor = f"D:P(A;{inheritance};FA;;;{sid})"
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = security.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.DWORD)]
    convert.restype = wintypes.BOOL
    apply = security.SetFileSecurityW
    apply.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    apply.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    pointer = ctypes.c_void_p()
    if not convert(descriptor, 1, ctypes.byref(pointer), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION.
        if not apply(str(path), 0x00000004 | 0x80000000, pointer):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(pointer)


def copy_file_permissions(source: Path, destination: Path) -> None:
    """Preserve access permissions before publishing replacement or backup bytes."""
    import stat

    mode = stat.S_IMODE(source.stat().st_mode)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        security = ctypes.WinDLL("advapi32", use_last_error=True)
        get_security = security.GetFileSecurityW
        get_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                                 wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        get_security.restype = wintypes.BOOL
        size = wintypes.DWORD()
        get_security(str(source), 0x00000004, None, 0, ctypes.byref(size))
        if not size.value:
            raise ctypes.WinError(ctypes.get_last_error())
        descriptor = ctypes.create_string_buffer(size.value)
        if not get_security(str(source), 0x00000004, descriptor, size.value, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        get_control = security.GetSecurityDescriptorControl
        get_control.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.WORD),
                               ctypes.POINTER(wintypes.DWORD)]
        get_control.restype = wintypes.BOOL
        control, revision = wintypes.WORD(), wintypes.DWORD()
        if not get_control(descriptor, ctypes.byref(control), ctypes.byref(revision)):
            raise ctypes.WinError(ctypes.get_last_error())
        apply = security.SetFileSecurityW
        apply.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
        apply.restype = wintypes.BOOL
        protection = 0x80000000 if control.value & 0x1000 else 0x20000000
        if not apply(str(destination), 0x00000004 | protection, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
    os.chmod(destination, mode)
