"""Shared workspace and sensitive-path policy; kept independent of tool execution."""
import os
from pathlib import Path

from .refusals import RefusalClass, refusal


def _validate_workspace_path(path: str | Path) -> Path | str:
    """Validate that path is inside the current workspace."""
    p = Path(path).resolve()
    cwd = Path.cwd().resolve()
    try:
        home = Path.home().resolve()
    except Exception:
        home = None

    # Agent tools must never edit their own grants, history, or configuration, even
    # when a project overlaps a state directory. Check resolved paths for aliases.
    state_dirs = [home / ".nexus-agent"] if home is not None else []
    if os.environ.get("NEXUS_AGENT_HOME"):
        state_dirs.append(Path(os.environ["NEXUS_AGENT_HOME"]))
    for state_dir in state_dirs:
        state_root = state_dir.resolve()
        if p == state_root or state_root in p.parents:
            return refusal(RefusalClass.SENSITIVE_PATH, "ERROR: Security Sandbox Access Denied: Nexus state directories are protected.")

    # Boundary safeguard: When running in home directory (Global Mode), prevent direct operations
    # on the home root itself or root dotfiles/configuration (e.g. ~/.bashrc, ~/.gitconfig).
    if home and cwd == home:
        if p == home:
            return refusal(RefusalClass.SENSITIVE_PATH,
                "ERROR: Security Sandbox Access Denied: Direct targeting of the root home directory is forbidden. "
                "Nexus is running in the home folder, the home root is protected, "
                "create a project folder (mkdir my-project, cd my-project) and relaunch."
            )
        if p.parent == home and p.name.startswith("."):
            return refusal(RefusalClass.SENSITIVE_PATH, f"ERROR: Security Sandbox Access Denied: Root configuration file '{p.name}' in home directory is protected.")

    try:
        if p.is_relative_to(cwd):
            return p
    except AttributeError:
        try:
            if os.path.commonpath([str(p), str(cwd)]) == str(cwd):
                return p
        except ValueError:
            pass
    return refusal(RefusalClass.SENSITIVE_PATH, f"ERROR: Security Sandbox Access Denied: Path '{path}' is outside the project workspace ({cwd}).")


_SENSITIVE_NAMES = frozenset({
    ".git", ".env", "config", "credentials", "secrets", ".npmrc", ".pypirc", "netrc",
    ".ssh", ".aws", ".azure", ".kube", ".gnupg", ".bashrc", ".zshrc", ".profile",
    ".bash_profile", "authorized_keys", "known_hosts", "id_rsa", "id_ed25519",
})
_SENSITIVE_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".crt", ".token", ".secret")
_SENSITIVE_MARKERS = ("credential", "secret", "token", "password", "passwd")


def _is_sensitive_path(p: Path) -> bool:
    """Check if a path points to sensitive credentials or configuration."""
    filename_lower = p.name.lower()
    return (
        filename_lower.startswith(".env")
        or filename_lower in _SENSITIVE_NAMES
        or any(part.lower() in _SENSITIVE_NAMES for part in p.parts)
        or any(filename_lower.endswith(ext) for ext in _SENSITIVE_SUFFIXES)
        or any(marker in filename_lower for marker in _SENSITIVE_MARKERS)
        or "id_rsa" in filename_lower
        or "id_ed25519" in filename_lower
    )
