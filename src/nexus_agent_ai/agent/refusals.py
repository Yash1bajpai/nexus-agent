"""Source-classified tool refusals, retaining the public string contract."""
from enum import Enum


class RefusalClass(str, Enum):
    TARGETING = "targeting"
    SYNTAX = "syntax"
    CONTENT_LOSS = "content_loss"
    AMBIGUITY = "ambiguity"
    BLIND_EDIT = "blind_edit"
    STALE_STATE = "stale_state"
    SENSITIVE_PATH = "sensitive_path"
    NOT_FOUND = "not_found"
    PERMISSION = "permission"
    OTHER = "other"


class ToolResult(str):
    """Public tool text with a private, source-authored execution receipt."""

    def __new__(cls, message: str, *, executed: bool, outcome: str):
        instance = super().__new__(cls, message)
        instance.executed = bool(executed)
        instance.outcome = outcome
        return instance


class RefusalText(ToolResult):
    """A string whose classification survives tool/guard hint wrapping."""

    def __new__(cls, message: str, refusal_class: RefusalClass):
        instance = super().__new__(
            cls, message, executed=False, outcome="Not executed."
        )
        instance.refusal_class = RefusalClass(refusal_class)
        return instance


def refusal(kind: RefusalClass, message: str) -> RefusalText:
    return RefusalText(message, kind)


def execution_result(text: str, *, executed: bool, outcome: str) -> ToolResult:
    """Attach a trusted receipt without losing refusal or recovery metadata.

    Outcomes must be bounded status facts authored by the handler, never output
    excerpts. Constructing through str avoids imposing a subclass constructor's
    private signature and preserves existing string-compatible return types.
    """
    result_type = type(text) if isinstance(text, str) and type(text) is not str else ToolResult
    result = str.__new__(result_type, text)
    result.__dict__.update(getattr(text, "__dict__", {}))
    result.executed = bool(executed)
    result.outcome = outcome
    return result


def execution_ran(result: str) -> bool:
    """Prefer receipts; fail conservatively for legacy refusal/suppression text."""
    executed = getattr(result, "executed", None)
    if isinstance(executed, bool):
        return executed
    if isinstance(result, RefusalText):
        return False
    text = str(result).lstrip().casefold()
    return not text.startswith((
        "error", "sandbox", "[system", "not executed", "suppressed",
        "suppression", "[suppressed", "tool call suppressed",
    ))
