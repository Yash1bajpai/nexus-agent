"""Source-classified tool refusals, retaining the public string contract."""
from enum import Enum
import unicodedata


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
    CAPACITY = "capacity"
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


def replace_result_text(source: str, text: str) -> str:
    """Clone private source metadata while replacing only public string content."""
    if isinstance(source, str) and hasattr(source, "__dict__"):
        replacement = str.__new__(type(source), text)
        replacement.__dict__.update(vars(source))
        return replacement
    return text


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


def execution_metadata(result: str) -> dict:
    """Export only bounded source-authored receipt facts, never output or hints."""
    if not isinstance(result, ToolResult):
        return {}
    metadata = {}
    executed = getattr(result, "executed", None)
    if isinstance(executed, bool):
        metadata["executed"] = executed
    outcome = getattr(result, "outcome", None)
    if (
        isinstance(outcome, str) and len(outcome) <= 256 and outcome.strip()
        and not any(unicodedata.category(char) == "Cc" for char in outcome)
    ):
        metadata["outcome"] = str(outcome)
    kind = getattr(result, "refusal_class", None)
    if isinstance(kind, RefusalClass):
        metadata["refusal_class"] = kind.value
    return metadata


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
