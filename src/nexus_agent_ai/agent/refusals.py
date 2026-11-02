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


class RefusalText(str):
    """A string whose classification survives tool/guard hint wrapping."""

    def __new__(cls, message: str, refusal_class: RefusalClass):
        instance = super().__new__(cls, message)
        instance.refusal_class = RefusalClass(refusal_class)
        return instance


def refusal(kind: RefusalClass, message: str) -> RefusalText:
    return RefusalText(message, kind)
