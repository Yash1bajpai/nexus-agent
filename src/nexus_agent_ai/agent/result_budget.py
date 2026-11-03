"""Exact-fit tool output admission with bounded, source-attributed receipts.

Only the existing same-session artifact store may retain omitted output. Fit
probes never change canonical history, and an omission never guesses an outcome
from stdout or turns a completed action into a failed action.
"""
from __future__ import annotations

import hashlib

from .context import ContextBudgetError, estimate_tokens
from .refusals import (
    RefusalClass, ToolResult, execution_ran, execution_result, replace_result_text,
)
from . import budget_diagnostics as diagnostics


def _replacement(source, text, *, executed, outcome):
    """Replace string content without dropping refusal or private guard metadata."""
    replacement = replace_result_text(source, text)
    return execution_result(replacement, executed=executed, outcome=outcome)


@diagnostics.capture_receipt
def guard_tool_result(name, result, *, memory, fits):
    """Return an exactly fitting result or raise without misreporting execution.

    ``fits`` probes the complete prospective tool-result content, including the
    receipt and omission metadata. A protected artifact is admitted only after
    its complete reference fits. Capacity refusal, missing persistence and store
    errors all use a truthful flat discarded-output receipt instead.
    """
    if fits(result):
        return result

    executed = execution_ran(result)
    outcome = getattr(result, "outcome", "unknown")
    # Outcomes originate at execution sites, never in raw output. Reject an
    # unbounded or malformed receipt rather than quote/truncate arbitrary prose.
    if (not isinstance(outcome, str) or not outcome.strip() or len(outcome) > 256
            or any(ord(char) < 32 for char in outcome)):
        outcome = "unknown"
    if outcome.lower() == "unknown":
        status = "Executed; outcome unknown." if executed else "Not executed; outcome unknown."
    else:
        status = f"Outcome: {outcome}"
    content = str(result)
    omitted = estimate_tokens(content)

    def replacement(text):
        return _replacement(result, text, executed=executed, outcome=outcome)

    retain = getattr(memory, "retain_tool_output", None)
    discard_reason = "artifact store unavailable; output discarded"
    if callable(retain):
        discard_reason = "artifact receipt cannot fit the context budget; output discarded"
        identity = hashlib.sha256((name + "\0" + content).encode("utf-8")).hexdigest()
        retained = replacement(
            f"{status}\n~{omitted} tokens omitted from context; retained as artifact {identity}."
        )
        if fits(retained):
            plan = {"artifact_id": identity, "tool_name": name,
                    "byte_len": len(content.encode("utf-8")), "content": content}
            try:
                admitted = retain(plan) is True
                diagnostics.receipt_retention("success" if admitted else "full")
                discard_reason = "artifact store full; output discarded"
            except Exception:
                admitted = False
                diagnostics.receipt_retention("error")
                discard_reason = "artifact retention failed; output discarded"
            if admitted:
                return retained

    for note in (
        f"{status}\n~{omitted} tokens omitted from context; {discard_reason}.",
        f"{status}\nOutput discarded.",
    ):
        discarded = replacement(note)
        if fits(discarded):
            return discarded
    error = ContextBudgetError(
        "The tool's truthful outcome and output-omission receipt cannot fit the current context. "
        "Execution status is preserved; no over-budget request was sent."
    )
    error.refusal_class = RefusalClass.CAPACITY
    if isinstance(result, ToolResult):
        error.executed = result.executed
        error.outcome = result.outcome
    raise error
