"""Wire-view estimation boundary (providers.messages.serialized_wire_views).

Context cost must be estimated over what adapters actually serialize. Canonical
metadata (executor facts, refusal classes, mutation outcomes, claim records) is
session bookkeeping: openai_messages strips it, anthropic_messages and
gemini_messages never carry it. Charging it blocked real headroom the wire
never spent (run-4 recovery probe: a truthful minimal tool receipt was refused
while ~789 real tokens were free).
"""
import json

from nexus_agent_ai.agent import context
from nexus_agent_ai.providers.messages import (
    anthropic_messages, gemini_messages, openai_messages, wire_message,
)


def _conversation():
    return [
        {"role": "user", "content": "Fix the bug so multiply works."},
        {"role": "assistant", "content": "Reading the file first.",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": '{"path": "calc.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "def multiply(a, b):\n    return a + b\n",
         "metadata": {"executor": True, "executed": True,
                      "refusal_class": None,
                      "mutation_outcomes": [{"action": "read_file", "path": "/w/calc.py",
                                             "success": True, "status": "File read completed."}]}},
        {"role": "assistant", "content": "Now patching.",
         "tool_calls": [{"id": "c2", "type": "function",
                         "function": {"name": "patch_file",
                                      "arguments": '{"path": "calc.py", "target": "a + b", "replacement": "a * b"}'}}],
         "metadata": {"turn": "t1", "segment": "s1",
                      "mutation_outcomes": [{"action": "patch_file", "path": "/w/calc.py",
                                             "success": False, "status": "unchanged"}]}},
        {"role": "tool", "tool_call_id": "c2",
         "content": "ERROR: 'calc.py' changed on disk since you last read it.",
         "metadata": {"executor": True, "executed": False, "refusal_class": "stale_state"}},
    ]


def test_wire_message_keeps_only_provider_visible_fields():
    message = _conversation()[2]
    view = wire_message(message)
    assert set(view) == {"role", "content", "tool_call_id"}
    assert "metadata" not in view


def test_wire_message_keeps_tool_name_for_gemini_binding():
    # The Gemini function_response binding serializes the tool name; keeping it
    # can only over-estimate on adapters that drop it.
    message = {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "data",
               "metadata": {"executor": True}}
    assert wire_message(message)["name"] == "read_file"


def test_estimator_ignores_metadata_but_charges_content():
    base = _conversation()[2]
    twin = dict(base, metadata={"executor": True, "executed": True, "extra": "x" * 500})
    assert context.message_tokens(base) == context.message_tokens(twin)
    longer = dict(base, content=base["content"] + "y" * 400)
    assert context.message_tokens(longer) > context.message_tokens(base)


def test_estimator_matches_old_basis_when_no_metadata_present():
    # A plain text message has no adapter wrapper cheaper than the canonical
    # form, and no metadata to remove: the estimate equals the old basis.
    message = {"role": "user", "content": "plain text, no bookkeeping"}
    old_basis = context.estimate_tokens(json.dumps(message, ensure_ascii=False)) \
        + context.MESSAGE_OVERHEAD_TOKENS
    assert context.message_tokens(message) >= old_basis


def test_estimator_covers_every_adapter_view_by_construction():
    # The estimate is the maximum of the actual per-adapter serializations plus
    # the unchanged per-message overhead, so it can never sit below any
    # supported adapter's real wire cost - including Anthropic content-block
    # and Gemini parts wrappers, which the old canonical-JSON basis
    # UNDER-charged for metadata-free messages.
    messages = _conversation() + [
        {"role": "user", "content": "plain text, no bookkeeping"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": f"tool_{i}", "arguments": json.dumps({"path": f"src/f{i}.py"})}}
            for i in range(20)]},
        {"role": "user", "parts": [{"text": "gemini-shaped input"}]},
    ]
    for message in messages:
        views = context.serialized_wire_views(message)
        expected = max(context.estimate_tokens(v) for v in views.values()) \
            + context.MESSAGE_OVERHEAD_TOKENS
        assert context.message_tokens(message) == expected
        for view in views.values():
            assert context.message_tokens(message) >= \
                context.estimate_tokens(view) + context.MESSAGE_OVERHEAD_TOKENS


def test_multi_tool_call_wrapper_cost_tracks_actual_serialization():
    # No source-level cap bounds tool_calls per message and the iteration cap
    # does not bound N, so the per-call wrapper cost is measured from the real
    # transforms, not assumed: for N in 1..20 calls the estimate must stay at
    # or above every adapter's actual serialization, and grow with N.
    previous = 0
    for n in (1, 5, 12, 20):
        message = {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": f"tool_{i}",
                          "arguments": json.dumps({"path": f"src/f{i}.py", "mode": "fast"})}}
            for i in range(n)]}
        estimate = context.message_tokens(message)
        for transform in (openai_messages, anthropic_messages, gemini_messages):
            actual = context.estimate_tokens(
                json.dumps(transform([message]), ensure_ascii=False))
            assert estimate >= actual, (n, transform.__name__, estimate, actual)
        assert estimate > previous
        previous = estimate


def test_cross_message_gemini_name_binding_is_charged():
    # Stateful binding: a nameless tool result is serialized by Gemini as a
    # function_response carrying the name bound by the earlier assistant call.
    # Tool names in the real path are unbounded (model-emitted), so the
    # estimate must track the actual bound name at 100 and 1000 chars - no
    # single-message view or fixed constant can see this duplication.
    for name_len in (100, 1000):
        name = "n" * name_len
        conversation = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": name, "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result bytes"},
        ]
        estimate = context.history_tokens(conversation)
        actual = context.estimate_tokens(
            json.dumps(gemini_messages(conversation), ensure_ascii=False)) \
            + context.MESSAGE_OVERHEAD_TOKENS * len(conversation)
        assert estimate >= actual, (name_len, estimate, actual)
        # And the duplication is real: the bound name appears on the wire.
        serialized = json.dumps(gemini_messages(conversation))
        assert serialized.count(name) >= 2


def test_binding_is_ordered_per_result_not_final_state():
    # A later call reusing an id must NOT change an earlier result's applied
    # name: Gemini binds in conversation order, so the earlier result still
    # carries the 1000-char name even though the id ends bound to 's'.
    long_name = "n" * 1000
    conversation = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": long_name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result bytes"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "s", "arguments": "{}"}}]},
    ]
    estimate = context.history_tokens(conversation)
    actual = context.estimate_tokens(
        json.dumps(gemini_messages(conversation), ensure_ascii=False)) \
        + context.MESSAGE_OVERHEAD_TOKENS * len(conversation)
    assert estimate >= actual, (estimate, actual)
    assert json.dumps(gemini_messages(conversation)).count(long_name) >= 2


def test_bound_name_takes_precedence_over_explicit_result_name():
    # Gemini prefers the bound name over the result's own 'name' field, so an
    # explicit short name must not suppress the long bound name's surcharge.
    long_name = "n" * 1000
    conversation = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": long_name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "s", "content": "result bytes"},
    ]
    estimate = context.history_tokens(conversation)
    actual = context.estimate_tokens(
        json.dumps(gemini_messages(conversation), ensure_ascii=False)) \
        + context.MESSAGE_OVERHEAD_TOKENS * len(conversation)
    assert estimate >= actual, (estimate, actual)


def test_adversarial_name_characters_are_charged_escaped():
    # The wire carries the JSON-escaped name; charging the raw string
    # under-counts quotes, backslashes and newlines (decode_native_tool_call
    # validates only a non-empty str - no character bound). The estimate must
    # cover the actual escaped serialization for adversarial names.
    for name in ('"' * 1000, "\\" * 1000, "\n" * 500, 'q"u\\o\nte'):
        conversation = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": name, "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result bytes"},
        ]
        estimate = context.history_tokens(conversation)
        actual = context.estimate_tokens(
            json.dumps(gemini_messages(conversation), ensure_ascii=False)) \
            + context.MESSAGE_OVERHEAD_TOKENS * len(conversation)
        assert estimate >= actual, (repr(name[:20]), estimate, actual)


def test_legacy_parts_function_call_binds_before_charging():
    # Owner-reproduced gap: the adapter NORMALIZES legacy native shapes before
    # binding; a raw pass over message["tool_calls"] misses a name embedded in
    # parts. assistant parts:[{function_call: {id: c1, name: x*1000, args:{}}}]
    # + nameless result c1 estimated 363 vs 639 actual before the fix.
    name = "x" * 1000
    conversation = [
        {"role": "assistant", "content": "",
         "parts": [{"function_call": {"id": "c1", "name": name, "args": {}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
    ]
    estimate = context.history_tokens(conversation)
    actual = context.estimate_tokens(
        json.dumps(gemini_messages(conversation), ensure_ascii=False)) \
        + context.MESSAGE_OVERHEAD_TOKENS * len(conversation)
    assert estimate >= actual, (estimate, actual)
    assert json.dumps(gemini_messages(conversation)).count(name) >= 2


def test_legacy_functioncall_camelcase_binds_before_charging():
    name = "y" * 500
    conversation = [
        {"role": "assistant", "content": "",
         "parts": [{"functionCall": {"id": "c1", "name": name, "args": {}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
    ]
    estimate = context.history_tokens(conversation)
    actual = context.estimate_tokens(
        json.dumps(gemini_messages(conversation), ensure_ascii=False)) \
        + context.MESSAGE_OVERHEAD_TOKENS * len(conversation)
    assert estimate >= actual, (estimate, actual)


def test_anthropic_tool_use_block_binds_before_charging():
    name = "z" * 400
    conversation = [
        {"role": "assistant", "content": "",
         "parts": [{"type": "tool_use", "id": "c1", "name": name, "input": {}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
    ]
    estimate = context.history_tokens(conversation)
    actual = context.estimate_tokens(
        json.dumps(gemini_messages(conversation), ensure_ascii=False)) \
        + context.MESSAGE_OVERHEAD_TOKENS * len(conversation)
    assert estimate >= actual, (estimate, actual)


def test_pending_empty_id_matched_by_name_before_charging():
    # normalize_messages binds a nameless canonical result to a pending
    # empty-id call by name; the raw pass used to miss it entirely.
    name = "p" * 600
    conversation = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "", "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "name": name, "content": "x"},
    ]
    estimate = context.history_tokens(conversation)
    actual = context.estimate_tokens(
        json.dumps(gemini_messages(conversation), ensure_ascii=False)) \
        + context.MESSAGE_OVERHEAD_TOKENS * len(conversation)
    assert estimate >= actual, (estimate, actual)


def test_compound_native_message_maps_surcharge_to_original():
    # One native message carrying BOTH a call and a result expands to several
    # canonical messages; the result's surcharge must attach to the original.
    name = "n" * 300
    conversation = [
        {"role": "user", "parts": [
            {"function_call": {"id": "c1", "name": name, "args": {}}},
            {"function_response": {"id": "c2", "name": name,
                                   "response": {"result": "earlier"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result bytes"},
    ]
    estimate = context.history_tokens(conversation)
    actual = context.estimate_tokens(
        json.dumps(gemini_messages(conversation), ensure_ascii=False)) \
        + context.MESSAGE_OVERHEAD_TOKENS * len(conversation)
    assert estimate >= actual, (estimate, actual)


def test_slice_cut_direction_is_conservative():
    # Splitting a conversation where a result's binding lives in the other
    # half can only make the parts sum to LESS than the whole (the dropped
    # binding's surcharge stays with the retained total), so the fit's
    # subtractive slice arithmetic over-retains and never under-retains.
    name = "n" * 200
    call = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}]}]
    result = [{"role": "tool", "tool_call_id": "c1", "content": "result bytes"}]
    whole = context.history_tokens(call + result)
    parts = context.history_tokens(call) + context.history_tokens(result)
    assert parts <= whole
    assert whole - parts == context.estimate_tokens(json.dumps(name))  # the dropped surcharge


def test_estimator_never_exceeds_old_basis():
    # The correction removes only provably-unsent bookkeeping; it can never
    # charge MORE than the previous canonical-JSON basis for the same history.
    messages = _conversation()
    old_total = sum(
        context.estimate_tokens(json.dumps(m, default=str, ensure_ascii=False))
        + context.MESSAGE_OVERHEAD_TOKENS
        for m in messages
    )
    assert context.history_tokens(messages) <= old_total
    assert context.history_tokens(messages) < old_total  # metadata present above


def test_wire_view_covers_every_adapter_serialization():
    # For each adapter, every serialized message field must come from the
    # wire_message field set (adapters add only bounded structural wrappers).
    messages = _conversation()
    allowed = set(wire_message({k: None for k in
                                ("role", "content", "tool_calls", "tool_call_id", "name", "parts")}))
    for view in openai_messages(messages):
        assert set(view) <= allowed
    for view in anthropic_messages(messages):
        assert set(view) <= {"role", "content"}
    for view in gemini_messages(messages):
        assert set(view) <= {"role", "parts"}


def test_recovery_injection_suffix_is_charged():
    # The designed edit-recovery snapshot is appended to the refusal result in
    # the provider-bound view; it is real wire content and must stay charged.
    messages = _conversation()
    before = context.history_tokens(messages)
    injected = json.loads(json.dumps(messages))
    injected[4]["content"] += "\n\n--- BEGIN HARNESS EDIT RECOVERY ---\nfresh snapshot\n--- END ---"
    assert context.history_tokens(injected) > before
