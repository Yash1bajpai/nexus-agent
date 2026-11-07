"""Focused deterministic rejected-patch recovery probe (experiment-side only).

No sealed-protocol execution: this probe replicates run_driver's verification
chain (baseline commit/tree/dirty checks, pinned harness sha, runner sha, native
gate evidence, installed-wheel provenance equality, pinned model download with
digest/size enforcement) and then drives ONE synthetic recovery session instead
of the 14-row protocol. The sealed harness module is byte-identical and only its
public primitives (PtySession, CONTROLLER_CODE, child_environment, evidence cap
helpers) are reused; PROTOCOL_TASKS is never modified or executed.

What is measured: whether the model repairs ONE authentic patch_file exact-byte
refusal (real tool, real current file bytes returned) within the unchanged
10-iteration turn budget and 420s row timeout.

How the refusal is induced (INTENDED, not guaranteed - see outcome taxonomy):
the fixture's buggy multiply return line carries an unguessable random nonce
comment, so a blind target that includes that full old line cannot match. A
narrower partial target that omits the marker text can still match, so the
induction is a strong bias, never a certainty - the scorer records what
actually happened rather than assuming it. The session prompt names NO file and
carries NO attached fixture content: verified against the sealed app source that
the only two attachment routes (parse_at_mentions via '@', and attach_named_files
via a bare existing-filename token under edit intent) both require a filename
token this prompt does not contain, and the system prompt injects only the
working directory path (Global Mode, no listing). A content-derived target
therefore requires a read_file observation, and if the model reads the fixture
first, a watcher thread
rewrites the nonce on that same line once after the read observation is
rendered (model decode takes seconds, so the mutation lands first in practice),
so a read-derived target then mismatches the real current bytes. Either way the
REAL tool issues the refusal - no fake tool output, no app/wheel/harness change.
A blind substring guess (e.g. 'return a + b' without the marker) can still
match without any read; that remains an honest NO_REFUSAL_FIRST_PATCH_ACCEPTED,
never recovery.
Line endings are NOT a mechanism: the tool normalizes actual LF targets to CRLF
(agent/tools.py), so CRLF-vs-LF alone never refuses. The observed run-1 refusal
came from the model emitting literal backslash-n bytes (double-escaped JSON) in
its target; the tool's refusal returned the exact correct target as JSON, and
the turn ended when the app's historical-omission coverage gap blocked the
retry request - the model never attempted the repair. This probe isolates the
repair step in a fresh small session so the model has the full unchanged budget
for it. That is necessary-but-not-sufficient evidence: it does NOT exercise the
historical-omission guard under long real sessions (the thing that actually
ended run 1's turn), which is a separate out-of-scope question. Render/poll
races or an unusually narrow patch target can still produce NO refusal; that is
recorded honestly as NO_REFUSAL_FIRST_PATCH_ACCEPTED and never as recovery.

Scoring is canonical, never terminal prose: the persisted session history
(history.db via --persist) yields the ordered tool calls (assistant tool_calls)
and their results (tool message metadata: executed, refusal_class,
mutation_outcomes path/action/success). Joins are mandatory: every scored tool
receipt must bind to exactly one unique assistant call by id, receipt action
must equal the call name, message_json must parse, and mutation paths resolve
(relative paths against the session workspace, Windows dot-segments collapsed)
to EXACTLY this session's fixture path; any missing, duplicate, malformed or
mismatched join is CANONICAL_SCORING_UNAVAILABLE, never a silent skip.
RECOVERY_OBSERVED requires an authentic
targeting refusal on the fixture FOLLOWED BY an accepted patch_file mutation on
the same fixture, in that order, in the same single turn, plus static
final-bytes checks. Rendered 'Successfully patched' text is recorded only as a
cross-check and can never produce RECOVERY_OBSERVED by itself.

Candidate seal: the workflow and this probe both fail closed unless the pushed
experiment checkout matches the reviewed candidate - HEAD~1 commit and tree
equal the reviewed content commit/tree pinned in
github/experiments/candidate-seal.json, the seal commit touches only that
file, and the probe/workflow file bytes equal the pinned hashes.

Exit semantics: honest model observations (any taxonomy outcome from
RECOVERY_OBSERVED through NO_PATCH_ATTEMPT) exit 0; harness errors, scoring
unavailability, and any required-evidence loss (transcript, row, model digest,
log capture) exit 1 after bounded diagnostics are persisted. A green workflow
means complete evidence, never that recovery was observed.

Static-only: no model-written file is executed or imported. Assessment uses
final bytes sha and a narrow AST contract against the known fixture shape
(unique top-level add/multiply defs; exact arg names (a, b); no decorators,
defaults, or extra body statements; return operands exactly a and b with the
expected operator); plain substrings are recorded as corroborating
markers only, never as fix proof. The recovery OUTCOME never
changes the run's exit code, any axis, or the seal; seal-chain or infrastructure
failures before the session still fail closed nonzero, exactly as prior probes.

Not a default switch, release, or native/core requalification. Any failed axis
stays failed.
"""
import argparse
import hashlib
import importlib.util
import json
import ntpath
import os
from pathlib import Path
import re
import secrets
import subprocess
import threading
import sys
import time
import urllib.request

BASE_COMMIT = 'f803d5d16add2edb0963b181ee530e75cbcf91d8'
BASE_TREE = '7f371e3d8acf46c6091527cad19b02faed68ef0a'
MODEL_REPO = 'unsloth/Qwen3-4B-Instruct-2507-GGUF'
MODEL_FILE = 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
MODEL_REVISION = 'a06e946bb6b655725eafa393f4a9745d460374c9'
MODEL_SHA = '3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597'
MODEL_SIZE = 2497281120
HARNESS_SHA = 'a96392bef56afc1ff8faf1fbc89648364c5e2fc2d7f12ba657d5d4ca0ba53da0'
MODEL_URL = f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}'

RECOVERY_FIXTURE = 'recover_calc.py'
RECOVERY_ROW_TIMEOUT_S = 420.0
MUTATION_TOOLS = ('patch_file', 'write_file')
DEADLINE_RESERVE_S = 10 * 60
ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r')

OBSERVATION_OUTCOMES = frozenset({
    'RECOVERY_OBSERVED', 'RECOVERY_ACCEPTED_BUT_STATIC_MISMATCH',
    'RECOVERY_VIA_WRITE_FILE', 'WRITE_AFTER_REFUSAL_STATIC_MISMATCH',
    'NO_REFUSAL_FIRST_PATCH_ACCEPTED', 'MODEL_USED_WRITE_FILE_NO_REFUSAL',
    'RECOVERY_NOT_OBSERVED_TURN_ENDED', 'NO_PATCH_ATTEMPT'})


def evidence_complete(rec):
    """Green workflow = complete evidence, never that recovery was observed."""
    required = ('initial_sha256', 'final_sha256', 'canonical_events', 'outcome')
    return bool(
        rec is not None
        and rec.get('outcome') in OBSERVATION_OUTCOMES
        and 'canonical_scoring_error' in rec
        and rec.get('canonical_scoring_error') is None
        and not rec.get('evidence_errors')
        and all(rec.get(key) is not None for key in required))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def strip_ansi(text):
    return ANSI_RE.sub('', text)


class PeakMemorySampler:
    """Telemetry-only peak RSS sampler for llama-server processes under the
    experiment root. Never alters the run: read-only psutil polls on a daemon
    thread, written once in the finally block. A sampling failure records an
    error string, never a changed outcome."""

    def __init__(self, root):
        self._root = Path(root).resolve()
        self._stop = threading.Event()
        self.samples = 0
        self.peak = {}
        self.error = None

    def _poll_once(self):
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                exe = proc.info.get('exe') or ''
                if 'llama-server' not in (proc.info.get('name') or '') or not exe:
                    continue
                if not Path(exe).resolve().is_relative_to(self._root):
                    continue
                rss = proc.memory_info().rss
                key = str(proc.info['pid'])
                if rss > self.peak.get(key, 0):
                    self.peak[key] = rss
                self.samples += 1
            except Exception:
                continue

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                self.error = type(exc).__name__
            self._stop.wait(2.0)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        return False


def download_pinned_model(home, end, root):
    """Same bounded, digest-enforced download as the accepted probe's env wrapper."""
    target = Path(home) / Path('.nexus-agent/models/unsloth--Qwen3-4B-Instruct-2507-GGUF') / MODEL_FILE
    if target.exists():
        raise RuntimeError('Model destination not fresh')
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    count = 0
    h = hashlib.sha256()
    start = time.monotonic()
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, part.open('xb') as f:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                if time.time() >= end - 600 or time.monotonic() - start > 1200:
                    raise TimeoutError('Bounded model download exhausted; no retry')
                count += len(block)
                if count > MODEL_SIZE:
                    raise RuntimeError('Model exceeds pinned bytes')
                h.update(block)
                f.write(block)
        if count != MODEL_SIZE or h.hexdigest() != MODEL_SHA:
            raise RuntimeError('Pinned model digest/size mismatch')
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    if sha(target) != MODEL_SHA:
        raise RuntimeError('Pinned model post-write digest mismatch')
    (root / 'output/model-experiment.json').write_text(json.dumps({
        'kind': 'ISOLATED_MODEL_EXPERIMENT_NOT_RELEASE_QUALIFICATION',
        'baseline_commit': BASE_COMMIT, 'baseline_tree': BASE_TREE,
        'harness_sha256': HARNESS_SHA, 'model_repo': MODEL_REPO,
        'model_revision': MODEL_REVISION, 'model_filename': MODEL_FILE,
        'model_sha256': MODEL_SHA, 'model_bytes': count,
        'download_seconds': time.monotonic() - start,
        'post_write_digest_verified': True,
        'default_changed': False, 'app_source_changed': False,
        'runtime': 'existing llama.cpp b7075, compatibility unproven until observed'}, indent=2), encoding='utf-8')


def verify_candidate_seal(experiment_root):
    """Fail-closed check that the pushed experiment checkout is exactly the
    reviewed candidate: HEAD~1 is the reviewed content commit/tree, the seal
    commit on top touches only candidate-seal.json, and the probe/workflow file
    bytes equal the pinned hashes. The seal file exists only in the tip commit,
    so its pins can name the content commit without self-reference."""
    seal = json.loads((experiment_root / '.github/experiments/candidate-seal.json')
                      .read_text(encoding='utf-8'))

    def git(*git_args):
        return subprocess.check_output(['git', '-C', str(experiment_root), *git_args],
                                       text=True).strip()

    if git('rev-parse', 'HEAD~1') != seal['c1_commit']:
        raise RuntimeError('Candidate content commit differs from reviewed seal')
    if git('rev-parse', 'HEAD~1^{tree}') != seal['c1_tree']:
        raise RuntimeError('Candidate content tree differs from reviewed seal')
    changed = git('diff', '--name-only', 'HEAD~1', 'HEAD').splitlines()
    if changed != ['.github/experiments/candidate-seal.json']:
        raise RuntimeError('Candidate seal commit modifies more than the seal file')
    if git('status', '--porcelain'):
        raise RuntimeError('Candidate experiment checkout is dirty')
    if sha(experiment_root / '.github/experiments/qwen_recovery_probe.py') != seal['probe_sha256']:
        raise RuntimeError('Candidate probe bytes differ from reviewed seal')
    if sha(experiment_root / '.github/workflows/smollm3-probe.yml') != seal['workflow_sha256']:
        raise RuntimeError('Candidate workflow bytes differ from reviewed seal')
    return seal


def verify_seal_chain(core, source, root, wheel, native_evidence, candidate_seal=None):
    """run_driver's verification chain, replicated without its protocol execution."""
    runner_path = source / '.github/qualification/runner.py'
    if sha(runner_path) != core.RUNNER_SHA256:
        raise RuntimeError('Runner hash mismatch')
    verifier = core.load_module(source / '.github/qualification/verify_windows_interactive.py',
                                'verify_windows_interactive')
    native_gate = verifier.check_native_evidence(native_evidence, commit=BASE_COMMIT,
                                                 tree=BASE_TREE, wheel=wheel)
    runner_mod = core.load_module(runner_path, 'q1_runner')
    installed = runner_mod.verify_installed(wheel, source, BASE_COMMIT)
    if installed.get('commit') != BASE_COMMIT:
        raise RuntimeError('Installed runtime commit differs from approved commit seal')
    if installed.get('tree') != BASE_TREE:
        raise RuntimeError('Installed runtime tree differs from approved tree seal')
    if installed['package_py'] != native_gate['provenance']['package_py']:
        raise RuntimeError('Installed runtime package bytes differ from native gate package_py')
    if installed['wheel_sha256'] != native_gate.get('wheel_sha256'):
        raise RuntimeError('Installed wheel SHA256 differs from native gate wheel_sha256')
    (root / 'output/seal-chain.json').write_text(json.dumps({
        'kind': 'SEAL_CHAIN_REPLICATED_NO_PROTOCOL_EXECUTION',
        'baseline_commit': BASE_COMMIT, 'baseline_tree': BASE_TREE,
        'harness_sha256': HARNESS_SHA, 'runner_sha256': core.RUNNER_SHA256,
        'native_gate_success': bool(native_gate.get('success')),
        'wheel_sha256': installed['wheel_sha256'],
        'candidate_seal': candidate_seal}, indent=2), encoding='utf-8')


def run_recovery_session(core, root, env, deadline_end, output_dir, byte_tracker):
    """Drive one synthetic recovery session; static-only evidence, never a gate."""
    rec = {
        'kind': 'HARNESS_INDUCED_REFUSAL_RECOVERY_STATIC_ONLY',
        'meaning': ('Whether the model repairs one authentic patch_file exact-byte refusal '
                    '(real tool, real current file bytes returned) within the unchanged '
                    '10-iteration turn budget and 420s row timeout. The refusal is induced by '
                    'the harness (unguessable nonce fixture, plus at most one post-read '
                    'mutation of the exact buggy line; line endings are NOT a mechanism - '
                    'the tool normalizes real LF to CRLF) - INTENDED, not guaranteed; it '
                    'is NOT '
                    'a natural model failure and says nothing by itself about natural patch '
                    'accuracy. refusal_path records which induction path, if any, actually '
                    'produced the observed refusal. Static-only: nothing model-written is '
                    'executed or imported.'),
        'budgets_unchanged': {'max_iterations': 10, 'row_timeout_s': RECOVERY_ROW_TIMEOUT_S},
        'tool_contract_unchanged': True,
        'outcome': 'HARNESS_ERROR',
        'evidence_errors': [],
    }
    pty = None
    transcript = ['']
    try:
        if time.time() >= deadline_end - DEADLINE_RESERVE_S:
            rec['outcome'] = 'SKIPPED_DEADLINE_RESERVE'
            return rec
        ws = root / 'recovery-workspace'
        if ws.exists():
            raise RuntimeError('Recovery workspace not fresh')
        ws.mkdir()
        nonce = secrets.token_hex(8)
        fixture = ws / RECOVERY_FIXTURE
        marker = f'# BUG-MARKER {nonce}'
        initial = ('def add(a, b):\r\n    return a + b\r\n\r\n'
                   'def multiply(a, b):\r\n    return a + b  ' + marker + '\r\n')
        fixture.write_bytes(initial.encode('utf-8'))
        rec['initial_sha256'] = sha(fixture)
        env2 = dict(env)
        env2['NEXUS_E2E_WORKDIR'] = str(ws)
        argv = [sys.executable, '-I', '-X', 'utf8', '-c', core.CONTROLLER_CODE,
                'repl', '--provider', 'local', '--context-size', '4096',
                '--max-output-tokens', '512', '--max-iterations', '10',
                '--persist', '--session', 'experiment-recovery-001', '--no-stream']
        pty = core.PtySession(argv, cwd=ws, env=env2, root=root, byte_tracker=byte_tracker)
        if pty.error_flag:
            raise RuntimeError(f'session start failed: {pty.error_flag}')
        init_out, ready, consent = pty.wait_for_ready_prompt(
            timeout=min(600.0, max(0.0, deadline_end - time.time() - 60)), deadline=deadline_end)
        transcript[0] += init_out
        if not ready:
            rec['outcome'] = 'STARTUP_NOT_READY'
            rec['consent_wait_at_startup'] = consent
            return rec

        state = {'mutated': False, 'saw_read_observe': False, 'saw_patch_action': False,
                 'mutation_error': None}

        def watcher():
            read_re = re.compile(r'read_file\([^)]*' + re.escape(RECOVERY_FIXTURE))
            while not pty.stop_event.is_set():
                try:
                    text = strip_ansi(pty.all_output)
                    m = read_re.search(text)
                    if m and '[OBSERVE]' in text[m.end():]:
                        state['saw_read_observe'] = True
                    if re.search(r'patch_file\([^)]*' + re.escape(RECOVERY_FIXTURE), text):
                        state['saw_patch_action'] = True
                    if state['saw_read_observe'] and not state['mutated']:
                        # Rewrite the nonce on the buggy line: a read-derived target
                        # that includes that full old line then mismatches the real
                        # current bytes. Partial targets can still match; the scorer
                        # records what happened. CRLF is realism only (the tool
                        # normalizes real LF targets).
                        current = fixture.read_bytes()
                        mutated_bytes = current.replace(
                            marker.encode('utf-8'),
                            f'# BUG-MARKER {secrets.token_hex(8)}'.encode('utf-8'), 1)
                        if mutated_bytes == current:
                            state['mutation_error'] = 'marker not found'
                            return
                        fixture.write_bytes(mutated_bytes)
                        state['mutated'] = True
                        return
                    if state['saw_patch_action'] and not state['saw_read_observe']:
                        return  # blind patch: nonce + CRLF content is expected to mismatch
                except Exception as exc:  # never disturb the session over telemetry
                    state['mutation_error'] = type(exc).__name__
                    return
                pty.stop_event.wait(0.2)

        watch = threading.Thread(target=watcher, daemon=True)
        watch.start()
        # The prompt names no file and no filename token, so the app's two
        # context-attachment routes (at-mentions and bare-name auto-attach)
        # cannot fire: a content-derived patch target requires a tool read,
        # which arms the watcher. Recorded for evidence.
        prompt = ('Find and fix the bug in the Python file in the current directory '
                  'so multiply works correctly. Preserve add and all unrelated code.')
        rec['prompt_text'] = prompt
        rec['prompt_carries_fixture_content'] = False
        remaining = max(0.0, deadline_end - time.time() - 60)
        pty.send_input(prompt + '\r')
        raw, ready, consent_wait = pty.wait_for_ready_prompt(
            timeout=min(RECOVERY_ROW_TIMEOUT_S, remaining), deadline=deadline_end)
        transcript[0] += raw
        watch.join(timeout=5)

        clean = strip_ansi(raw)
        rec['refusal_path'] = ('post_read_buggy_line_mutation' if state['mutated']
                               else 'blind_patch_unguessable_content' if state['saw_patch_action']
                               else 'none_observed')
        rec['refusal_path_is_observation_not_certainty'] = True  # watcher may lose the render race
        rec['mutation_error'] = state['mutation_error']
        rec['turn_returned_to_prompt'] = bool(ready)
        rec['consent_wait'] = bool(consent_wait)
        rec['fixture_mutated_by_harness'] = state['mutated']
        # Terminal prose is a cross-check only; it can never produce RECOVERY_OBSERVED.
        rec['terminal_patch_markers'] = max(
            len(re.findall(r'\[ACTION\]\s*patch_file\([^)]*' + re.escape(RECOVERY_FIXTURE), clean)),
            len(re.findall(r'Running tool:\s*patch_file\([^)]*' + re.escape(RECOVERY_FIXTURE), clean)))
        rec['terminal_accept_prose_count'] = clean.count('Successfully patched')

        final_bytes = fixture.read_bytes()
        rec['final_sha256'] = sha(fixture)
        rec['final_bytes'] = len(final_bytes)
        import ast
        try:
            tree = ast.parse(final_bytes.decode('utf-8'))
            rec['static_syntax_ok'] = True
            # Narrow AST contract against the known fixture shape. An operator-only
            # check would pass 'return 7 + 9', wrong operands, duplicate defs,
            # decorators, defaults, or extra body statements, so the contract
            # requires exactly the expected structure. Still static-only: the
            # file is never executed, so this is shape proof, not behavior proof.
            contract = {'top_level_other_nodes': 0, 'duplicate_defs': [], 'ok': True}

            def check_func(node, expected_op):
                detail = {'args_ok': False, 'body_ok': False, 'return_op': None,
                          'operands_ok': False, 'no_decorators': not node.decorator_list,
                          'no_defaults': not node.args.defaults and not node.args.kw_defaults,
                          'no_varargs': node.args.vararg is None and node.args.kwarg is None
                          and not node.args.kwonlyargs and not node.args.posonlyargs}
                detail['args_ok'] = [a.arg for a in node.args.args] == ['a', 'b']
                detail['body_ok'] = len(node.body) == 1 and isinstance(node.body[0], ast.Return)
                if detail['body_ok'] and isinstance(node.body[0].value, ast.BinOp):
                    val = node.body[0].value
                    detail['return_op'] = type(val.op).__name__
                    detail['operands_ok'] = (isinstance(val.left, ast.Name) and val.left.id == 'a'
                                             and isinstance(val.right, ast.Name) and val.right.id == 'b')
                detail['ok'] = all([detail['args_ok'], detail['body_ok'], detail['no_decorators'],
                                    detail['no_defaults'], detail['no_varargs'],
                                    detail['return_op'] == expected_op, detail['operands_ok']])
                return detail

            seen = {}
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in ('add', 'multiply'):
                    if node.name in seen:
                        contract['duplicate_defs'].append(node.name)
                    seen.setdefault(node.name, node)
                else:
                    contract['top_level_other_nodes'] += 1
            contract['add'] = check_func(seen['add'], 'Add') if 'add' in seen else None
            contract['multiply'] = check_func(seen['multiply'], 'Mult') if 'multiply' in seen else None
            contract['ok'] = (not contract['duplicate_defs']
                              and contract['top_level_other_nodes'] == 0
                              and contract['add'] is not None and contract['add']['ok']
                              and contract['multiply'] is not None and contract['multiply']['ok'])
            rec['static_fixture_contract'] = contract
            if contract['add']:
                rec['static_ast_add_return_op'] = contract['add']['return_op']
            if contract['multiply']:
                rec['static_ast_multiply_return_op'] = contract['multiply']['return_op']
        except SyntaxError:
            rec['static_syntax_ok'] = False
        text = final_bytes.decode('utf-8', errors='replace')
        # Substring markers only - corroborating signals, never fix proof.
        rec['marker_add_substring_present'] = 'def add(a, b):' in text and 'return a + b' in text
        rec['marker_multiply_fix_substring_present'] = 'return a * b' in text
        rec['static_add_preserved'] = bool(
            rec.get('static_fixture_contract') and rec['static_fixture_contract'].get('add')
            and rec['static_fixture_contract']['add']['ok'])
        rec['static_multiply_changed'] = bool(
            rec.get('static_fixture_contract') and rec['static_fixture_contract'].get('multiply')
            and rec['static_fixture_contract']['multiply']['ok'])

        # Path normalization (Windows semantics) shared by the mandatory join
        # checks below and the fixture identity gate: relative paths resolve
        # against the session workspace, dot-segments collapse, slashes unify,
        # case folds. Raw paths stay on the events as evidence.
        ws_abs = str(ws.resolve())
        expected_fixture = ntpath.normpath(
            ntpath.join(ws_abs, RECOVERY_FIXTURE)).replace('\\', '/').lower()

        def norm_path(raw):
            p = str(raw or '').strip()
            if not p:
                return ''
            if not re.match(r'^[A-Za-z]:[\\\\/]', p) and not p.startswith('\\\\'):
                p = ntpath.join(ws_abs, p)
            return ntpath.normpath(p).replace('\\', '/').lower()

        # Canonical scoring from the persisted session history (read-only):
        # ordered tool calls joined to their results by tool_call_id. Joins are
        # MANDATORY: a scored receipt without exactly one unique matching
        # assistant call, a duplicated call id, unparseable message_json, or a
        # mutation action that differs from the call name makes the whole
        # scoring CANONICAL_SCORING_UNAVAILABLE - unbound metadata is never
        # scored silently.
        events = []
        scoring_error = None
        try:
            history_db = Path(env['NEXUS_AGENT_HOME']) / 'history.db'
            msgs = core.fetch_sqlite_messages(history_db, 'experiment-recovery-001')
            start = next((i for i, m in enumerate(msgs)
                          if m.get('role') == 'user'
                          and 'Find and fix the bug' in str(m.get('content') or '')
                          and 'multiply works correctly' in str(m.get('content') or '')), None)
            if start is None:
                scoring_error = 'recovery prompt not found in canonical history'
            else:
                calls = {}
                seen_receipts = set()
                for m in msgs[start:]:
                    if m.get('role') == 'assistant' and m.get('tool_calls'):
                        if not isinstance(m['tool_calls'], list):
                            scoring_error = 'malformed tool_calls (not a list)'
                            break
                        for call in m['tool_calls']:
                            fn = (call or {}).get('function') or {}
                            call_id = (call or {}).get('id')
                            call_name = fn.get('name')
                            if not call_id or not call_name:
                                scoring_error = 'malformed assistant tool call (missing id or name)'
                                break
                            if call_id in calls:
                                scoring_error = f'duplicate assistant tool call id {call_id}'
                                break
                            raw_args = fn.get('arguments')
                            if raw_args:
                                try:
                                    call_args = json.loads(raw_args)
                                except Exception:
                                    scoring_error = f'malformed tool call arguments (id={call_id})'
                                    break
                                if not isinstance(call_args, dict):
                                    scoring_error = f'tool call arguments not an object (id={call_id})'
                                    break
                            else:
                                call_args = {}
                            if call_name in MUTATION_TOOLS:
                                # A mutation call must carry its invocation
                                # payload: parsed object with a non-empty path.
                                # Empty/absent arguments can never score a
                                # mutation receipt.
                                if not str(call_args.get('path') or '').strip():
                                    scoring_error = (f'mutation call {call_name} without '
                                                     f'invoked path (id={call_id})')
                                    break
                            calls[call_id] = (call_name, call_args)
                        if scoring_error:
                            break
                    elif m.get('role') == 'tool':
                        call_id = m.get('tool_call_id')
                        if not call_id or call_id not in calls:
                            scoring_error = f'tool receipt without unique assistant call (id={call_id!r})'
                            break
                        if call_id in seen_receipts:
                            scoring_error = f'duplicate tool receipt for call id {call_id}'
                            break
                        seen_receipts.add(call_id)
                        name, call_args = calls[call_id]
                        raw_json = m.get('message_json')
                        if not raw_json:
                            scoring_error = f'tool receipt {call_id} missing message_json'
                            break
                        try:
                            meta = json.loads(raw_json).get('metadata') or {}
                        except Exception:
                            scoring_error = f'tool receipt {call_id} message_json parse failure'
                            break
                        outcomes = meta.get('mutation_outcomes') or []
                        if outcomes:
                            for mo in outcomes:
                                action = mo.get('action')
                                if not action:
                                    scoring_error = (f'mutation outcome without action '
                                                     f'(id={call_id})')
                                    break
                                if action != name:
                                    scoring_error = (f'mutation action {action!r} does not match '
                                                     f'call name {name!r} (id={call_id})')
                                    break
                                if action in MUTATION_TOOLS:
                                    mo_path = str(mo.get('path') or '').strip()
                                    invoked_path = str(call_args.get('path') or '').strip()
                                    if not invoked_path:
                                        scoring_error = (f'mutation receipt for call without '
                                                         f'invoked path (id={call_id})')
                                        break
                                    if mo_path and norm_path(mo_path) != norm_path(invoked_path):
                                        scoring_error = (f'receipt path {mo_path!r} does not match '
                                                         f'invoked path {invoked_path!r} (id={call_id})')
                                        break
                                events.append({
                                    'tool': action,
                                    'path': str(mo.get('path') or call_args.get('path') or ''),
                                    'executed': bool(meta.get('executed')),
                                    'refusal_class': meta.get('refusal_class'),
                                    'success': bool(mo.get('success')),
                                    'status': mo.get('status')})
                            if scoring_error:
                                break
                        else:
                            events.append({
                                'tool': name,
                                'path': str(call_args.get('path') or ''),
                                'executed': bool(meta.get('executed')),
                                'refusal_class': meta.get('refusal_class'),
                                'success': None,
                                'status': meta.get('outcome')})
                if not scoring_error:
                    unreceipted = [cid for cid in calls if cid not in seen_receipts]
                    if unreceipted:
                        scoring_error = (f'assistant tool call(s) without tool receipt: '
                                         f'{unreceipted}')
                if scoring_error:
                    events = []
        except Exception as exc:
            scoring_error = f'{type(exc).__name__}: {str(exc)[:200]}'
            events = []
        rec['canonical_events'] = events
        rec['canonical_scoring_error'] = scoring_error

        # Fixture identity: each raw path is normalized (above) and compared
        # for EXACT equality with this session's expected fixture path.
        def on_fixture(e):
            raw = str(e.get('path') or '')
            n = norm_path(raw)
            if not n:
                return False
            e['path_normalized'] = n
            matched = n == expected_fixture
            is_rel = bool(raw.strip()) and not re.match(r'^[A-Za-z]:[\\\\/]', raw.strip()) \
                and not raw.strip().startswith('\\\\')
            e['path_basis'] = (('relative_resolved_to_session_cwd' if is_rel else 'full_path_exact')
                               if matched else 'full_path_other')
            return matched

        fixture_events = [e for e in events
                          if e['tool'] in MUTATION_TOOLS and on_fixture(e)]
        rec['canonical_same_name_elsewhere_events'] = sum(
            1 for e in events if e['tool'] in MUTATION_TOOLS
            and norm_path(e.get('path')).endswith('/' + RECOVERY_FIXTURE.lower())
            and norm_path(e.get('path')) != expected_fixture)
        refusals = [i for i, e in enumerate(fixture_events)
                    if e['tool'] == 'patch_file' and not e['executed']
                    and e['refusal_class'] == 'targeting']
        accepts = [i for i, e in enumerate(fixture_events)
                   if e['tool'] == 'patch_file' and e['executed'] and e['success']]
        writes = [e for e in fixture_events if e['tool'] == 'write_file' and e['executed'] and e['success']]
        rec['canonical_targeting_refusals'] = len(refusals)
        rec['canonical_patch_accepts'] = len(accepts)
        rec['canonical_write_file_events'] = len(writes)
        refusal_then_accept = any(a > r for r in refusals for a in accepts)
        rec['canonical_ordered_refusal_then_accept'] = refusal_then_accept

        # Gate on the FULL contract, not the per-function fields alone: duplicate
        # defs or extra top-level imports/calls must also deny statics_ok.
        contract_ok = bool(rec.get('static_fixture_contract')
                           and rec['static_fixture_contract'].get('ok'))
        statics_ok = bool(rec['static_syntax_ok']) and contract_ok
        rec['statics_ok'] = statics_ok
        write_after_refusal = bool(refusals) and any(
            i > refusals[0] for i, e in enumerate(fixture_events)
            if e['tool'] == 'write_file' and e['executed'] and e['success'])
        if scoring_error:
            rec['outcome'] = 'CANONICAL_SCORING_UNAVAILABLE'
        elif refusal_then_accept and statics_ok:
            rec['outcome'] = 'RECOVERY_OBSERVED'
        elif refusal_then_accept:
            rec['outcome'] = 'RECOVERY_ACCEPTED_BUT_STATIC_MISMATCH'
        elif write_after_refusal and statics_ok:
            rec['outcome'] = 'RECOVERY_VIA_WRITE_FILE'  # write-mediated, statics proven
        elif write_after_refusal:
            rec['outcome'] = 'WRITE_AFTER_REFUSAL_STATIC_MISMATCH'
        elif not refusals and accepts:
            rec['outcome'] = 'NO_REFUSAL_FIRST_PATCH_ACCEPTED'
        elif writes and not any(e['tool'] == 'patch_file' for e in fixture_events):
            rec['outcome'] = 'MODEL_USED_WRITE_FILE_NO_REFUSAL'
        elif refusals:
            rec['outcome'] = 'RECOVERY_NOT_OBSERVED_TURN_ENDED'
        else:
            rec['outcome'] = 'NO_PATCH_ATTEMPT'
    except Exception as exc:
        rec['outcome'] = 'HARNESS_ERROR'
        rec['error'] = f'{type(exc).__name__}: {str(exc)[:256]}'
    finally:
        if pty is not None:
            try:
                pty.send_input('/exit\r')
                pty.wait_for_exit(timeout=15.0, deadline=deadline_end)
            except Exception:
                pass
            pty.close()
        try:
            core.charge_and_write(output_dir / 'transcripts' / 'recovery.log',
                                  strip_ansi(transcript[0]).encode('utf-8', errors='replace')[:2 * 1024 * 1024],
                                  byte_tracker)
        except Exception as exc:
            rec['transcript_error'] = type(exc).__name__
            rec['evidence_errors'].append('transcript:' + type(exc).__name__)
        try:
            core.charge_and_write(output_dir / 'recovery-row.json',
                                  json.dumps(rec, indent=2).encode('utf-8'), byte_tracker)
        except Exception as exc:
            rec['evidence_errors'].append('recovery_row:' + type(exc).__name__)
            print('RECOVERY_EVIDENCE_WRITE_INCOMPLETE:' + type(exc).__name__, file=sys.stderr)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'root', 'wheel', 'native-evidence', 'deadline-file'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    source = Path(args.source_root).resolve(strict=True)
    root = Path(args.root).resolve()
    wheel = Path(args.wheel).resolve(strict=True)
    native_evidence = Path(args.native_evidence).resolve(strict=True)
    if os.name != 'nt':
        raise RuntimeError('Windows required')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != BASE_COMMIT:
        raise RuntimeError('Baseline commit changed')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD^{tree}'], text=True).strip() != BASE_TREE:
        raise RuntimeError('Baseline tree changed')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Baseline source is dirty')
    hp = source / '.github/qualification/interactive_core.py'
    if sha(hp) != HARNESS_SHA:
        raise RuntimeError('Baseline harness changed')
    spec = importlib.util.spec_from_file_location('unchanged_interactive_core', hp)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    end = json.loads(Path(args.deadline_file).read_text(encoding='utf-8-sig'))['work_deadline_utc']

    workspace, home, temporary = root / 'workspace', root / 'home', root / 'tmp'
    output_dir = root / 'output'
    if any(d.exists() for d in (workspace, home, temporary)):
        raise RuntimeError('Fresh workspace, HOME and temporary directories required')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError('Fresh dedicated output required')
    for d in (workspace, home, temporary, output_dir, output_dir / 'transcripts'):
        d.mkdir(parents=True, exist_ok=True)
    (home / '.nexus_agent_initialized').write_text('Nexus-Agent initialized.\n', encoding='utf-8')

    candidate_seal = verify_candidate_seal(Path(__file__).resolve().parents[2])
    verify_seal_chain(core, source, root, wheel, native_evidence, candidate_seal)

    download_pinned_model(home, end, root)
    env = core.child_environment(home, temporary, workspace)
    env['NEXUS_AGENT_MODEL_REPO'] = MODEL_REPO
    env['NEXUS_AGENT_MODEL_FILENAME'] = MODEL_FILE

    # Exit semantics: honest model observations exit 0; harness errors, scoring
    # unavailability, and any required-evidence loss exit 1 after bounded
    # diagnostics are persisted. A green workflow means complete evidence,
    # never that recovery was observed.
    byte_tracker = [0]
    with PeakMemorySampler(root) as sampler:
        result = {'exit': 1}
        try:
            rec = run_recovery_session(core, root, env, end, output_dir, byte_tracker)
            result['exit'] = 0 if evidence_complete(rec) else 1
            if result['exit']:
                print('RECOVERY_EVIDENCE_INCOMPLETE:outcome='
                      + str(rec and rec.get('outcome')), file=sys.stderr)
        finally:
            model_path = home / '.nexus-agent/models/unsloth--Qwen3-4B-Instruct-2507-GGUF' / MODEL_FILE
            try:
                final_matches = model_path.is_file() and sha(model_path) == MODEL_SHA
                (output_dir / 'model-final-sha.json').write_text(json.dumps({
                    'kind': 'PINNED_MODEL_FINAL_DIGEST',
                    'expected': MODEL_SHA,
                    'final_matches_pin': final_matches},
                    indent=2), encoding='utf-8')
                if not final_matches:
                    # A missing or changed pinned model file is evidence/tamper
                    # incompleteness: the workflow must go red, never green.
                    result['exit'] = 1
                    print('MODEL_FINAL_DIGEST_MISMATCH', file=sys.stderr)
            except Exception as error:
                result['exit'] = 1
                print('MODEL_FINAL_DIGEST_INCOMPLETE:' + type(error).__name__, file=sys.stderr)
            # Diagnostic capture cannot mask the probe outcome or skip its cleanup.
            output = root / 'output'
            try:
                output.mkdir(parents=True, exist_ok=True)
                (output / 'peak-memory.json').write_text(json.dumps({
                    'kind': 'TELEMETRY_OPTIONAL_PEAK_RSS_SAMPLES',
                    'required': False,
                    'meaning': 'Optional telemetry only; absence or sampler error does not affect evidence completeness',
                    'samples': sampler.samples,
                    'peak_rss_bytes_by_pid': sampler.peak,
                    'sampler_error': sampler.error}, indent=2), encoding='utf-8')
            except Exception as error:
                # Optional telemetry loss is reported but never turns the run red.
                print('TELEMETRY_OPTIONAL_INCOMPLETE:' + type(error).__name__, file=sys.stderr)
            try:
                output.mkdir(parents=True, exist_ok=True)
                logs = root / 'home/.nexus-agent/llama-server'
                if logs.is_symlink() or not logs.resolve().is_relative_to(root.resolve()):
                    raise RuntimeError('Unsafe owned log directory')
                captured_logs = 0
                for index, log in enumerate(sorted(logs.glob('server-*.log')) if logs.exists() else []):
                    if index >= 4 or log.is_symlink() or not log.is_file() or not log.resolve().is_relative_to(logs.resolve()):
                        continue
                    cap = 2 * 1024 * 1024
                    core.check_evidence_cap(output, reserve=cap + core.MANIFEST_RESERVE)
                    with log.open('rb') as stream:
                        raw = stream.read(cap + 1)
                    if not raw:
                        continue  # empty log is not evidence
                    # Child environment contains no tokens and corpus is synthetic.
                    # Still redact conventional credential-shaped assignments defensively.
                    text = raw[:cap].decode('utf-8', errors='replace')
                    text = re.sub(r'(?i)(authorization|api[_-]?key|access[_-]?token|password)([=: ]+)[^\s,;]+', r'\1\2[REDACTED]', text)
                    encoded = text.encode('utf-8')[:cap]
                    (output / f'model-server-{index}.log').write_bytes(encoded)
                    (output / f'model-server-{index}-capture.json').write_text(json.dumps({
                        'owned_file': log.name, 'original_bytes': log.stat().st_size,
                        'captured_bytes': len(encoded), 'truncated': len(raw) > cap,
                        'meaning': 'Actual load/template/thinking diagnostics, not a separate model invocation'}, indent=2), encoding='utf-8')
                    captured_logs += 1
                (output / 'model-server-capture-summary.json').write_text(json.dumps({
                    'kind': 'MODEL_SERVER_LOG_CAPTURE',
                    'required': True,
                    'captured_logs': captured_logs,
                    'meaning': 'At least one nonempty model server log is required evidence'},
                    indent=2), encoding='utf-8')
                if captured_logs == 0:
                    raise RuntimeError('No nonempty model server log captured')
                core.check_evidence_cap(output)
            except Exception as error:
                # Log collection failure is evidence incompleteness, never a changed model score.
                result['exit'] = 1
                print('MODEL_LOG_CAPTURE_INCOMPLETE:' + type(error).__name__, file=sys.stderr)
        return result['exit']



if __name__ == '__main__':
    raise SystemExit(main())
