# Static-only OpenRouter candidate, review required

This is a pipeline candidate, not a completed cloud benchmark and not a dispatch approval.

## Scope

- Initial dispatch only: smoke A and B N=5, ten normal prompts across five persistent two-turn Core sessions.
- Exact model: `nvidia/nemotron-3-ultra-550b-a55b:free`.
- Product source pinned to released SHA `019e54eee0882ac7629869fa83959a1d8d55c582`.
- Alternative A selected: key-holding static-only environment, Zone B absent.
- No cloud request was made during preparation. Repository secret remains unprovisioned.
- Reviewer's written CLEAR and administrator provisioning are both still required.

## Isolation

Only read_file, write_file, patch_file and list_directory are offered. Smoke A writes the exact AST-derived filtered payload; the HTTP transport compares the actual product payload against it and refuses mismatches. Both product execution dispatch references are replaced by the same closed allowlist. Calls to absent tools return an unavailable classification. Paths are limited to calc.py and listing its single scratch directory. Git, search, run_code, run_tests, and commit are absent. A Python audit hook refuses all subprocess/shell/fork/exec paths after SDK preparation and before any API response. Model-written fixture bytes are only read, stored, and AST-parsed; never imported, evaluated or executed. Core and dependencies load from the installed pinned wheel, not the scratch directory.

This intentionally differs from the local full-tool CLI benchmark. It measures the released Core with a constrained external harness, not unchanged CLI entrypoint, SQLite persistence or C-stage execution. All C1-C4 classifications are NOT_EXECUTABLE_IN_THIS_LEG. No C stage is run and none is counted pass/fail. Static oracle reports syntax validity and changed bytes only, never functional correctness or model success.

## Key and logs

Only the driver step gets the repository secret. No job-level secret, inherited secret, credential-persisting checkout, or execution job. No agent provisions or reads the key. Public endpoint verification receives no Authorization header. SDK debug logging is not enabled; HTTP request headers are never logged. Raw API responses, bounded-by-token-budget histories, fixture bytes and product console capture become artifacts. Auth/other exception strings and tracebacks are not emitted. A driver scan and separate always-run scan delete all local evidence and fail on sk-or- patterns before any artifact upload. There is no raw runner-console model transcript.

## Caps

Transport enforces exact destination POST /api/v1/chat/completions and exact model, nonstreaming only. Every actual POST, including the startup greeting, forced tool smoke and each retry, goes through one limiter: 18/minute,45/run-local-day and45/model. The single allowed model makes its model subcap equal to45. There is no cross-run durable quota ledger; concurrency plus separate dispatch approval is the boundary, and another dispatch must reconcile remaining provider quota. 401/402/403 hard abort;429 max3retries, exponential5/10/20seconds, Retry-After honored within5-60seconds. Each API operation has a180-second timeout and exactly one transport-error retry, counted through the same limiter. Ten Core iterations per turn,480-second signal deadline per turn,2400-second driver global limit,45-minute job timeout. Runner kill or cap exhaustion is stage-void, never a model failure or success.

## Evidence prepared offline

- Nine unit tests pass: minute cap, daily cap,429retries,auth no-retry,free endpoint checks,secret deletion,AST filtered tools.
- Synthetic-response full pipeline passes A and completes B5/5 with13requests. Synthetic authentication only, no real credential, no external model traffic. The result is a harness check, not model evidence.
- Source compiles; secret-reference scope gate passes.
- Workflow path intentionally replaces registered bench.yml on a new experimental branch if later approved, avoiding an undispatchable new workflow path. Nothing merges to main.

## Required review changes implemented

R1: openai3.56.0 and httpx0.28.1 exact-pinned. All37 resolved dependency packages, including product dependencies and build tools, are in a SHA256-hashed lockfile. Workflow uses --require-hashes --only-binary for dependencies and --no-deps --no-build-isolation for pinned product. pip freeze --all is captured before the secret step and copied into scanned evidence. Offline pip hash-install and product-install checks pass.

R2:180-second timeout, one transport/timeout retry per operation, separately from the max3 rate retries. Every attempt counted. Nine unit tests include repeated timeout/connect failure stops and single-failure recovery.

R3: synthetic first B turn issues a valid read_file and an outside-path read, then a final answer. Assertions verify real fixture read, outside-path denial, second request counted, and exact payload enforcement. Synthetic A+B5/5 pipeline now uses13requests.

Non-blocking fixes: malformed Stage A arguments classify as explicit tool-gate failure; driver scan rejects symlinks and deletes evidence. Context4096 remains a comparability limit.

ONE dispatch per UTC day. Any additional same-day dispatch requires live quota reconciliation plus reviewer approval. User/admin provisioning and post-push matching tree hashes remain dispatch conditions.

## Reviewer focus

Please assess the API/Core deviation, audit hook and closed dispatch, destination/model/price checks, static C classifications, limiter and cap-stop semantics, artifact handling, and exact workflow. No dispatch until written CLEAR. The patch is not yet committed or pushed.
