# Nexus-Agent

A terminal coding assistant with a ReAct tool loop, local GGUF inference and optional cloud providers. It can inspect a workspace, make targeted edits, search documentation and help debug code. Model output can be wrong. Review changes before keeping them.

Python 3.11+ | MIT | Package: **nexus-agent-ai** | Commands: `nexus-agent-ai`, `nexus-agent`, `agent`

## Release status

**Version: 2.10.4** (see [PyPI](https://pypi.org/project/nexus-agent-ai/)). Update with `python -m pip install -U nexus-agent-ai`, then run `nexus-agent --version` and `nexus-agent doctor`.

**2.10.4 fixes:** real-user routing fixes and honest tool outcomes over published 2.10.3:
- **Routing and recovery:** smart tool routing recognizes explicit tool names, arithmetic requests, and file operations. Bare retries such as "try again" inherit the user's prior request context and edit intent rather than internal steering messages. Read-before-patch recovery provides exact target context so unread patch attempts recover reliably.
- **Claim guards:** affirmative claims of code modifications require actual file changes; claims that tests were run or changes were verified require matching tool evidence (executed passing test runs without subsequent modifications, post-edit reads of changed files). Unbacked self-verification claims report an unverified outcome and exit nonzero. Action promises without tool execution mark the response incomplete.
- **Search grounding:** search results surface actual retrieved source URLs; live-search requests require retrieved results in that turn. Python patch fallback accepts unique token-equivalent whitespace spans without semantic guessing.
- **Named-function completion guard:** explicit named Python function requests verify actual top-level definitions in the target file, reporting incomplete requests with nonzero exit codes without automatic repair or invented targets.
- **Truthful command exits:** all one-shot commands (`chat`, `debug`, `review`, `generate`) consistently exit with code 1 on unapplied edits, unbacked verification claims, incomplete requirements, or unresolved tool errors.

Local defaults remain **4096 context / 2048 configured output-ceiling tokens**. Both budgets are configurable; cloud defaults use documented model capabilities. Every request reserves the same effective output allowance it sends to the engine. Local sessions can visibly lower that allowance to fit an active tool conversation; cloud sessions reserve their full configured cap. Increasing limits is not evidence of better model reliability.

What changed since 2.8.1:

- 2.10.0: loop guardrails for small local models (see Loop guardrails below).
- 2.10.1: fixes `/pull-model` crashing inside the interactive REPL.
- 2.10.2: adds the `nexus-agent-ai` command, resumable verified model downloads with no compiled dependencies, per-folder permission before running a project's tests, and honest failure when an edit is requested but not applied.
- Interactive first-run setup asks for optional cloud keys, the default provider, a local model, and permission before downloading either the engine or weights. The provider choice applies immediately; an explicit `--provider` still wins. Enter keeps an existing configured model, or the hardware suggestion if none is configured. A replacement model choice is saved only after download approval. Declining leaves saved model settings unchanged and prevents local startup downloads in that invocation; a configured cloud provider can still run. `nexus-agent-ai setup` can replace a saved model. The wizard is skipped without a terminal; non-interactive local startup can download its selected model without prompting.
- `nexus-agent doctor` checks Python, RAM, disk, download speed, engine, model fit and PATH. `nexus-agent` and `nexus-agent chat` with no question open the REPL.
- `--context-size` and `/context`, token-based history trimming, `/compact`, a context meter, and one retry on context overflow.
- Reliable edits: read before patch, CRLF tolerant matching, closest-text hint on a miss, a warning when an edit request ends without a successful write, and a nonzero exit code on unresolved tool errors.
- API keys are typed hidden and stored owner-only (0600 on Linux/macOS, an owner-only ACL on Windows).
- Sessions are saved by default; `--continue` resumes the last one.
- Windows CI covers tests, diagnostics, key-file ACLs, session IDs, CRLF patching and pseudo-console permission prompts. A historical manual CI run exercised a real model first run and one chat on Python 3.12. The current checkout's first-run wizard and REPL were also exercised through a real Windows pseudo-console on Python 3.13, using a hidden fake key and a local HTTP provider fixture while declining engine/weight downloads; that verifies the interaction, not live cloud or model quality.

`pip install nexus-agent` is an unrelated package. Use `nexus-agent-ai`.

## Install

For the published release:

```bash
python -m pip install nexus-agent-ai
nexus-agent-ai --version
nexus-agent --help
```

### Running tests safely (permission)

`run_tests` runs the project's own code (pytest imports it) on your computer, with your permissions, not sandboxed. Nexus never does this without your say-so:

- In a terminal, when the model first wants to run tests in a folder, Nexus asks: `[o]` once (this folder, until Nexus exits, nothing saved), `[a]` always (this exact folder only, saved in `~/.nexus-agent/trusted_folders.json`), or `[N]` no (default).
- A saved folder covers only that folder, not its subfolders or parent.
- See and undo your choices: `nexus-agent-ai trust list`, `nexus-agent-ai trust add [folder]`, `nexus-agent-ai trust revoke [folder]` (`--all` removes everything).
- Non-interactive runs require a saved grant for that exact folder or `NEXUS_ALLOW_PROJECT_EXECUTION=1`. Use either only for code you trust.
- Optional: set `NEXUS_AGENT_AUTO_CHECK=1` to run the tests automatically after each successful edit. It only works in folders you already allowed, and is off by default.

If the model says "the tests pass" without a current passing run, one-shot commands exit with code 1, including verification-only questions. Streaming output displays the correction; the REPL reports the failure and remains open.

### Model downloads

Local models come from Hugging Face. Downloads resume from `.part` files after interruption, and every completion path, including HTTP 416, checks available size and SHA-256 metadata. Without a digest, checks are best-effort: HTTP framing can detect truncation but not same-size corruption. Cache destinations reject traversal and escaping symlinks while allowing nested repository filenames. For private or gated models, set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`, or use the Hugging Face CLI token file). Downloads and speed probes send the token only to the configured Hugging Face host and strip it on cross-host redirects. Private-repo access is covered by local server fixtures, not a real private-repository run.

**Known limits:** small local models (1.2B) often fail file edits and may not follow multi-step instructions. In our small checks (5 one-line bug fixes, 2 runs each, CPU only) the 1.2B model fixed 0 to 1 out of 10, so check every change it makes. Harness correctness is not a model-quality benchmark.

**Python versions:** CI covers 3.11, 3.12 and 3.13; newer versions have separate platform-specific coverage below. The downloader uses the standard library, not huggingface-hub/PyYAML. Default dependencies can use binary wheels on supported Windows versions, so a compiler is not needed when those wheels are available.

**Supported platforms (what has actually been run).** Python 3.11 or newer. An engine download exists for more platforms than have been tested, so the two are listed separately.

| Platform | Engine asset available | Tested in CI |
|---|---|---|
| Windows x64 | yes | wheel install on Python 3.11-3.14; full tests on 3.11 and 3.12; real model first run and one chat (3.12); permission prompt through a real Windows pseudo-console (3.12) |
| macOS arm64 (GitHub runner) | yes | wheel install and tests on Python 3.11-3.13; real model first run and one chat (3.12) |
| Linux x64 | yes | tests on Python 3.11-3.13; real model chat and permission prompt checked by hand on 3.13 |
| Windows ARM, macOS Intel | yes | not tested |
| Linux ARM | no prebuilt engine (build llama.cpp yourself or use Ollama) | not tested |

CI machines have fast datacenter networks and are not a measure of your hardware, RAM or download speed. Python 3.15 and GPUs are untested; the model download and the engine are CPU only.

**Command not found.** `pip` cannot edit your PATH. If `nexus-agent-ai` is not recognized (common after `pip install --user`, or when the Python installer's "Add to PATH" box was not ticked), run `python -m nexus_agent_ai`; `nexus-agent-ai doctor` prints the exact folder to add to PATH. CI checks this on Windows with the user Scripts folder removed from PATH. If the model download is interrupted, run the same command again: it resumes from the saved partial file.

For this checkout:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install .
```

The default installation has the CLI, Hugging Face download support and web search. Cloud SDKs are optional:

```bash
python -m pip install 'nexus-agent-ai[cloud]'
# Or choose one: [openai], [anthropic], [gemini]
```

For an unpublished checkout, use `python -m pip install '.[cloud]'` instead. The `[local]` extra installs torch/transformers; `[cpu]` installs llama-cpp-python and can require a C++ compiler. Neither is required for the downloaded llama-server path on supported desktop platforms.

## Use

Provider flags belong to the subcommand, not the root command:

```bash
nexus-agent chat -p local "What is a Python generator?"
nexus-agent chat -p local "Explain @calculator.py"
nexus-agent review calculator.py -p local
nexus-agent debug calculator.py -p local --error "Wrong calculation"
nexus-agent repl -p local
nexus-agent commit -p local
nexus-agent chat --help
```

Run inside the project you want Nexus to inspect. Regular chat/generate/debug sessions may write files in that workspace. Review sessions advertise only inspection tools, and the dispatcher rejects tools outside the session's allowed list, including REPL `/review`. Tool failures are appended to the final answer so a model cannot silently claim success after a failed operation. Commit-message generation is also read-only; the commit command asks for confirmation unless you deliberately pass `--yes`.

**Loop guardrails (since 2.10.0):** when a model asks for a file that does not exist, the tool result now includes the real directory listing. Before a model may say "tests pass", the loop checks the actual test result and appends an "unverified" note if the run failed. Answers to `git_status` and `git_diff` questions are grounded with the real git output. These checks fix the loop, not the model: on a 12-task local re-score, the guardrails moved CodeForge adapters from 4/12 to 6/12, and both gains came from the git tasks. Wrong code and wrong arithmetic from small models are not fixed.

`@file.py`, REPL review/debug reads and `read_file` share workspace and sensitive-path checks. Credential-like filenames, `.env` files, keys, sensitive configuration directories, and Nexus's own configuration/history/grant directories are blocked, including resolved symlink targets. Git inspection omits protected and outside-workspace paths, including both sides of a rename, and disables external diff, textconv, fsmonitor, hooks and configured clean/process filters. This is not content-based secret detection: an innocently named source file may still contain a secret.

## Project execution and Git mutations

Pytest imports arbitrary project code. A working-directory check and allowed arguments do **not** sandbox that code. It can read/write outside the workspace, access the network and run commands with your account's permissions.

In a terminal `run_tests` is offered, but its first execution requires the permission prompt described above. In non-interactive runs it is omitted unless an exact-folder saved grant or the environment opt-in allows it. To opt in explicitly for trusted code:

```bash
# Linux/macOS
NEXUS_ALLOW_PROJECT_EXECUTION=1 nexus-agent chat -p local "Run tests for calculator.py"
# PowerShell
$env:NEXUS_ALLOW_PROJECT_EXECUTION='1'
nexus-agent chat -p local "Run tests for calculator.py"
```

Once opted in, the model may run pytest without another prompt. Pytest has a 45-second timeout, restricted flags, disabled external plugin auto-loading and a reduced child environment. These are risk reductions, not filesystem/network isolation. Use a disposable container or VM for untrusted projects. Do not opt in merely because a model tells you to.

Autonomous `git_commit` is also disabled by default. Prefer the separate `nexus-agent commit` command to review the generated message. `NEXUS_ALLOW_GIT_COMMIT=1` deliberately enables model-directed commits. Git hooks can execute code; use trusted repositories only.

`run_code` is for restricted pure-computation Python snippets with AST checks, reduced environment and a timeout. It is not a general Python interpreter or a security boundary for hostile code.

## Local models

The built-in model family is Liquid AI LFM2.5. The initial run downloads a GGUF and, on supported platforms, a checksum-verified llama-server binary. Subsequent local inference works offline after downloads are complete. Web search still uses the network. Local inference has no provider API charge, but uses your CPU/GPU, memory, disk and power.

Official model information:

- [LFM2.5-1.2B-Instruct](https://docs.liquid.ai/lfm/models/lfm25-1.2b-instruct): 32K model context and native tool calling.
- [LFM2.5-2.6B](https://docs.liquid.ai/lfm/models/lfm25-2.6b): 128K model context and training for agent workloads.
- [Tool-use format](https://docs.liquid.ai/lfm/key-concepts/tool-use): Pythonic calls by default; JSON calls can be requested.

A model's advertised maximum context is not Nexus's configured context. Local llama-server uses one slot. `--context-size` / `NEXUS_CONTEXT_SIZE` set the request window; `--max-output-tokens` / `NEXUS_MAX_OUTPUT_TOKENS` set the generation cap. All agent commands accept both options. Larger local windows consume more RAM; automatic model selection still prices the chosen context against available RAM.

```bash
nexus-agent chat "Explain this project" -p local --context-size 8192
nexus-agent repl -p local --context-size 8192 --persist --session my-project
```

For each budget independently: explicit CLI/constructor setting > environment variable > known-model default > unknown-model fallback. CLI context values must be integers of at least 512; output caps must be positive integers. Known cloud model overrides above documented maxima are clamped with a visible warning. Unknown IDs do not inherit a guessed family limit: using the conservative fallback below emits an unverified-capacity warning. It is not proof that an arbitrary endpoint supports that window; check its documentation and override downward when needed.

| Cloud model | Default context/input budget | Default output ceiling |
| --- | ---: | ---: |
| [OpenAI `gpt-4o-mini`](https://developers.openai.com/api/docs/models/gpt-4o-mini) | 128,000 | 16,384 |
| [Gemini `gemini-2.5-flash-lite`](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite) | 1,048,576 | 65,536 |
| [Anthropic `claude-sonnet-4-6`](https://platform.claude.com/docs/en/models/sonnet-4-6/overview) | 1,000,000 | 128,000 |
| [Anthropic `claude-haiku-4-5`](https://platform.claude.com/docs/en/models/haiku-4-5/overview) | 200,000 | 64,000 |
| Unknown cloud ID (unverified fallback, warns) | 32,768 | 16,384 |

Exact IDs, documented snapshots and additional historical limits are recorded with sources in `providers/capabilities.py`; prefixes do not inherit a newer model's capacity. Anthropic's 300K output option is Batches-beta-only and is not used by these synchronous sessions. Anthropic `complete()` collects streaming transport internally, as its SDK requires for large output allowances. Loopback OpenAI-compatible endpoints use local defaults rather than the cloud fallback.

```bash
nexus-agent chat "Explain this project" -p openai --context-size 128000 --max-output-tokens 8192
nexus-agent repl -p gemini --context-size 200000 --max-output-tokens 16384
```

Local llama-server and llama-cpp-python apply the context to the engine. Explicit engine-context overrides on unsupported Transformers or automatic Ollama-fallback routes still fail clearly; use `-p ollama` for a directly configured OpenAI-compatible session. The local output option applies to llama-server, llama-cpp-python and Transformers. Local limits are not inferred from arbitrary GGUF names: stay within the loaded model's trained limits and available RAM. Each local provider starts its own server on a private port and verifies the model alias/context before reuse; a foreign server is neither reused nor stopped. Logs live at `~/.nexus-agent/llama-server/server-nexus-*.log`.

**Long sessions:** requests are trimmed by estimated tokens against the active provider's context. Cloud sessions reserve the full configured output cap; Gemini input budgeting reserves output conservatively even though its documented limits are separate. Local sessions first try the configured ceiling. If the active tool conversation cannot fit, they halve the effective generation cap until it fits, stopping at 256 tokens (or preserving a smaller explicit cap). A visible warning reports any reduction; that exact reduced cap is also sent to llama-server, llama-cpp-python, Transformers or the loopback OpenAI-compatible engine. No input-only reservation shortcut is used. Truly unfit structured turns and impossible instruction/window combinations still fail before a request.

`/context` shows the configured output ceiling, not a guarantee that every local request receives it. Its meter conservatively includes that ceiling along with instructions, schemas and stored history. Lower the configured ceiling or raise context within model/RAM limits when needed.

History trimming preserves real user turns and tool-call/result bindings without deleting stored history. Large file/tool text can be shortened, but tool arguments remain intact; an active turn that cannot fit raises a context-budget error before sending. Server-reported overflow permits one tighter-budget retry only when the full output reservation still fits. Auto-fallback refits against each active backend's context and output settings. `/compact` summarizes earlier requests, full file paths and tools without a model, retaining the last two turns; `/clear` removes the current history. Token estimates use about 3.5 characters per token, not the provider's exact tokenizer.


### First run and model fit

On a first local run Nexus picks a model by **free** RAM, keeping about 1 GB of headroom plus context space. It falls back to the 1.2B model (~0.7 GB) when larger models do not fit. Automatic selection shows size, measured speed and ETA, then asks `[y]es / [s]mall instead / [n]o` in a terminal. Enter accepts that selection; EOF/Ctrl+C cancels. Non-interactive startup or `NEXUS_AGENT_ASSUME_YES=1` accepts without a prompt. Explicit `NEXUS_AGENT_MODEL_REPO`/`NEXUS_AGENT_MODEL_FILENAME` choices bypass automatic selection. Disk checks use the actual model destination, including a relocated `NEXUS_AGENT_HOME`.

Before requesting another download, automatic selection reuses an already-cached supported quantization only when it fits measured free RAM. A cached larger model does not override that limit; explicit model overrides remain the user's choice.

`nexus-agent` and `nexus-agent chat` with no question both open the interactive REPL.

`nexus-agent doctor` checks Python version, free RAM and disk, download speed to Hugging Face, the inference engine, and which model fits. It downloads nothing. Use `--no-network` to skip the speed test.

You can select a smaller model explicitly:

```bash
export NEXUS_AGENT_MODEL_REPO=LiquidAI/LFM2.5-1.2B-Instruct-GGUF
export NEXUS_AGENT_MODEL_FILENAME=LFM2.5-1.2B-Instruct-Q4_0.gguf
nexus-agent chat -p local "What is 17 times 23?"
```

### September 30 real-user smoke results

On a Linux x86_64 host with 2 CPUs and about 2GB RAM, genuine offline LFM2.5-1.2B-Instruct Q4_0 answered 17x23 correctly (391). One targeted model-directed patch and a read-only review succeeded before publication. After a fresh install from public PyPI, math passed again, but two file-repair attempts did not complete: one used a nonexistent filename, and one read the right file then stopped without editing. The failed tool call was reported and the original file stayed unchanged. Direct `patch_file` checks passed with a backup and independently verified output.

These are smoke tests, not a model benchmark or proof of unattended reliability. Review the actual diff and run your own trusted tests. LFM2.5-2.6B inference did not complete reliably on this low-memory host, so its quality on suitable hardware remains unverified.

Setup and response time depend on download size and hardware. A 2GB CPU test host is not a useful quality benchmark for 2.6B models. Small models can produce wrong answers and tool calls. Native tool support alone does not guarantee successful file repair in every harness.

The downloaded desktop binary supports Windows, macOS and Linux x86_64. ARM Linux/Android require a compatible external engine/build; Termux installation is not a promise of tested on-device inference. Windows is exercised in CI (see Release status); Android execution was not verified.

## Cloud providers

Install the matching extra, then configure the appropriate key in your own environment or the onboarding wizard. Never commit keys.

```bash
nexus-agent chat -p openai "Explain @calculator.py"
nexus-agent chat -p anthropic "Review this project"
nexus-agent chat -p gemini "Explain this function"
```

`-p auto` uses configured providers with fallback. Cloud calls can cost money and transmit your selected file context. Cost output is an **estimate** based on the supported pricing table, not an invoice; unknown model prices are not tracked. Local sessions show zero estimated provider cost. OpenAI streaming requests usage totals when supported by the endpoint.

## Tools and persistence

Default tools: `read_file`, `write_file`, `patch_file`, `list_directory`, `run_code`, `search_web`, `git_status`, `git_diff`. Optional trusted-project tools: `run_tests`, `git_commit`.

Python writes and patches are syntax-checked before changing the file. This detects malformed Python, not logical mistakes.

`patch_file` requires a unique target unless multiple replacements are explicitly requested. Writes and patches encode and validate content before modifying files, save a backup, and atomically replace the destination. Backup symlinks/hardlinks are refused; backup failures abort the edit. Unchanged content is a no-op, not evidence of an applied edit. Workspace path checks are not OS-level isolation or protection against concurrent filesystem changes.

**Edit safety.** Existing files must have been read in the current process and remain unchanged on disk before writing or patching. Matching tolerates CRLF and trailing-whitespace differences and provides nearby text on a miss. Atomic replacements and backups preserve the original file's permission bits and Windows DACL, rather than inheriting broader parent-directory access. All one-shot agent commands check unresolved tool errors, iteration exhaustion, unapplied edit claims and unverified test claims. `generate` verifies that this run actually changed its requested output, reporting creation versus update accurately. `debug` explicitly requests an applied fix when needed; read-only review and informational questions such as "How do I add a function?" do not require an edit. The REPL reports the same failures without ending the session.

**Bounded named-function completion check.** With loop guardrails enabled (the default), explicit unquoted Python requests such as `rename the subtract function to multiply`, `add a divide(a, b) function`, or `define function divide` are checked against the actual workspace file's top-level function definitions, including async definitions. A recognized request needs one explicit unquoted relative `.py` target; missing or ambiguous targets are reported as unverified, rather than guessed. Missing names and unreadable or syntactically uncheckable source mark the request incomplete, append a correction even after streaming, and make one-shot commands exit nonzero. Partial edits remain on disk; this check neither repairs nor reverts them and never executes the source.

This is a presence check, not a general requirement verifier. Successful checks visibly qualify their scope: signatures, function bodies, arithmetic, removal of old symbols, caller updates and other instructions are not verified. Quoted/code examples, inline-code identifiers, methods and unsupported phrasing are outside the recognized grammar; review the diff and run trusted tests before accepting a multi-part edit.

**Bounded literal-assignment completion check.** With loop guardrails enabled, a complete instruction such as `Change x to 2 in s2target.py` or `Set enabled to False in settings.py` recognizes one identifier, one literal and one unquoted relative Python path (forward-slash paths are supported). Optional `please` and final punctuation are accepted. Supported literals are finite numbers, booleans, `None`, strings, bytes, lists, tuples and dictionaries without unpacking or duplicate keys; types are distinguished, so `True` is not evidence for `1`. Strings may be quoted literal values within an otherwise unquoted instruction. Quoted/code-example instructions, questions, multiple targets, additional instructions and computed values are outside this grammar and receive no invented verification.

The actual file is AST-inspected without executing it. The requested literal must be the final supported module-level binding of the requested name, not an earlier assignment overridden later. Missing or mismatched bindings, unsupported/conditional writes and unreadable or unparseable source make recognized requests incomplete and one-shot commands exit nonzero, for both whole-file writes and patches. This is a **presence check on a recognized grammar**, not arbitrary semantic correctness: it does not execute functions, model dynamic side effects or establish runtime correctness. Streaming answers receive the same incomplete correction. The check happens after editing; partial edits and backups remain, with no automatic repair or revert.

**Honesty, not a capability fix.** A small model can still overwrite `x = 1` with bare `2`. For a recognized assignment request this now reports incomplete/exit 1 instead of false success. macOS S2 may therefore record **PASS - HONEST FAILURE** rather than a successful edit; only **PASS - EDITED** establishes the harness's actual assignment check.

**Private storage.** Keys entered during setup are hidden and saved to `~/.nexus-agent/.env`. Key and history files use owner-only permissions: 0600 files in 0700 private directories on POSIX, and a protected current-user-only DACL on Windows. Permission failures are reported rather than silently leaving private data exposed. `doctor` checks POSIX key-file permissions.

Sessions: the REPL now saves the conversation by default, one session per folder, in `~/.nexus-agent/history.db`. Open the REPL again in the same folder and it resumes. `nexus-agent repl --continue` (or `-c`) resumes the most recently used session from any folder, `--session NAME` picks one, and `--no-persist` gives a throwaway session. One-shot `nexus-agent chat "question"` still does not save unless you pass `--persist` or `--continue`. File contents you read can be kept in the saved history, so use `--no-persist` for sensitive work. File context may be retained in that history. Both persistent and in-memory history preserve user-turn/tool-result boundaries when pruning; an active long turn can exceed the nominal message limit rather than lose its original request.

Provider messages are stored as JSON-safe canonical dictionaries. Anthropic/Gemini/OpenAI translations happen at provider boundaries, so saved histories and fallback preserve tool names and IDs without storing SDK objects. Empty or null tool arguments normalize to an empty object; malformed JSON is rejected. Gemini function results remain separate from subsequent user instructions, and blocked or empty replies report a failure rather than an empty success. Existing native-format history is migrated when loaded. SQLite connections close after each operation. Local llama-server uses progressive HTTP SSE streaming; optional non-streaming engines return a final response without fabricated token chunks.

CodeForge-250M integration is planned, not a shipped inference backend.

## Development and tests

```bash
python -m pip install -e '.[dev,cloud]'
python -m pytest -q
python -m pip install build
python -m build
```

The module-entry subprocess test keeps its temporary working directory and home, but explicitly inherits the test runner's resolved import paths. Pytest's `pythonpath` setting does not configure child interpreters, and replacing `HOME` can hide user-site dependencies even after an editable project install. This is test-environment setup, not application import fallback.


CI covers Python 3.11–3.13 and additional Windows/macOS jobs. Regression coverage includes workspace confinement, linked backups, atomic edit failures, protected state, Git helper suppression, search outcomes/provenance, token-safe Python matching, capability clamping, cloud history budgets, cutoff-safe dispatch and truthful command exits. Linux qualification exercises the installed cloud SDKs against a local HTTP/SSE server; it does not establish live API access or model accuracy. Real Windows console/DACL behavior requires Windows CI. Passing tests are not proof of model accuracy or a hostile-code sandbox.

The `Windows` and `macOS` workflows' manual-only `first-run` jobs download LFM2.5-1.2B Q4_0 and record three qualification scenarios: S1 requires exit 0 and the exact tool-arithmetic result `12541447`; S2 requires a real `x = 2` assignment after editing `s2target.py`, or a completed nonzero agent outcome explicitly labeled **PASS - HONEST FAILURE**, not completed work; S3 requires exit 0, an observed web search and an external source URL in the final answer. Exit 0 with an unchanged S2 file fails, as do startup/fatal failures. Search qualification does not certify the model's version claim against every source. Dispatch against the branch being qualified; ordinary PR checks do not run these model jobs. Logs, status records and S2 file state are uploaded as `windows-first-run-qualification` or `macos-first-run-qualification`, including after failures. macOS scenarios use verbose output; its artifact also includes matching `~/.nexus-agent/llama-server/server-nexus-*.log` files, preserving whatever engine build/backend-selection evidence the server emitted.

Release publication is maintainer-triggered through the `Publish` workflow with an existing version tag. It builds and checks the wheel/source distribution, tests the installed wheel on Linux, macOS and Windows with Python 3.11–3.13, creates the GitHub release, and uploads those same artifacts to PyPI through trusted publishing. The PyPI trusted publisher must authorize `Yash1bajpai/nexus-agent`, workflow `publish.yml`, environment `pypi`. Publication fails without that account-side authorization; committing a version does not update PyPI.

## License

MIT. See [LICENSE](LICENSE).
