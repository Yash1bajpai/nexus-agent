# Nexus-Agent

A terminal coding assistant with a ReAct tool loop, local GGUF inference and optional cloud providers. It can inspect a workspace, make targeted edits, search documentation and help debug code. Model output can be wrong. Review changes before keeping them.

Python 3.11+ | MIT | Package: **nexus-agent-ai** | Commands: `nexus-agent-ai`, `nexus-agent`, `agent`

## Release status

**Version: 2.10.3** (see [PyPI](https://pypi.org/project/nexus-agent-ai/)). Update with `python -m pip install -U nexus-agent-ai`, then run `nexus-agent --version` and `nexus-agent doctor`.

**2.10.3 fixes:** atomic file edits and backups preserve access permissions and reject linked backup destinations; workspace tools protect agent state; Git inspection filters protected paths and disables execution helpers. Provider histories are JSON-safe across persistence and fallback, local llama-server streams real tokens on a private per-instance port, and command outcomes distinguish actual edits from unsupported success claims. Download refusal no longer persists a replacement model or allows same-run local startup downloads; automatic cached-model selection respects free RAM.

**Unreleased fixes (not in the published 2.10.3 package):**

- Local routing recognizes explicit tool names and arithmetic requests. Bare retries such as “try again” inherit the last real request's routing and edit intent, not internal steering messages or an older topic.
- An English action announcement such as “I'll verify” or “run_code initiated” without any tool execution is reported as incomplete/unverified; one-shot commands exit unsuccessfully and streaming output includes the correction. Quoted examples and “run through an example” are not execution promises. This is a conservative language heuristic, not proof that every unsupported claim can be detected.
- Search keeps successful numbered results and URLs. Failures are errors with categories `empty`, `irrelevant`, `backend_error` (imports unavailable or only one backend executed and failed), or `both_failed` (both executed and failed). Diagnostics distinguish import/execution exceptions; empty results do not establish rate limiting or network blocking. Modern-backend exceptions can fall back to the legacy backend; empty/irrelevant responses do not trigger fallback.
- Native calls with malformed JSON, non-object arguments, or missing names are rejected before dispatch, rather than converted into guessed arguments. Omitted arguments remain supported for parameterless calls.
- llama-server, llama-cpp and OpenAI stop reasons are retained. A reported `length` stop marks the response incomplete and executes none of its attached calls, including partially generated calls. Transformers stopping status remains unobserved.

Context still defaults to **4096**. The llama-server, llama-cpp and Transformers output cap is still **2048** per generation; neither value is increased here. Follow-up proposal, not implemented: align request fitting's answer reservation with the generation allowance before making output limits configurable. Increasing either limit alone is not evidence of better model reliability and can increase memory use or reduce available history.

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

A model's advertised maximum context is not Nexus's configured context. Nexus defaults to **4096**, with one llama-server slot. Use `--context-size` on `chat` or `repl` to change it for that run, or set `NEXUS_CONTEXT_SIZE` as a default; a larger context consumes more RAM. Local HTTP errors now retain the server detail, including context overflow. Shorten input/history or choose a suitable context/model if a request will not fit.

```bash
nexus-agent chat "Explain this project" -p local --context-size 8192
nexus-agent repl -p local --context-size 8192 --persist --session my-project
```

Priority: `--context-size` > `NEXUS_CONTEXT_SIZE` > 4096. Values must be integer token counts of at least 512. The setting covers the whole window, including instructions, tool schemas, conversation, and generated output. It does not enlarge a model's trained limit, compact history, or guarantee that your RAM can hold the requested window. Start with 4096 on low-memory machines and raise it only within your model and hardware limits.

This option applies to local llama-server and llama-cpp-python, not cloud providers, `auto`, Ollama, or Transformers. Unsupported routes fail clearly. Each local provider starts its own server on a private port and verifies its model alias and configured context before reusing that owned process; a foreign healthy server is neither reused nor stopped. Per-instance logs are stored at `~/.nexus-agent/llama-server/server-nexus-*.log` (under `NEXUS_AGENT_HOME` when set). In REPL, `/context` shows the active window. Changing the window requires a restart; resume history with the same session ID.

**Long sessions:** history is trimmed by estimated tokens, preserving real user turns and tool-call/result bindings. Internal steering messages do not create new user turns. Large file/tool text can be shortened, but tool arguments remain valid and intact; if the active turn cannot fit, Nexus reports a context-budget error rather than sending a malformed or oversized request. One tighter-budget retry handles server-reported overflow. `/compact` summarizes earlier requests, full file paths and tools without calling a model, retaining the last two turns. `/clear` removes the current history. Counts are estimates (about 3.5 characters per token); auto-fallback requests are budgeted against each active local backend.


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

CI covers Python 3.11–3.13 and additional Windows/macOS jobs. Regression coverage includes real workspace confinement, linked backup destinations, atomic edit failures, protected state, Git helper suppression and secret filtering, download integrity/resumption, provider-native history migration, multi-turn fallback, streamed corrections and truthful command exits. The current checkout was verified on Windows/Python 3.13; POSIX permission tests require a POSIX runner. Passing tests are not proof of model accuracy or a hostile-code sandbox.

Release publication is maintainer-triggered through the `Publish` workflow with an existing version tag. It builds and checks the wheel/source distribution, tests the installed wheel on Linux, macOS and Windows with Python 3.11–3.13, creates the GitHub release, and uploads those same artifacts to PyPI through trusted publishing. The PyPI trusted publisher must authorize `Yash1bajpai/nexus-agent`, workflow `publish.yml`, environment `pypi`. Publication fails without that account-side authorization; committing a version does not update PyPI.

## License

MIT. See [LICENSE](LICENSE).
