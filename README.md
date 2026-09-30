# Nexus-Agent

A terminal coding assistant with a ReAct tool loop, local GGUF inference and optional cloud providers. It can inspect a workspace, make targeted edits, search documentation and help debug code. Model output can be wrong. Review changes before keeping them.

Python 3.11+ | MIT | Package: **nexus-agent-ai** | Commands: `nexus-agent`, `agent`

## Release status

This branch prepares **2.8.1**. It is not a PyPI release until a maintainer publishes it. At the September 30, 2026 review, PyPI served 2.7.2 and main described 2.8.0. PyPI 2.7.2 does not contain the newer `patch_file` and `run_tests` tools or these safety fixes. Do not assume `pip install` retrieves branch code.

`pip install nexus-agent` is an unrelated package. Use `nexus-agent-ai`.

## Install

For the published release:

```bash
python -m pip install nexus-agent-ai
nexus-agent --version
nexus-agent --help
```

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

Run inside the project you want Nexus to inspect. Regular chat/generate/debug sessions may write files in that workspace. Review sessions advertise only inspection tools, and the dispatcher rejects tools outside the session's allowed list. Commit-message generation is also read-only; the commit command asks for confirmation unless you deliberately pass `--yes`.

`@file.py`, REPL review/debug reads and `read_file` share workspace and sensitive-path checks. Credential-like filenames, `.env` files, keys and sensitive configuration directories are blocked, including resolved symlink targets. This is not content-based secret detection: an innocently named source file may still contain a secret. Do not ask Nexus to inspect such files.

## Project execution and Git mutations

Pytest imports arbitrary project code. A working-directory check and allowed arguments do **not** sandbox that code. It can read/write outside the workspace, access the network and run commands with your account's permissions.

`run_tests` is **disabled and omitted from the model's tools by default**. For a project you trust, opt in before launching Nexus:

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

A model's advertised maximum context is not Nexus's configured context. Nexus defaults to **4096**, with one llama-server slot. Set `NEXUS_CONTEXT_SIZE` to change it; a larger context consumes more RAM. Local HTTP errors now retain the server detail, including context overflow. Shorten input/history or choose a suitable context/model if a request will not fit.

You can select a smaller model explicitly:

```bash
export NEXUS_AGENT_MODEL_REPO=LiquidAI/LFM2.5-1.2B-Instruct-GGUF
export NEXUS_AGENT_MODEL_FILENAME=LFM2.5-1.2B-Instruct-Q4_0.gguf
nexus-agent chat -p local "What is 17 times 23?"
```

Setup and response time depend on download size and hardware. A 2GB CPU test host is not a useful quality benchmark for 2.6B models. Small models can produce wrong answers and tool calls. Native tool support alone does not guarantee successful file repair in every harness.

The downloaded desktop binary supports Windows, macOS and Linux x86_64. ARM Linux/Android require a compatible external engine/build; Termux installation is not a promise of tested on-device inference. Windows and Android execution were not verified in this review.

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

`patch_file` creates a backup and requires a unique target unless multiple replacements are explicitly requested. Inspect the backup and diff. Workspace path checks resolve symlinks; they are not OS-level isolation or protection against concurrent filesystem changes.

The REPL stores session history under `~/.nexus-agent`. File context may be retained in that history. Both persistent and in-memory history preserve user-turn/tool-result boundaries when pruning; an active long turn can exceed the nominal message limit rather than lose its original request.

CodeForge-250M integration is planned, not a shipped inference backend.

## Development and tests

```bash
python -m pip install -e '.[dev,cloud]'
python -m pytest -q
python -m pip install build
python -m build
```

CI runs Python 3.11, 3.12 and 3.13. The September 30 branch suite has 133 tests, including regression coverage for read-only capability bypass, credential attachment, pytest opt-in and environment stripping, Git mutation gating, SQLite history, local cost and detailed context errors. Passing tests are not proof of model accuracy or a hostile-code sandbox.

Use a clean environment to install the built wheel and test the CLI before publishing. Publication requires a separate maintainer approval and matching version; branch fixes do not update existing PyPI installations.

## License

MIT. See [LICENSE](LICENSE).
