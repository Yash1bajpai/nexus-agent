# Changelog

All notable changes to the Nexus-Agent project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.7] - 2026-07-15

### Fixed & Security Hardened (Audit Verification Release)
- **Strict AST Sandbox Hardening (`Issue 1`)**: Closed static analysis sandbox bypasses inside `run_code` (`tools.py`). Blocked indirect import/execution mechanisms including `importlib`, `runpy`, `getattr(builtins, ...)`, `__builtins__`, `open()`, `eval()`, and `exec()`.
- **Verify Download Error Handling (`Issue 2`)**: Updated `LocalQwenProvider.setup_model(verify_download=True)` (`local_provider.py`) to raise explicit `RuntimeError` if `huggingface_hub` is missing or model download fails during `nexus-agent pull-model`, preventing false-positive success reports.
- **Dynamic Version & Package Naming Alignment (`Issues 3 & 8`)**: Aligned `--version` CLI check (`app.py`), `pyproject.toml`, and `_get_mock_response()` pyproject simulation (`local_provider.py`) to consistently report `nexus-agent-ai` (`v2.2.7`).
- **Autonomous Debug & Review Real File Operations (`Issue 4 & 7`)**: Updated `debug` and `review` commands under `LocalQwenProvider` to autonomously invoke `read_file` to inspect target files and execute `write_file` to apply code fixes (`demo_bug.py`, `demo_review.py`, or arbitrary user targets).
- **True Real-Time Token Streaming (`Issue 5`)**: Updated `Agent.run()` (`core.py`) and provider implementations (`base.py`, `local_provider.py`, `anthropic_provider.py`, `openai_provider.py`, `gemini_provider.py`) to utilize generator-based token streaming (`provider.stream()`) whenever `stream=True`, outputting tokens live via `print_stream_chunk`.
- **Safe `.env` Target Traversal (`Issue 6`)**: Replaced hardcoded `.parent.parent.parent` (`onboarding.py`, `config.py`) with safe upward directory traversal (`_find_project_root()`), ensuring `.env` files are created cleanly at the `pyproject.toml` or git repository root without leaking to upper filesystem directories (`C:\Yash\.env`).
- **Prioritized Code Generation (`Issue 9`)**: Reordered `LocalQwenProvider` prompt intent routing (`local_provider.py`) so file generation/writing commands (`generate` / `write_file`) take precedence over file reading checks, ensuring `nexus-agent generate ... --output generated_smoke_test.py` correctly creates target code files.
- **Offline & Cached Search Labeling (`Issue 10`)**: Updated `execute_search_web()` (`tools.py`) to explicitly prefix fallback responses (`_get_curated_fallback`) with `[Offline/Cached Reference Data - Live search unavailable or rate-limited]:` when live DuckDuckGo results are unavailable or off-topic.

## [2.2.6] - 2026-07-15

### Fixed
- **Termux ARM64 Marker Evaluation**: Fixed PEP 508 environment marker syntax to use exact boolean logic (`sys_platform != 'android' and (platform_system != 'Linux' or platform_machine != 'aarch64')`). On ARM64 Linux / Termux (`aarch64`), `pip install nexus-agent-ai` now cleanly evaluates to `False` for `anthropic` and `openai`, avoiding `jiter`/`maturin` Rust compilation entirely while keeping prebuilt wheel support across Windows, macOS, and x86_64 Linux desktops.

## [2.2.5] - 2026-07-15

### Fixed
- **Complete Termux/ARM64 Environment Marker Coverage**: In Termux on Android (`aarch64-unknown-linux-android`), Python reports `sys.platform == 'linux'` and `platform.system() == 'Linux'`, causing `sys_platform != 'android'` to still evaluate to `True`. Updated environment markers for `anthropic` and `openai` (`not (platform_system == 'Linux' and platform_machine == 'aarch64')`) so that `pip install nexus-agent-ai` never attempts to compile Rust dependencies (`jiter`/`maturin`) on ARM64 Linux / Termux.

## [2.2.4] - 2026-07-14

### Fixed
- **Mobile/Termux/Android Rust Build Exclusions**: Fixed `pip install nexus-agent-ai` crashing when compiling Rust C-extensions (`jiter` via `anthropic` and `pydantic-core` via `openai`) on Android (`aarch64-linux-android`) when `maturin`/`rustc` are not installed. `anthropic` and `openai` are now restricted to non-Android platforms (`sys_platform != 'android'`) by default, allowing 100% pure-Python cross-platform installation on Android via `pip install nexus-agent-ai`. Desktop/Termux users who want all providers can optionally install via `pip install nexus-agent-ai[all]`.

## [2.2.3] - 2026-07-14

### Fixed
- **Mobile/Termux/Android Compatibility**: Fixed `pip install nexus-agent-ai` failing on Termux / Android with error `RuntimeError: platform android is not supported` during `psutil` C-wheel compilation. `psutil` is now restricted to `win32` and `darwin` using environment markers (`psutil>=5.9.0; sys_platform == 'win32' or sys_platform == 'darwin'`).
- **Zero-Dependency System Spec Detection**: Added pure-Python `/proc/meminfo` (for Linux/Android/Termux) and `ctypes.GlobalMemoryStatusEx` (for Windows) RAM and CPU detection in `onboarding.py`, so spec detection works flawlessly across all devices even without `psutil` installed.

## [2.2.2] - 2026-07-12

### Fixed
- **Critical: `IndentationError` in `onboarding.py` line 257**: Fixed bad indentation in `_print(...)` call inside `else` branch that caused first-run onboarding to crash with `IndentationError` before the wizard could start.
- **Critical: Wrong tool name `list_dir` in CPU fallback**: `local_provider.py` was calling `ToolCall("list_dir", ...)` but the registered tool is `list_directory`. This caused `ERROR: Unknown tool 'list_dir'` on every general command. Fixed to `list_directory`.
- **`[THINKING]` tags leaking into git commit messages**: When using the local provider, the commit message cleaner in `app.py` was not stripping `[THINKING]`, `[ACTION]`, `[OBSERVE]` markers, resulting in literal `[THINKING]` as the commit message. Now filtered out correctly.
- **`search_web` returning off-topic results**: DuckDuckGo sometimes returns irrelevant results (e.g. Wikipedia/YouTube) for technical queries. Added `_is_relevant()` keyword validator — if live results don't match any query keyword, a smart curated fallback is used.
- **Misleading startup download message**: `setup_model()` printed "Downloading core reasoning engine (~4.5 GB)..." even when the model was already cached. Now detects HuggingFace Hub cache and shows "Local engine cache found" when model is already present.
- **Search query truncation**: Fixed `last_msg[:50]` truncation in search query builder — now uses full stop-word filtered keyword extraction for cleaner, focused queries.

## [2.2.1] - 2026-06-29

### Added
- **`--max-iterations` / `-m` flag**: All 5 commands (`chat`, `repl`, `review`, `debug`, `generate`) now accept a `--max-iterations` option to override the default 10-step ReAct loop cap.
- **`run_file` tool**: New tool that runs an existing `.py` file directly via subprocess — the safe escape hatch when `run_code` sandbox blocks a user's existing script.
- **psutil dependency**: Added `psutil>=5.9.0` to `pyproject.toml` and `requirements.txt` so system spec detection in onboarding works out of the box.

### Fixed
- **Version sync**: `pyproject.toml` version bumped from `1.0.0` to `2.2.1` to match CLI output.
- **Dead code removed**: Cleaned up two `if False else` ternary blocks in `commit` command that were development leftovers.
- **SQLite Windows race condition**: `test_sqlite_memory` now explicitly calls `del mem` + closes connection before temp dir cleanup, preventing `PermissionError` on Windows.
- **`run_code` AST sandbox**: Added `_sandbox_check()` with `ast.walk()` analysis blocking `import os/subprocess/shutil/socket/...`, `open()`, `exec()`, and `eval()` calls before any execution occurs.

## [2.2.0] - 2026-06-29

### Added
- **Auto-Provider Fallback** (`src/providers/fallback_provider.py`): `FallbackProvider` chains `gemini → anthropic → openai` and silently switches on `RateLimitError` or missing API key, printing `[WARN]` instead of crashing.
- **First-Run Onboarding Wizard** (`src/cli/onboarding.py`): Detects first launch via `~/.nexus_agent_initialized`. Three-step setup: API key configuration, system spec detection (RAM/CPU via `psutil`, GPU via `nvidia-smi`), and default provider selection. Runs exactly once.
- **`agent commit` command**: Reads `git diff`, generates a conventional commit message via LLM, prompts for confirmation, and commits. Supports `--yes` flag for CI/automation.
- **`git_diff` and `git_commit` tools**: New tools powering the commit workflow.
- **Verbose ReAct Trace**: `--verbose` now shows `[THINKING]` / `[ACTION]` / `[OBSERVE]` labels. `run_code` shows full syntax-highlighted code (Monokai, line numbers) and up to 15 lines of output.
- **`--provider auto` flag**: Activates `FallbackProvider` for automatic multi-provider resilience.
- **`RateLimitError`**: Custom exception in `base.py` raised by all three providers on 429 responses.

### Fixed
- **Provider header inconsistency**: `get_provider_instance()` returns `(provider, resolved_name)` — `print_header()` always shows the actual active provider, not a stale/invalid input string.
- **429 raw crash**: All providers (`gemini`, `anthropic`, `openai`) now catch SDK-specific rate-limit exceptions and re-raise as `RateLimitError` for clean `[WARN]` output.
- **`run_code` verbose truncation**: Removed 40-char `args_str` truncation. `print_tool_result()` shows multi-line preview instead of 60-char single line.
- **README inconsistencies**: `claude-sonnet-4-6`, `gemini-2.5-flash`, correct LinkedIn URL (`yash-bajpai-b5a86332a`), removed broken `assets/demo.png`, added `agent debug`/`generate`/`commit` usage.

## [2.1.2] - 2026-06-28


### Added
- **GitHub Actions CI/CD Pipeline**: Automated matrix testing across Python 3.11, 3.12, and 3.13 (`.github/workflows/ci.yml`).
- **Architecture Decision Records (ADRs)**: Documented foundational engineering decisions in `docs/adr-001-react-loop.md`.
- **Competitive Comparison Matrix**: Highlighted key differentiators against GitHub Copilot CLI, Cursor, and Aider in `README.md`.
- **Mocked Provider Test Fixtures**: Isolated unit tests from real API keys and live network calls in `tests/test_providers.py`.

### Fixed
- **Eager Provider Import Crash**: Decoupled package imports in `src/providers/__init__.py` and `src/cli/app.py` so missing SDKs don't crash fallback providers.
- **Streaming Cost Doubling Bug**: Resolved duplicate `provider.stream()` billing calls when full completions are already cached in memory during streaming execution.
- **Robust `@mention` Path Resolution**: Upgraded file mention parser to support multi-file mentions, space-delimited paths, and absolute/relative path resolution against the working directory.

## [2.1.1] - 2026-06-27

### Added
- **Smart Startup Project Detection**: Automatically senses `.git`, `pyproject.toml`, or `package.json` to inject project directory context into LLM prompts.
- **`@filename` Context Injection**: Allows users to reference local files directly inside terminal chat queries.
- **Live Terminal Spinner**: Claude-like ASCII animation during thinking and execution stages.

### Changed
- Updated default model endpoints to `gemini-2.5-flash` and `claude-3-5-sonnet-20241022`.
- Standardized CLI syntax for `review`, `debug`, and `generate` subcommands.
