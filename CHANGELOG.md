# Changelog

All notable changes to the DevMind project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **First-Run Onboarding Wizard** (`src/cli/onboarding.py`): Detects first launch via `~/.devmind_initialized`. Three-step setup: API key configuration, system spec detection (RAM/CPU via `psutil`, GPU via `nvidia-smi`), and default provider selection. Runs exactly once.
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
