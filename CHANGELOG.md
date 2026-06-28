# Changelog

All notable changes to the DevMind project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
