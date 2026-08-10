# Nexus-Agent Deep Audit Report

**Date:** 2026-08-10 · **Version audited:** 2.4.0 (pyproject) · **Method:** real-user CLI usage + targeted probes. No source files were modified. (Note: to test anything at all, the broken editable install had to be repaired with `pip install -e .` — see BUG-01. This regenerated the gitignored `*.egg-info` metadata.)

**Environment:** Windows 24H2, Python 3.13.7, venv at repo root. `.env` contains a Gemini key (quota exhausted, limit = 0/min), an OpenRouter key (valid), and `test_key` placeholders for Anthropic/OpenAI. OS-level env vars exist for Gemini/Anthropic.

**What was exercised:** `chat`, `repl` (piped multi-turn), `review`, `debug`, `generate`, `commit`, `pull-model`, `--version`, `--help`, natural-language routing, onboarding first-run (fake HOME), all providers (`local`, `gemini`, `anthropic`, `openai`, `openrouter`, `auto`, bogus), `@mention`, all 9 tools, sandbox escapes, path security, memory pruning, pytest suite.

---

## 🔴 CRITICAL

### BUG-01 — Installation is broken out-of-the-box: CLI crashes, all tests fail to collect
- **Where:** repo root `nexus_agent_ai.egg-info/` (stale), `venv/Lib/site-packages/nexus_agent_ai-2.2.1.dist-info/`, console script `nexus-agent.exe`.
- **What happened:** Before repair, `nexus-agent --version` crashed with `ModuleNotFoundError: No module named 'src'` (the script's entry point was `src.cli.app:app`), `import nexus_agent_ai` failed, and `pytest tests/` produced 5 collection errors. pip listed both `nexus-agent 2.2.1` and `nexus-agent-ai 2.2.1` pointing at this repo.
- **Why:** a stale root-level `nexus_agent_ai.egg-info` with legacy flat-layout metadata (`entry_points.txt` → `src.cli.app:app`, `top_level.txt` → `assets/docs/src/tests/venv`) got picked up during an editable reinstall, so the editable finder mapped `src` instead of `nexus_agent_ai`. There is also a duplicate dist name (`nexus-agent` vs `nexus-agent-ai`) making pip state confusing.
- **Fix direction:** delete both egg-info dirs, `pip uninstall nexus-agent nexus-agent-ai`, then one clean `pip install -e .`. Consider renaming to a single canonical dist name.

### BUG-02 — `run_code` AST sandbox is escapable → full arbitrary command execution
- **Where:** [tools.py `_sandbox_check`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/tools.py#L145-L215).
- **What happened (executed PoC):**
  ```python
  import operator
  subs = operator.attrgetter("__subclasses__")(object)()
  wrap = [s for s in subs if "wrap_close" in str(s)][0]
  g = operator.attrgetter("__init__.__globals__")(wrap)
  # g["system"] is os.system → arbitrary shell commands
  ```
  Output: `ESCALATED: os module reachable -> True`. The AST checker only inspects attribute *nodes* and import names; string-based dunder access via `operator.attrgetter(...)` sails through.
- **Additional holes verified:** `import smtplib` / `import requests` / `import marshal` / `import threading` are all allowed (network exfiltration, e.g. sending email, is possible from agent-generated snippets).
- **Why:** blocklists of names/attributes cannot catch string-mediated attribute access; `operator` (and most stdlib) is not in `_FORBIDDEN_IMPORTS`.
- **Fix direction:** whitelist-only imports for `run_code`; block `operator`, `types`, `ctypes`, `smtplib`, `ftplib`, `telnetlib`, `requests`, `httpx`, `socketio`, … or run snippets with an import hook that denies everything not explicitly allowed.

### BUG-03 — Anthropic provider default model is retired → Anthropic mode is dead
- **Where:** [anthropic_provider.py L9](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/providers/anthropic_provider.py#L9) — `model: str = "claude-3-5-sonnet-20241022"`.
- **What happened:** `claude-3-5-sonnet-20241022` was deprecated 2025-08-13 and **retired 2025-10-22** (Anthropic deprecation policy). Any call today returns model-not-found even with a valid key. (Live verification blocked only by this sandbox's network; the retirement is documented.)
- **Why:** model pin never updated; README advertises `claude-sonnet-4-6` but the code never uses it (it's even in `PRICING` in config.py, unused).

### BUG-04 — OpenRouter provider model no longer exists → `--provider openrouter` dead, fallback chain has a dead link
- **Where:** [app.py L90](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/cli/app.py#L83-L93) and [fallback_provider.py `_make_openrouter`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/providers/fallback_provider.py#L10-L21) — hardcoded `poolside/laguna-m.1:free`.
- **What happened (live):** `Error code: 404 - No endpoints found for poolside/laguna-m.1:free.` with a valid key.
- **Why:** free OpenRouter models rotate; the model id was hardcoded without validation or config override.

### BUG-05 — Gemini agentic tool loop is broken in three independent ways
- **Where:** [gemini_provider.py](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/providers/gemini_provider.py).
- **5a (code-confirmed):** `stream()` yields its final `ProviderResponse` **without** `raw_assistant_message` (L172-177). In [core.py L166](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/core.py#L165-L167) `memory.add_raw(response.raw_assistant_message)` then inserts `None` into history → next request sends `None` as a content item → API error. So **Gemini + streaming + any tool call crashes the loop**. (`chat` streams by default, so this is the default path.)
- **5b (code-confirmed):** `format_tool_result_message` puts `tool_call_id` into `function_response.name`, and the id is `f"{fc.name}_{uuid.uuid4().hex[:8]}"` (L99). Gemini requires the response name to match a declared function (`read_file`), not `read_file_a1b2c3d4` → 400 on the second turn of every tool loop.
- **5c (API contract):** function responses are appended with role `"model"` directly after the model's `functionCall` turn. Gemini requires alternating user/model turns ("function call turns must come immediately after a user turn or a function response turn") → 400 even if 5b were fixed.
- **Note:** live proof was impossible because the `.env` Gemini key has **quota 0** (`quota_limit_value: '0'` — the key is disabled), which is itself worth fixing for demo purposes.

### BUG-06 — `.env` secrets are readable by the agent and can be shipped to cloud LLMs
- **Where:** [tools.py `execute_read_file`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/tools.py#L40-L54), [core.py `parse_at_mentions`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/core.py#L35-L74).
- **What happened:** `read_file(path=".env")` returns the full file including `GEMINI_API_KEY=...` / `OPENROUTER_API_KEY=...`. `@.env` mentions attach it too. Nothing filters secret patterns; the content is then embedded in prompts sent to Gemini/OpenAI/Anthropic/OpenRouter. `list_directory` hides `.env` from trees, which makes this look handled when it isn't.
- **Fix direction:** deny-list `.env*`, `*.pem`, credential files in `_validate_workspace_path` or the read path; mask lines matching `*_KEY=`/`*_TOKEN=`/`*_SECRET=`.

---

## 🟠 HIGH

### BUG-07 — The "local" provider runs no model at all, and "offline" mode secretly calls cloud APIs
- **Where:** [local_provider.py `_run_local_cpu_inference`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/providers/local_provider.py#L70-L188) and [config.py `_find_project_root`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/utils/config.py#L5-L17).
- **What happened:**
  1. On CPU (no torch), responses are **hardcoded heuristics**: `what is 25*4?` → regex+`eval`; file questions → parrot the file back as an "Observation Report"; everything else → a canned "I am ready to assist…" string. README/marketing describes real local inference (GGUF/AWQ engines).
  2. Even in a bare temp dir, every "local" query took **10–26 s**. Profiling proved why: `_run_local_cpu_inference` first silently calls `FallbackProvider().complete(...)` whenever any API key env var exists — and keys always exist because config.py loads the repo `.env` **from anywhere** (its second fallback walks `Path(__file__)` parents, which reaches the repo). So "offline" mode makes live network calls to Gemini/OpenRouter/Anthropic/OpenAI on every single turn, waits for their failures, then returns the canned text.
  3. `setup_model()` swallows download failures and still prints `✅ Core engine ready! Booting up...`.
- **Impact:** false offline claims, 10–25 s latency for trivial answers, silent cloud usage (cost/privacy), and the header claims model "Liquid LFM (2.6B-Q6_K Local)" while nothing of the sort runs.

### BUG-08 — Config/env loading leaks across projects and precedence is inverted
- **Where:** [config.py](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/utils/config.py#L5-L17).
- **What happened (live):** from a temp dir, `PROJECT_ROOT` resolved to the temp dir but `GEMINI_API_KEY`/`ANTHROPIC_API_KEY` were still loaded (repo `.env` via `__file__` parents), while `OPENAI_API_KEY`/`DEFAULT_PROVIDER` from the same file were missing in onboarding (they were shadowed/absent depending on OS env). `load_dotenv()` does not override pre-existing OS env vars, so machine-level vars silently beat the workspace `.env`.
- **Impact:** running nexus-agent in project A picks up keys/config from project B (or the repo checkout); users editing `.env` see no effect when OS vars exist. Onboarding wrote `[MISSING] OpenAI` although the workspace `.env` contained it.

### BUG-09 — Memory pruning can orphan tool messages and break long sessions
- **Where:** [memory.py `_prune`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/memory.py#L21-L63).
- **What happened:** pruning chooses cut points at any message with `role == "user"` — but Anthropic tool results are stored as `role: "user"` (content = `tool_result` blocks). Cutting there leaves a `tool_result` with no preceding `tool_use` → Anthropic 400 mid-session, unrecoverable until restart. Simple sequences I built happened to cut safely, but the invariant ("user-role = safe boundary") is structurally wrong for Anthropic's format. Same class of risk for Gemini `None` entries from BUG-05a.
- **Fix direction:** only cut at plain-text user messages; additionally drop leading assistant/tool messages after slicing.

### BUG-10 — `search_web` silently returns fabricated results
- **Where:** [tools.py `execute_search_web`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/tools.py#L262-L358); deps in pyproject (`duckduckgo-search>=6.0.0`).
- **What happened (live):** the installed `duckduckgo_search` 8.1.1 is the deprecated shim — it emits `RuntimeWarning: This package has been renamed to ddgs` and returns `[]`. The `ddgs` package is not installed (not in deps). So every search falls into `_get_curated_fallback`, which returns **hardcoded fake "search results"**: querying `python asyncio tutorial` returned Python 3.13 release notes; GPU queries return hardcoded RTX 5090 marketing copy. The `[Offline/Cached Reference Data]` label does not stop the LLM from citing these fabricated URLs as facts.
- **Fix direction:** depend on `ddgs`, and when unavailable return an explicit "no search available" message instead of invented results.

### BUG-11 — `commit` flow is inconsistent and commits garbage in local mode
- **Where:** [app.py `commit`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/cli/app.py#L339-L425) + [tools.py `execute_git_commit`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/tools.py#L406-L423).
- **What happened (live):**
  1. With only **unstaged** changes, `execute_git_diff` happily shows the working-tree diff and a commit message is generated — then `execute_git_commit` fails with `ERROR: No staged changes found`. The tool schema description even claims "Auto-stages tracked modified files if nothing is staged" — it doesn't.
  2. With `-p local --yes`, the canned local responses produced the commit message `Changed Files (git status --short):` — and it was committed (`git log` confirmed). The message cleaner only strips a few prefixes; nothing validates that the message looks like a conventional commit.

### BUG-12 — REPL dies on any provider error; EOF handled badly
- **Where:** [app.py `repl`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/cli/app.py#L167-L235).
- **What happened:** the inner loop catches only `KeyboardInterrupt`. A rate limit, auth error, or network blip on one message propagates to the outer `except Exception` → entire session terminates with `Fatal execution error`. EOF (Ctrl+D / closed stdin) raises `EOFError` from `typer.prompt` → same fatal path instead of a clean exit.

### BUG-13 — OpenAI provider silently routes to Ollama when `OLLAMA_HOST` is set
- **Where:** [openai_provider.py L12](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/providers/openai_provider.py#L10-L19).
- **Why:** `base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_HOST")`. A user who has `OLLAMA_HOST` exported for other tooling and asks for `-p openai` sends their OpenAI-keyed requests to their local Ollama server with no warning.

---

## 🟡 MEDIUM

### BUG-14 — `@mention` + local provider always reads the wrong file
- **Where:** [local_provider.py L119-L132](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/providers/local_provider.py#L119-L132).
- **What happened (live):** `chat "summarize @demo.py briefly"` → `ERROR: File not found: src/cli/app.py` (the hardcoded default). **Why:** `parse_at_mentions` attaches the file as `[Context attached from @demo.py: …]`; the provider's word scanner strips only `.,'"\`@` from words — not `:` — so `@demo.py:` → `demo.py:` never matches `endswith(".py")`, and the default target (itself a stale path from an older layout) is used.

### BUG-15 — Misleading rate-limit messaging and mislabeled failures
- **Where:** [core.py L156-L160](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/core.py#L156-L160), [fallback_provider.py `_switch_next`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/providers/fallback_provider.py#L98-L101).
- **What happened (live):** with `-p gemini` (no fallback configured) the app printed `[WARN] Gemini rate limit hit. Switching to next provider...` and then crashed — nothing switches. In `auto` mode the final 401 auth failure was reported as `openai rate limit hit` because `_switch_next` raises `RateLimitError` for *any* failure type.

### BUG-16 — Streaming output is duplicated; verbose trace leaks into stream
- **Where:** [core.py L223-L238](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/core.py#L223-L238).
- **What happened (live):** in streaming mode the full answer first prints as raw chunks, then the identical text renders again inside the `Response` panel. In local mode the literal `[THINKING]\n...` tag was streamed mid-sentence with no newline.

### BUG-17 — Header display bugs
- **Where:** [display.py `print_header`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/cli/display.py#L23-L29), [app.py `model_display`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/cli/app.py#L124).
- **What happened (live):** `provider_name.capitalize()` mangles names: `Liquid lfm (2.6b-q6_k local)`, `Openrouter (laguna-m.1:free)`, `Auto (gemini)`. Local provider has no `.model` attribute and `resolved_name != "local"`, so the header shows `(unknown)`.

### BUG-18 — Onboarding wizard issues
- **Where:** [onboarding.py](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/cli/onboarding.py).
- **Findings (live first-run with fake HOME):**
  - `test_key` placeholders show `[OK] configured` — zero validation of key format/liveness.
  - Step [4/4] attempts a **~2.2 GB model download on first launch without consent** (harmless here only because `huggingface_hub` was missing).
  - AVX2 detection is broken: the `python -c` subprocess result is computed and **discarded**; `platform.processor()` never contains `"avx2"`; the name-heuristic fallback misses real CPU strings — this Alder Lake i5 machine reported `AVX2: Not detected`.
  - Identity confusion in the same wizard: option list says `local (Liquid LFM 2.6B Offline)`, the note below says `Local Qwen-2.5-7B-Instruct-AWQ (4-bit) reasoning engine is built-in`.
  - Rich markup bug renders `[[1/4]]` instead of `[1/4]`.
  - `suggest_local_model()` is dead code (never called).

### BUG-19 — Natural-language router ambiguity
- **Where:** [app.py `NaturalAgentGroup`](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/cli/app.py#L27-L42).
- **What happened (live):** `nexus-agent review --provider local` → click usage error (`Missing argument 'FILE_PATH'`) instead of treating it as a chat query, because the first token matched the registered `review` command. Any real user sentence starting with a command word (`review`, `debug`, `generate`, `commit`) is hijacked.

### BUG-20 — `run_file` tool description contradicts behavior
- **Where:** [tools.py RUN_FILE_TOOL](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/tools.py#L597-L611).
- **What happened (live):** description says `No import restrictions unlike run_code`, but `execute_run_file` runs the same `_sandbox_check` and blocked a trivial `import os` script. The LLM is actively misled by its own tool schema, and virtually any real-world script (imports os/sys/pathlib) is blocked anyway.

### BUG-21 — `read_file` fails on non-UTF-8 files
- **Where:** [tools.py L47](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/tools.py#L40-L54).
- **What happened (live):** latin-1 and binary-ish files return `ERROR: Could not read file: 'utf-8' codec can't decode...`. `open(..., encoding="utf-8")` lacks `errors="replace"` (which @mention reading and REPL review *do* use). Common cp1252 files on Windows become unreadable.

### BUG-22 — SQLite persistence exists but is wired to nothing
- **Where:** [persistence.py](file:///c:/Yash/Programmer_Assistant/src/nexus_agent_ai/agent/persistence.py) — referenced only by tests.
- **Impact:** no session survives a restart; the passing `test_sqlite_memory` gives false confidence. Either wire it into REPL or drop it.

### BUG-23 — `generate` produces placeholder stubs in local mode
- **What happened (live):** `generate "a hello world python script" --output hello_gen.py` wrote a generic template that doesn't even print "hello world", then printed `✅ Verified generated file created`. Fine with a real LLM, but combined with BUG-07 the flagship demo command silently produces junk.

---

## 🔵 LOW / DOCS / HYGIENE

| # | Issue | Where |
|---|-------|-------|
| L1 | Default models contradict README everywhere: README says `claude-sonnet-4-6` / `gemini-2.5-flash` / `GPT-4o`; code defaults are `claude-3-5-sonnet-20241022` / `gemini-2.5-flash-lite` / `gpt-4o-mini` | providers, README |
| L2 | `pull-model` help says "4-bit AWQ (~4.5 GB)", README says Q6_K (~2 GB), code pulls `LFM2.5-2.6B-Q6_K.gguf`; its error message recommends `pip install nexus-agent-ai[all]` but `[all]` contains anthropic/openai, not huggingface-hub | app.py, local_provider.py |
| L3 | `.env.example` lists providers `anthropic|openai|gemini` only (no local/auto/ollama/openrouter) and `MAX_CONVERSATION_MESSAGES=20` vs config default 60 vs README "20 turns" | .env.example, config.py |
| L4 | `demo_local_qwen.py` crashes: `ModuleNotFoundError: No module named 'src.providers'` (stale pre-src-layout imports) | demo_local_qwen.py |
| L5 | `get_package_version()` reads `pyproject.toml` of whatever project the user runs inside → `--version` can report another project's version | config.py |
| L6 | `list_directory` docstring/schema say "2 levels deep" but code walks 5 | tools.py |
| L7 | Duplicate-tool-call guard blocks legitimate re-execution of identical calls within one query (e.g., re-read after write) | core.py L196-L206 |
| L8 | `estimated_cost` falls back to `claude-3-5-sonnet` pricing when provider lacks `.model` (LocalQwenProvider) | core.py L110-L112 |
| L9 | Layering: `agent.core` imports `cli.display` (agent depends on CLI/UI); spinner calls sprinkled through the cognitive loop | core.py |
| L10 | `chat`'s empty-query check is dead code — typer rejects the missing/empty argument before the body runs | app.py |
| L11 | Anthropic raw messages keep SDK pydantic objects in memory — fine for the API, but any JSON persistence (see BUG-22) would crash | anthropic_provider.py |
| L12 | Redirected/piped output on Windows PowerShell 5 shows mojibake box-drawing chars (stdout forced to UTF-8, consumers decode ANSI) | display.py/app.py |
| L13 | Onboarding can't be re-run (no `setup` command); init flag lives in `~/.nexus_agent_initialized` | onboarding.py |

---

## What works well (verified)

- After a clean reinstall: all **36 tests pass**; CLI starts fast (~0.3 s imports); `--version`, `--help`, `agent` alias OK.
- Workspace path sandbox correctly blocks reads/writes outside cwd (`C:\Windows\win.ini`, `C:\Users\Admin\...` denied); temp-dir writes allowed by design.
- `run_code` timeout enforcement works (infinite loop killed at 10 s).
- `write_file` creates nested directories; `generate` verifies output file on disk.
- REPL basic loop, `exit`/`quit`, per-message timing footer, bogus-provider warning → local fallback all behave.
- Memory pruning kept valid boundaries in every concrete sequence I constructed (the hazard in BUG-09 is structural, not trivially reproducible).
- Onboarding runs end-to-end non-interactively and writes the init flag.

## Suggested priority order

1. BUG-02 (sandbox RCE), BUG-06 (secret leakage) — security.
2. BUG-01 (install), BUG-03/04/05 (three dead providers) — core functionality.
3. BUG-07/08 (local mode honesty + config bleed), BUG-10 (fake search), BUG-09 (memory corruption), BUG-11/12 (commit + REPL robustness).
4. Medium/low items as polish; then update README/`.env.example` to match reality.
