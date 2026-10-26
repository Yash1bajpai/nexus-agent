# ADR-001: Autonomous ReAct Tool-Use Loop & Provider Abstraction

## Status
Accepted

## Context
When building Nexus-Agent, an autonomous AI coding assistant for the terminal, we needed to make key architectural choices regarding how the agent interacts with LLMs, manages execution loops, executes tools locally, and handles multiple LLM vendors.

## Decisions

### 1. ReAct Loop vs. Simple Prompt Chaining
We chose to implement a custom **ReAct (Reason + Act)** autonomous execution loop rather than simple prompt chaining or fixed workflows.
- **Rationale**: Real-world software engineering tasks require iterative investigation (reading files, searching codebases, checking git diffs) and dynamic reasoning before taking action. A flexible ReAct loop allows the LLM to inspect intermediate tool results and course-correct autonomously up to a configurable max iteration limit.

### 2. Custom Provider Abstraction vs. LiteLLM / External Gateways
We use a small provider interface (`BaseProvider`, `Tool`, `ToolCall`, `ProviderResponse`) with JSON-safe canonical conversation dictionaries. Assistant messages contain text and explicit tool calls; tool results reference stable call IDs. Adapters translate that representation to OpenAI, Anthropic and Gemini wire formats.
- **Rationale**: SDK-specific objects must not leak into storage or another provider after fallback. Native histories are migrated at ingestion. Optional SDKs remain lazy-loaded; usage and cost are estimates, not exact billing.

### 3. In-Memory and SQLite Conversation Storage
`ConversationMemory` supports throwaway sessions; `SQLiteMemory` saves REPL sessions by default and closes each connection deterministically. Persistent files and their private directories enforce owner-only access.
- **Rationale**: Both backends share real-user-turn boundaries and canonical messages. Local request budgeting preserves tool bindings, excludes internal nudges as turn boundaries, and fails explicitly when an active turn cannot fit. Compaction summarizes older turns without a model call.

### 4. Direct DuckDuckGo Search vs. Paid Search APIs
We integrated `ddgs` (DuckDuckGo Search) for live web exploration.
- **Rationale**: Eliminates the barrier to entry by removing the need for paid API keys (like Google Custom Search or Serper) while providing robust live developer documentation retrieval.

### 5. Restricted Execution and Workspace Edits
`run_code` uses AST restrictions, a reduced subprocess environment and a timeout. Project pytest execution requires explicit exact-folder consent or environment opt-in and is not sandboxed. Read-only Git inspection disables executable helpers and omits protected/outside-workspace paths.
- **Rationale**: Timeouts and syntax checks reduce mistakes; they do not isolate hostile code. File edits pre-encode content, reject linked backup destinations and use atomic replacement. Nexus state is never writable through workspace tools. Local inference servers use owned per-instance ports and verified model aliases; foreign processes are not adopted or stopped.

## Consequences
- **Positive**: No gateway lock-in; portable JSON-safe persistence and fallback; explicit execution consent; consistent completion/error reporting after streamed and non-streamed responses.
- **Negative**: Provider SDK protocols still require maintained adapters and integration tests in `src/nexus_agent_ai/providers/`. Model correctness, hostile-code isolation and changing files concurrently remain outside these guarantees.
