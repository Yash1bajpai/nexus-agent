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
We designed a lightweight, zero-dependency unified provider interface (`BaseProvider`, `Tool`, `ToolCall`, `ProviderResponse`) supporting Anthropic Claude, Google Gemini, and OpenAI models.
- **Rationale**: Relying on heavy external wrappers or proxy gateways adds dependency overhead, hides token-level cost accounting, and creates potential points of failure. By implementing native adapters for official SDKs with lazy loading, Nexus-Agent maintains complete control over streaming formats, tool schema translation, and exact billing tracking without requiring users to install unused SDK dependencies.

### 3. In-Memory Conversation Buffer vs. Heavy External Databases
We adopted an in-memory conversation buffer (`ConversationMemory`) with configurable rolling truncation (`MAX_CONVERSATION_MESSAGES`).
- **Rationale**: Terminal sessions are fast-paced and context-sensitive. Keeping memory lightweight ensures snappy REPL performance and predictable token limits while avoiding database locking or migration overheads during day-to-day development.

### 4. Direct DuckDuckGo Search vs. Paid Search APIs
We integrated `ddgs` (DuckDuckGo Search) for live web exploration.
- **Rationale**: Eliminates the barrier to entry by removing the need for paid API keys (like Google Custom Search or Serper) while providing robust live developer documentation retrieval.

### 5. Sandboxed Subprocess Execution with Timeout Enforcement
All code executions (`run_code`) are executed via isolated Python subprocesses with strict execution timeouts.
- **Rationale**: Prevents accidental infinite loops or hanging scripts from blocking the REPL interface, demonstrating production-grade security and resource awareness.

## Consequences
- **Positive**: High reliability, zero third-party gateway lock-in, real-time token cost estimation, and robust multi-vendor fallback capabilities.
- **Negative**: New provider SDK updates require maintaining schema conversion adapters inside `src/providers/`.
