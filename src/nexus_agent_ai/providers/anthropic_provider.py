from typing import Any, Dict, List, Optional
from .base import BaseProvider, ProviderResponse, Tool, RateLimitError
from .capabilities import resolve_limits
from ..utils.config import get_env_or_raise
from .messages import anthropic_messages, assistant_message, decode_native_tool_call, tool_result

def _finish_reason(stop_reason: Any) -> Optional[str]:
    """Report Claude's token-limit stops with the same "length" value other providers use."""
    if stop_reason == "max_tokens":
        return "length"
    return stop_reason if isinstance(stop_reason, str) else None


class AnthropicProvider(BaseProvider):
    """LLM Provider implementation for Anthropic Claude models."""

    def __init__(
        self, model: str = "claude-sonnet-4-6", context_size: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ):
        limits = resolve_limits("anthropic", model, context_size, max_output_tokens)
        self.context_size = limits.context_size
        self.max_output_tokens = limits.max_output_tokens
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("Anthropic provider requires the 'anthropic' package. Install nexus-agent-ai[all].") from e
        self._anthropic = anthropic
        api_key = get_env_or_raise("ANTHROPIC_API_KEY")
        self.client = self._anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _convert_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        # High synchronous output caps require streaming transport in the SDK.
        # Collect internally without rendering chunks; this is not the Batches API.
        # https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python#long-requests
        for chunk in self.stream(messages, tools, system):
            if isinstance(chunk, ProviderResponse):
                return chunk
        raise RuntimeError("Anthropic stream completed without a final response.")

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        converted_tools = self._convert_tools(tools)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": anthropic_messages(messages),
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        try:
            full_text = []
            with self.client.messages.stream(**kwargs) as stream:
                for text_chunk in stream.text_stream:
                    full_text.append(text_chunk)
                    yield text_chunk
                final_msg = stream.get_final_message()

            text = "".join(full_text)
            tool_calls = []
            finish_reason = _finish_reason(getattr(final_msg, "stop_reason", None))
            for content_block in final_msg.content:
                if content_block.type == "tool_use" and finish_reason != "length":
                    tool_calls.append(
                        decode_native_tool_call(content_block.name, content_block.input, content_block.id)
                    )

            raw_msg = assistant_message(text, tool_calls)
            in_tokens = final_msg.usage.input_tokens if final_msg.usage else 0
            out_tokens = final_msg.usage.output_tokens if final_msg.usage else 0

            yield ProviderResponse(
                text=text,
                tool_calls=tool_calls,
                raw_assistant_message=raw_msg,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                finish_reason=finish_reason,
            )
        except self._anthropic.RateLimitError as e:
            raise RateLimitError("Anthropic", str(e))
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "rate limit" in err_str or "quota" in err_str:
                raise RateLimitError("Anthropic", str(e))
            raise

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return tool_result(tool_call_id, result)
