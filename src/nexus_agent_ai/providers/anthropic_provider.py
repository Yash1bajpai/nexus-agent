import anthropic
from typing import Any, Dict, List, Optional
from .base import BaseProvider, ProviderResponse, Tool, ToolCall, RateLimitError
from ..utils.config import get_env_or_raise

class AnthropicProvider(BaseProvider):
    """LLM Provider implementation for Anthropic Claude models."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        api_key = get_env_or_raise("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
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
        converted_tools = self._convert_tools(tools)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise RateLimitError("Anthropic", str(e))
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "rate limit" in err_str or "quota" in err_str:
                raise RateLimitError("Anthropic", str(e))
            raise

        text = ""
        tool_calls = []
        for content_block in response.content:
            if content_block.type == "text":
                text += content_block.text
            elif content_block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=content_block.id,
                        name=content_block.name,
                        args=content_block.input,
                    )
                )

        raw_msg = {"role": "assistant", "content": response.content}

        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            raw_assistant_message=raw_msg,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        converted_tools = self._convert_tools(tools)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
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
            for content_block in final_msg.content:
                if content_block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=content_block.id,
                            name=content_block.name,
                            args=content_block.input,
                        )
                    )

            raw_msg = {"role": "assistant", "content": final_msg.content}
            in_tokens = final_msg.usage.input_tokens if final_msg.usage else 0
            out_tokens = final_msg.usage.output_tokens if final_msg.usage else 0

            yield ProviderResponse(
                text=text,
                tool_calls=tool_calls,
                raw_assistant_message=raw_msg,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
        except anthropic.RateLimitError as e:
            raise RateLimitError("Anthropic", str(e))
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "rate limit" in err_str or "quota" in err_str:
                raise RateLimitError("Anthropic", str(e))
            raise

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": result,
                }
            ],
        }
