import json
import openai
from typing import Any, Dict, List, Optional
from .base import BaseProvider, ProviderResponse, Tool, ToolCall, RateLimitError
from ..utils.config import get_env_or_raise

class OpenAIProvider(BaseProvider):
    """LLM Provider implementation for OpenAI GPT models."""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = get_env_or_raise("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def _convert_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def _format_messages(self, messages: List[Dict[str, Any]], system: str) -> List[Dict[str, Any]]:
        formatted = []
        if system:
            formatted.append({"role": "system", "content": system})
        for m in messages:
            formatted.append(m)
        return formatted

    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        converted_tools = self._convert_tools(tools)
        formatted_msgs = self._format_messages(messages, system)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_msgs,
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        response = None
        try:
            response = self.client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            raise RateLimitError("OpenAI", str(e))
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "rate limit" in err_str or "quota" in err_str:
                raise RateLimitError("OpenAI", str(e))
            raise
        choice = response.choices[0]
        msg = choice.message

        text = msg.content or ""
        tool_calls = []
        raw_tool_calls = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    parsed_args = json.loads(tc.function.arguments)
                except Exception:
                    parsed_args = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        args=parsed_args,
                    )
                )
                raw_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })

        raw_msg: Dict[str, Any] = {"role": "assistant"}
        if msg.content is not None:
            raw_msg["content"] = msg.content
        else:
            raw_msg["content"] = ""

        if raw_tool_calls:
            raw_msg["tool_calls"] = raw_tool_calls

        in_tokens = 0
        out_tokens = 0
        if response.usage:
            in_tokens = response.usage.prompt_tokens
            out_tokens = response.usage.completion_tokens

        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            raw_assistant_message=raw_msg,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        converted_tools = self._convert_tools(tools)
        formatted_msgs = self._format_messages(messages, system)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_msgs,
            "stream": True,
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        try:
            response_stream = self.client.chat.completions.create(**kwargs)
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.RateLimitError as e:
            raise RateLimitError("OpenAI", str(e))
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "rate limit" in err_str or "quota" in err_str:
                raise RateLimitError("OpenAI", str(e))
            raise

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result),
        }
