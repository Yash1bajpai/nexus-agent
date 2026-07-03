import json
import openai
from typing import Any, Dict, List, Optional
from .base import BaseProvider, ProviderResponse, Tool, ToolCall, RateLimitError
from ..utils.config import get_env_or_raise

class OpenAIProvider(BaseProvider):
    """LLM Provider implementation for OpenAI GPT models and local OpenAI-compatible servers (Ollama/LM Studio)."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: Optional[str] = None, api_key: Optional[str] = None):
        import os
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_HOST")
        if api_key:
            pass  # use the provided key directly
        elif base_url:
            api_key = os.getenv("OPENAI_API_KEY", "ollama")
        else:
            api_key = get_env_or_raise("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
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

        if not tool_calls and '"name"' in text and '"arguments"' in text:
            import re, uuid
            matches = re.findall(r'(\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})', text, re.DOTALL)
            for m in matches:
                try:
                    tc_data = json.loads(m)
                    if "name" in tc_data and "arguments" in tc_data:
                        args_val = tc_data["arguments"]
                        parsed_args = args_val if isinstance(args_val, dict) else (json.loads(args_val) if isinstance(args_val, str) else {})
                        tc_id = f"call_{uuid.uuid4().hex[:8]}"
                        tool_calls.append(ToolCall(id=tc_id, name=tc_data["name"], args=parsed_args))
                        raw_tool_calls.append({
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": tc_data["name"], "arguments": json.dumps(parsed_args)}
                        })
                        text = text.replace(m, "").strip()
                except Exception:
                    pass

        if tool_calls and text:
            import re
            text = re.sub(r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', '', text, flags=re.DOTALL).strip()

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
            # Accumulate full response so we can apply JSON leakage stripping
            # before yielding anything (critical for local models that leak tool JSON into text)
            import re
            full_text = ""
            response_stream = self.client.chat.completions.create(**kwargs)
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content

            # Strip any raw tool JSON blocks that leaked into the text
            cleaned = re.sub(
                r'\{[\s]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
                '',
                full_text,
                flags=re.DOTALL
            ).strip()

            # Also strip trailing comment lines that local models add after JSON
            cleaned = re.sub(r'^\s*//.*$', '', cleaned, flags=re.MULTILINE).strip()

            if cleaned:
                yield cleaned

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
