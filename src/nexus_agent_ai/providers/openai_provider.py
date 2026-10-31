from typing import Any, Dict, List, Optional
from ipaddress import ip_address
from urllib.parse import urlsplit
from .base import BaseProvider, ProviderResponse, Tool, RateLimitError
from .capabilities import resolve_limits
from ..utils.config import get_env_or_raise
from .messages import assistant_message, decode_native_tool_call, extract_json_tool_calls, openai_messages, tool_result
import uuid

class OpenAIProvider(BaseProvider):
    """LLM Provider implementation for OpenAI GPT models and local OpenAI-compatible servers (Ollama/LM Studio)."""
    _supports_output_budget = True

    def __init__(
        self, model: str = "gpt-4o-mini", base_url: Optional[str] = None,
        api_key: Optional[str] = None, context_size: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("OpenAI-compatible provider requires the 'openai' package. Install nexus-agent-ai[all].") from e
        self._openai = openai
        import os
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        hostname = urlsplit(base_url).hostname if base_url else "api.openai.com"
        self.is_local = hostname == "localhost" or bool(hostname and hostname.endswith(".localhost"))
        if hostname and not self.is_local:
            try:
                self.is_local = ip_address(hostname).is_loopback
            except ValueError:
                pass
        limits = resolve_limits("openai", model, context_size, max_output_tokens, local=self.is_local)
        self.context_size = limits.context_size
        self.max_output_tokens = limits.max_output_tokens
        # OpenAI Chat Completions supports the replacement on gpt-4o-mini too:
        # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        # Other compatible servers may implement only the original max_tokens.
        self._max_tokens_field = "max_completion_tokens" if hostname == "api.openai.com" else "max_tokens"
        if api_key:
            pass  # use the provided key directly
        elif base_url:
            api_key = os.getenv("OPENAI_API_KEY", "local-key")
        else:
            api_key = get_env_or_raise("OPENAI_API_KEY")
        self.client = self._openai.OpenAI(api_key=api_key, base_url=base_url)
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
        formatted.extend(openai_messages(messages))
        return formatted

    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        converted_tools = self._convert_tools(tools)
        formatted_msgs = self._format_messages(messages, system)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_msgs,
            self._max_tokens_field: self.request_output_cap,
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        response = None
        try:
            response = self.client.chat.completions.create(**kwargs)
        except self._openai.RateLimitError as e:
            raise RateLimitError("OpenAI", str(e))
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "rate limit" in err_str or "quota" in err_str:
                raise RateLimitError("OpenAI", str(e))
            raise
        choice = response.choices[0]
        msg = choice.message

        text = msg.content or ""
        finish_reason = choice.finish_reason if isinstance(getattr(choice, "finish_reason", None), str) else None
        tool_calls = []
        if finish_reason != "length":
            for tc in msg.tool_calls or []:
                tool_calls.append(decode_native_tool_call(
                    getattr(tc.function, "name", None), getattr(tc.function, "arguments", ""), tc.id,
                ))
            if not tool_calls:
                text, tool_calls = extract_json_tool_calls(text)
        raw_msg = assistant_message(text, tool_calls)

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
            finish_reason=finish_reason,
        )

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        converted_tools = self._convert_tools(tools)
        formatted_msgs = self._format_messages(messages, system)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_msgs,
            self._max_tokens_field: self.request_output_cap,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        try:
            full_text = ""
            # tool_call_deltas: {index: {"id": str, "name": str, "arguments": str}}
            tc_deltas: Dict[int, Dict[str, str]] = {}
            in_tokens = 0
            out_tokens = 0
            finish_reason = None
            native_error = None

            response_stream = self.client.chat.completions.create(**kwargs)
            for chunk in response_stream:
                if not chunk.choices:
                    # Final chunk may carry usage without choices
                    if chunk.usage:
                        in_tokens = chunk.usage.prompt_tokens or 0
                        out_tokens = chunk.usage.completion_tokens or 0
                    continue

                delta = chunk.choices[0].delta
                reason = getattr(chunk.choices[0], "finish_reason", None)
                if isinstance(reason, str):
                    finish_reason = reason
                if delta and delta.content:
                    full_text += delta.content
                    yield delta.content

                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tc_deltas:
                            tc_deltas[idx] = {"id": f"call_{uuid.uuid4().hex}", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tc_deltas[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name is not None:
                                if not isinstance(tc_delta.function.name, str):
                                    native_error = "Native tool call function name must be a string"
                                else:
                                    tc_deltas[idx]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments is not None:
                                if not isinstance(tc_delta.function.arguments, str):
                                    native_error = "Native tool call argument fragments must be JSON text"
                                else:
                                    tc_deltas[idx]["arguments"] += tc_delta.function.arguments

                if chunk.usage:
                    in_tokens = chunk.usage.prompt_tokens or 0
                    out_tokens = chunk.usage.completion_tokens or 0

            # Build complete tool calls from accumulated deltas
            tool_calls = []
            if finish_reason != "length":
                if native_error:
                    raise ValueError(native_error)
                for idx in sorted(tc_deltas):
                    tcd = tc_deltas[idx]
                    tool_calls.append(decode_native_tool_call(tcd["name"], tcd["arguments"], tcd["id"]))
                if not tool_calls:
                    full_text, tool_calls = extract_json_tool_calls(full_text)
            raw_msg = assistant_message(full_text, tool_calls)

            yield ProviderResponse(
                text=full_text,
                tool_calls=tool_calls,
                raw_assistant_message=raw_msg,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                finish_reason=finish_reason,
            )

        except self._openai.RateLimitError as e:
            raise RateLimitError("OpenAI", str(e))
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "rate limit" in err_str or "quota" in err_str:
                raise RateLimitError("OpenAI", str(e))
            raise

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return tool_result(tool_call_id, result)
