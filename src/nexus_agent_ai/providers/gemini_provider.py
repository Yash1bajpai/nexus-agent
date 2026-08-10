import uuid
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
from typing import Any, Dict, List
# pyrefly: ignore [missing-import]
from .base import BaseProvider, ProviderResponse, Tool, ToolCall, RateLimitError
# pyrefly: ignore [missing-import]
from ..utils.config import get_env_or_raise

class GeminiProvider(BaseProvider):
    """LLM Provider implementation for Google Gemini models via google-genai SDK."""

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        api_key = get_env_or_raise("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        formatted = []
        for msg in messages:
            if not isinstance(msg, dict):
                formatted.append(msg)
            elif "parts" in msg:
                msg_copy = dict(msg)
                if msg_copy.get("role") == "tool":
                    msg_copy["role"] = "user"
                formatted.append(msg_copy)
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "assistant":
                    role = "model"
                elif role == "tool":
                    role = "user"
                    tool_name = msg.get("name") or msg.get("tool_call_id", "unknown_tool")
                    if "_" in tool_name and not tool_name.startswith("call_"):
                        tool_name = tool_name.rsplit("_", 1)[0]
                    formatted.append({
                        "role": "user",
                        "parts": [{
                            "function_response": {
                                "name": tool_name,
                                "response": {"result": content},
                            }
                        }]
                    })
                    continue
                elif role == "system":
                    continue
                formatted.append({"role": role, "parts": [{"text": content}]})
        return formatted

    def _convert_tools(self, tools: List[Tool]) -> Any:
        if not tools:
            return None
        declarations = []
        for t in tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.input_schema,
                )
            )
        return [types.Tool(function_declarations=declarations)]

    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        genai_tools = self._convert_tools(tools)

        config = types.GenerateContentConfig(
            system_instruction=system if system else None,
            tools=genai_tools if genai_tools else None,
        )

        gemini_contents = self._format_messages(messages)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=gemini_contents,
                config=config,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "resource_exhausted" in err_str or "rate limit" in err_str or "quota" in err_str:
                raise RateLimitError("Gemini", str(e))
            raise

        text = ""
        tool_calls = []

        candidate = response.candidates[0]
        parts = None
        if candidate.content and candidate.content.parts:
            parts = candidate.content.parts

        if parts:
            for part in parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"{fc.name}_{uuid.uuid4().hex[:8]}",
                            name=fc.name,
                            args=dict(fc.args) if fc.args else {},
                        )
                    )

        if not text and not tool_calls:
            try:
                fallback = response.text
                if fallback:
                    text = fallback
            except Exception:
                pass

        raw_msg = candidate.content

        in_tokens = 0
        out_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            in_tokens = int(getattr(response.usage_metadata, "prompt_token_count", 0) or 0)
            out_tokens = int(getattr(response.usage_metadata, "candidates_token_count", 0) or 0)

        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            raw_assistant_message=raw_msg,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        genai_tools = self._convert_tools(tools)
        config = types.GenerateContentConfig(
            system_instruction=system if system else None,
            tools=genai_tools if genai_tools else None,
        )
        gemini_contents = self._format_messages(messages)
        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=gemini_contents,
                config=config,
            )
            full_text = []
            tool_calls = []
            last_chunk = None
            for chunk in response_stream:
                last_chunk = chunk
                if hasattr(chunk, "text") and chunk.text:
                    full_text.append(chunk.text)
                    yield chunk.text
                if hasattr(chunk, "candidates") and chunk.candidates:
                    for candidate in chunk.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, "function_call") and part.function_call:
                                    fc = part.function_call
                                    tool_calls.append(
                                        ToolCall(
                                            id=f"{fc.name}_{uuid.uuid4().hex[:8]}",
                                            name=fc.name,
                                            args=dict(fc.args) if fc.args else {},
                                        )
                                    )

            in_tokens = 0
            out_tokens = 0
            if last_chunk and hasattr(last_chunk, "usage_metadata") and last_chunk.usage_metadata:
                in_tokens = int(getattr(last_chunk.usage_metadata, "prompt_token_count", 0) or 0)
                out_tokens = int(getattr(last_chunk.usage_metadata, "candidates_token_count", 0) or 0)

            raw_msg = {"role": "model", "parts": [{"text": "".join(full_text)}]}
            yield ProviderResponse(
                text="".join(full_text),
                tool_calls=tool_calls,
                raw_assistant_message=raw_msg,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "resource_exhausted" in err_str or "rate limit" in err_str or "quota" in err_str:
                raise RateLimitError("Gemini", str(e))
            raise

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        func_name = tool_call_id.rsplit("_", 1)[0] if ("_" in tool_call_id and not tool_call_id.startswith("call_")) else tool_call_id
        return {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": func_name,
                        "response": {"result": result},
                    }
                }
            ],
        }
