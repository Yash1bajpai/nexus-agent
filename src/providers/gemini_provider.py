from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
from typing import Any, Dict, List
# pyrefly: ignore [missing-import]
from .base import BaseProvider, ProviderResponse, Tool, ToolCall
# pyrefly: ignore [missing-import]
from ..utils.config import get_env_or_raise

class GeminiProvider(BaseProvider):
    """LLM Provider implementation for Google Gemini models via google-genai SDK."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = get_env_or_raise("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        formatted = []
        for msg in messages:
            if not isinstance(msg, dict):
                formatted.append(msg)
            elif "parts" in msg:
                formatted.append(msg)
            else:
                role = msg.get("role", "user")  
                content = msg.get("content", "")
                if role == "assistant":
                    role = "model"
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
        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=config,
        )

        text = ""
        tool_calls = []

        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=fc.name,
                            name=fc.name,
                            args=dict(fc.args) if fc.args else {},
                        )
                    )

        raw_msg = candidate.content

        in_tokens = 0
        out_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            in_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            out_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

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
        response_stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=gemini_contents,
            config=config,
        )
        for chunk in response_stream:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return {
            "role": "tool",
            "parts": [
                {
                    "function_response": {
                        "name": tool_call_id,
                        "response": {"result": result},
                    }
                }
            ],
        }
