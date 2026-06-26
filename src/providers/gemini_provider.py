import warnings
# Suppress google-generativeai deprecation warnings in CLI outputs
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

import google.generativeai as genai
from typing import Any, Dict, List
from src.providers.base import BaseProvider, ProviderResponse, Tool, ToolCall
from src.utils.config import get_env_or_raise

class GeminiProvider(BaseProvider):
    """LLM Provider implementation for Google Gemini models."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = get_env_or_raise("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = model

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        formatted = []
        for msg in messages:
            if not isinstance(msg, dict):
                # Protobuf Content object from previous assistant turn
                formatted.append(msg)
            elif "parts" in msg:
                # Structured function response ContentDict
                formatted.append(msg)
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "assistant":
                    role = "model"
                elif role == "system":
                    continue
                formatted.append({"role": role, "parts": [content]})
        return formatted

    def _convert_tools(self, tools: List[Tool]) -> Any:
        if not tools:
            return None
        declarations = []
        for t in tools:
            declarations.append(
                genai.types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.input_schema,
                )
            )
        return [genai.types.Tool(function_declarations=declarations)]

    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        genai_tools = self._convert_tools(tools)

        model = genai.GenerativeModel(
            model_name=self.model,
            tools=genai_tools,
            system_instruction=system if system else None,
        )

        gemini_contents = self._format_messages(messages)
        response = model.generate_content(gemini_contents)

        text = ""
        tool_calls = []

        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(
                    ToolCall(
                        id=fc.name,
                        name=fc.name,
                        args=dict(fc.args),
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

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str):
        genai_tools = self._convert_tools(tools)
        model = genai.GenerativeModel(
            model_name=self.model,
            tools=genai_tools,
            system_instruction=system if system else None,
        )
        gemini_contents = self._format_messages(messages)
        response = model.generate_content(gemini_contents, stream=True)
        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return {
            "role": "function",
            "parts": [
                {
                    "function_response": {
                        "name": tool_call_id,
                        "response": {"result": result},
                    }
                }
            ],
        }
