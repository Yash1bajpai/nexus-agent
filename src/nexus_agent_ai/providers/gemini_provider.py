from typing import Any, Dict, List, Optional
# pyrefly: ignore [missing-import]
from .base import BaseProvider, ProviderResponse, Tool, RateLimitError
# pyrefly: ignore [missing-import]
from ..utils.config import get_env_or_raise
from .messages import assistant_message, decode_native_tool_call, gemini_messages, tool_result

def _finish_reason(raw: Any) -> Optional[str]:
    """Report Gemini's token-limit stops with the same "length" value other providers use."""
    if raw is None:
        return None
    name = getattr(raw, "name", None)
    text = (name if isinstance(name, str) else str(raw)).rsplit(".", 1)[-1].upper()
    if text == "MAX_TOKENS":
        return "length"
    return text.lower() or None


class GeminiProvider(BaseProvider):
    """LLM Provider implementation for Google Gemini models via google-genai SDK."""

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise RuntimeError("Gemini provider requires the 'google-genai' package.") from e
        self._genai = genai
        self._types = types
        api_key = get_env_or_raise("GEMINI_API_KEY")
        self.client = self._genai.Client(api_key=api_key)
        self.model = model

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        return gemini_messages(messages)

    def _convert_tools(self, tools: List[Tool]) -> Any:
        if not tools:
            return None
        declarations = []
        for t in tools:
            declarations.append(
                self._types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=t.input_schema,
                )
            )
        return [self._types.Tool(function_declarations=declarations)]

    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        genai_tools = self._convert_tools(tools)

        config = self._types.GenerateContentConfig(
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

        if not getattr(response, "candidates", None):
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None) or "empty response"
            raise RuntimeError(f"Gemini returned no candidates ({reason}).")
        candidate = response.candidates[0]
        finish_reason = _finish_reason(getattr(candidate, "finish_reason", None))
        parts = None
        if candidate.content and candidate.content.parts:
            parts = candidate.content.parts

        if parts:
            for part in parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
                if finish_reason != "length" and hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    # The SDK represents an omitted parameterless args field as None.
                    arguments = getattr(fc, "args", None)
                    tool_calls.append(
                        decode_native_tool_call(
                            fc.name, {} if arguments is None else arguments,
                            fc.id if isinstance(getattr(fc, "id", None), str) else None,
                        )
                    )

        if not text and not tool_calls:
            try:
                fallback = response.text
                if fallback:
                    text = fallback
            except Exception:
                pass

        raw_msg = assistant_message(text, tool_calls)

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
            finish_reason=finish_reason,
        )

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        genai_tools = self._convert_tools(tools)
        config = self._types.GenerateContentConfig(
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
            pending_calls = []
            last_chunk = None
            finish_reason = None
            for chunk in response_stream:
                last_chunk = chunk
                if hasattr(chunk, "text") and chunk.text:
                    full_text.append(chunk.text)
                    yield chunk.text
                if hasattr(chunk, "candidates") and chunk.candidates:
                    for candidate in chunk.candidates:
                        reason = _finish_reason(getattr(candidate, "finish_reason", None))
                        if reason is not None and finish_reason != "length":
                            finish_reason = reason
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, "function_call") and part.function_call:
                                    pending_calls.append(part.function_call)

            # A later chunk can report a cutoff after an incomplete function call.
            # Validate only after the final status is known, never on partial args.
            if finish_reason != "length":
                for fc in pending_calls:
                    arguments = getattr(fc, "args", None)
                    tool_calls.append(
                        decode_native_tool_call(
                            fc.name, {} if arguments is None else arguments,
                            fc.id if isinstance(getattr(fc, "id", None), str) else None,
                        )
                    )
            if not full_text and not tool_calls and finish_reason != "length":
                feedback = getattr(last_chunk, "prompt_feedback", None)
                reason = getattr(feedback, "block_reason", None) or "empty stream"
                raise RuntimeError(f"Gemini returned no usable content ({reason}).")
            in_tokens = 0
            out_tokens = 0
            if last_chunk and hasattr(last_chunk, "usage_metadata") and last_chunk.usage_metadata:
                in_tokens = int(getattr(last_chunk.usage_metadata, "prompt_token_count", 0) or 0)
                out_tokens = int(getattr(last_chunk.usage_metadata, "candidates_token_count", 0) or 0)

            raw_msg = assistant_message("".join(full_text), tool_calls)
            yield ProviderResponse(
                text="".join(full_text),
                tool_calls=tool_calls,
                raw_assistant_message=raw_msg,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                finish_reason=finish_reason,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "429" in str(e) or "resource_exhausted" in err_str or "rate limit" in err_str or "quota" in err_str:
                raise RateLimitError("Gemini", str(e))
            raise

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return tool_result(tool_call_id, result)
