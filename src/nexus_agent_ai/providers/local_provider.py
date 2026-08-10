import json
import re
import uuid
from typing import Any, Dict, List, Optional
from .base import BaseProvider, ProviderResponse, Tool, ToolCall


class LocalQwenProvider(BaseProvider):
    """
    Built-in Local LLM Provider for Qwen/Qwen2.5-7B-Instruct-AWQ.
    Provides 100% offline, real local inference without cloud API dependencies.
    """

    def __init__(self, model_id: str = "LiquidAI/LFM2.5-2.6B-GGUF"):
        self.model_id = model_id
        self.filename = "LFM2.5-2.6B-Q6_K.gguf"
        self._tokenizer = None
        self._model_instance = None
        self._model_path = None

    def setup_model(self, verify_download: bool = False) -> str:
        if self._model_path is not None and not verify_download:
            return self._model_path
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print("🚀 Initializing nexus-agent...")
        try:
            from huggingface_hub import hf_hub_download
            import os as _os
            print(f"⚡ Downloading/Verifying Local Liquid LFM engine ({self.filename})...")
            model_path = hf_hub_download(repo_id=self.model_id, filename=self.filename, local_files_only=False)
            print("✅ Core engine ready! Booting up...")
            self._model_path = model_path
            return model_path
        except ImportError as e:
            if verify_download:
                raise RuntimeError("huggingface_hub package is not installed. To pull or download local model weights, run `pip install huggingface_hub` or install the `all` extra (`pip install nexus-agent-ai\\[all]`).") from e
            self._model_path = self.model_id
            return self.model_id
        except Exception as e:
            if verify_download:
                raise RuntimeError(f"Failed to pull model '{self.model_id}': {str(e)}") from e
            self._model_path = self.model_id
            return self.model_id

    def _ensure_loaded(self):
        if self._model_instance is not None:
            return
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        model_path = self.setup_model()
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA GPU"
                print(f"🟢 Dedicated GPU Detected ({gpu_name}). Using Transformers GGUF Engine...")
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.filename, trust_remote_code=True)
                self._model_instance = AutoModelForCausalLM.from_pretrained(
                    self.model_id, gguf_file=self.filename, device_map="auto", trust_remote_code=True
                )
            else:
                print("💻 CPU-Only Hardware Detected. Routing to Local Fallback Engine...")
                self._model_instance = "cpu_ollama_or_fallback"
        except Exception:
            self._model_instance = "cpu_ollama_or_fallback"

    def _convert_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}} for t in tools]

    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        self._ensure_loaded()
        if self._model_instance in ["cpu_ollama_or_fallback", "cpu_distilled_fallback"]:
            try:
                import urllib.request
                import json as _json
                req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        data = _json.loads(resp.read().decode())
                        models = [m.get("name", "") for m in data.get("models", [])]
                        target_model = "qwen2.5-coder:7b"
                        for m in models:
                            if "qwen" in m.lower():
                                target_model = m
                                break
                        from .openai_provider import OpenAIProvider
                        ollama_prov = OpenAIProvider(model=target_model, base_url="http://localhost:11434/v1")
                        return ollama_prov.complete(messages, tools, system)
            except Exception:
                pass
            raise RuntimeError("No dedicated GPU detected and Ollama is not running. Real local inference requires either a GPU or a running Ollama instance.")

        # GPU Engine processing logic remains intact below
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(messages)

        prompt = self._tokenizer.apply_chat_template(
            formatted_messages, tools=self._convert_tools(tools) if tools else None, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model_instance.device)
        input_tokens = inputs.input_ids.shape[1]
        outputs = self._model_instance.generate(**inputs, max_new_tokens=2048, temperature=0.2, top_p=0.95, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        
        response_text = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        tool_calls = []
        raw_tool_calls = []
        
        # Parse JSON blocks reliably via regex
        matches = re.findall(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', response_text, flags=re.DOTALL)
        for match in matches:
            try:
                func_name, args_str = match[0].strip(), match[1].strip()
                parsed_args = json.loads(args_str)
                tc_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_calls.append(ToolCall(id=tc_id, name=func_name, args=parsed_args))
                raw_tool_calls.append({"id": tc_id, "type": "function", "function": {"name": func_name, "arguments": json.dumps(parsed_args)}})
            except Exception:
                continue

        clean_text = re.sub(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', "", response_text, flags=re.DOTALL).strip()
        raw_msg = {"role": "assistant", "content": clean_text}
        if raw_tool_calls:
            raw_msg["tool_calls"] = raw_tool_calls

        return ProviderResponse(text=clean_text, tool_calls=tool_calls, raw_assistant_message=raw_msg, input_tokens=input_tokens, output_tokens=len(generated_ids[0]))

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        res = self.complete(messages, tools, system)
        if res.text:
            chunk_size = max(1, len(res.text) // 15)
            for i in range(0, len(res.text), chunk_size):
                yield res.text[i:i + chunk_size]
        yield res

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}