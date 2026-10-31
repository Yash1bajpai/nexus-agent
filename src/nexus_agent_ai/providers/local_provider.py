import atexit
import json
import os
import platform
import subprocess
import sys
import uuid
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from .base import BaseProvider, ProviderResponse, Tool
from .capabilities import resolve_limits
from .messages import assistant_message, decode_native_tool_call, extract_json_tool_calls, openai_messages, tool_result

# llama-server binary URLs per platform (llama.cpp b7075 release)
_LLAMA_RELEASE = "b7075"
_LLAMA_BASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{_LLAMA_RELEASE}"

# Official SHA-256 digests of the b7075 release ZIPs (from the GitHub release
# API). The download stays fail-closed: it is only executed if the digest
# matches. NEXUS_AGENT_LLAMA_SERVER_SHA256 overrides this table (e.g. for a
# new llama.cpp release after a manual review).
_LLAMA_SERVER_SHA256 = {
    "llama-b7075-bin-win-cpu-x64.zip": "ee57177a13347b386e8c097908fbb7f37c814f8a9c2948aef64f843a93221f13",
    "llama-b7075-bin-win-cpu-arm64.zip": "d01596413d704cbc66db01b1f686133d22749664d8516e9d56d6c5d540e5b2a4",
    "llama-b7075-bin-macos-arm64.zip": "d13d0654776e3e9e17ee77410d8622a1ca9858c5c3c0dbf96330b74d7332dd51",
    "llama-b7075-bin-macos-x64.zip": "4adb027ebc5508d899c2fd6ade65fb7b74aaeef14a898aadc146b320c504d744",
    # NOTE: b7075 publishes the Linux build as "ubuntu-x64" (there is no linux-x64 asset)
    "llama-b7075-bin-ubuntu-x64.zip": "eda853db069c545218eefb24afa126b557a5545ce19631e2f2a42ee7bce407d6",
}

def _get_llama_server_info() -> tuple:
    """Return (download_url, exe_name) for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        if machine in ("arm64", "aarch64"):
            return f"{_LLAMA_BASE_URL}/llama-{_LLAMA_RELEASE}-bin-win-cpu-arm64.zip", "llama-server.exe"
        return f"{_LLAMA_BASE_URL}/llama-{_LLAMA_RELEASE}-bin-win-cpu-x64.zip", "llama-server.exe"
    elif system == "darwin":
        if machine == "arm64":
            return f"{_LLAMA_BASE_URL}/llama-{_LLAMA_RELEASE}-bin-macos-arm64.zip", "llama-server"
        return f"{_LLAMA_BASE_URL}/llama-{_LLAMA_RELEASE}-bin-macos-x64.zip", "llama-server"
    else:  # Linux (asset is named ubuntu-x64 in the b7075 release)
        return f"{_LLAMA_BASE_URL}/llama-{_LLAMA_RELEASE}-bin-ubuntu-x64.zip", "llama-server"

_NEXUS_HOME = Path.home() / ".nexus-agent"
_SERVER_DIR = _NEXUS_HOME / "llama-server"
_DEFAULT_REPO = "LiquidAI/LFM2.5-2.6B-GGUF"
_DEFAULT_FILENAME = "LFM2.5-2.6B-Q6_K.gguf"

_ACTIVE_SERVER_PROCS: Set[subprocess.Popen] = set()

def _terminate_proc(proc: Optional[subprocess.Popen]):
    """Terminate and forcefully kill a subprocess if it does not exit promptly."""
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except (subprocess.TimeoutExpired, Exception):
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

def _cleanup_all_servers():
    """Ensure all spawned llama-server processes are terminated on interpreter exit."""
    for proc in list(_ACTIVE_SERVER_PROCS):
        _terminate_proc(proc)
    _ACTIVE_SERVER_PROCS.clear()

atexit.register(_cleanup_all_servers)


def ensure_llama_server_binary(verbose: bool = True) -> Optional[str]:
    """Pre-install the llama-server inference engine for the current platform.

    Called during onboarding / pull-model and before every local-provider
    session so the engine is ready before the first chat instead of
    downloading lazily mid-query. No-op (returns None) when a binary already
    exists, on ARM Linux where prebuilts cannot run, or when the download
    fails — never raises.
    """
    existing = _find_llama_server_exe()
    if existing:
        if verbose:
            print("✅ llama-server engine already installed.")
        return existing
    if system_machine_is_arm_linux():
        if verbose:
            print("ℹ️ ARM Linux detected — prebuilt llama-server cannot run here.")
            print("   Build it natively: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md")
        return None
    try:
        prov = LocalProvider()
        path = prov._download_llama_server()
        if path:
            # Download progress prints regardless of verbose — keep the final
            # status visible too so quiet pre-flights still explain the wait.
            print(f"✅ Inference engine installed at: {path}")
        return path
    except Exception as e:
        print(f"⚠️ llama-server pre-install skipped (will retry on first use): {e}")
        return None


def _startup_timeout_s(model_gb: float) -> int:
    """llama-server readiness budget: 90s floor, +45s per GB of model, 300s cap."""
    return int(min(300, max(90, 60 + 45 * model_gb)))


def system_machine_is_arm_linux() -> bool:
    """True on ARM/aarch64 Linux (e.g. Termux, Raspberry Pi) where prebuilt x86_64 llama-server binaries cannot run."""
    return platform.system() == "Linux" and platform.machine().lower() in ("aarch64", "arm64", "armv7l", "armv8l")


def _find_llama_server_exe() -> Optional[str]:
    """Find llama-server binary in the extracted directory.

    Handles both flat layouts (Windows zips) and nested ones — the ubuntu-x64
    release nests everything under build/bin/, and the binary must stay there:
    its libggml-cpu-*.so siblings resolve relative to the executable.
    """
    _, exe_name = _get_llama_server_info()
    # Check flat layout (files directly in _SERVER_DIR)
    flat = _SERVER_DIR / exe_name
    if flat.is_file():
        return str(flat)
    # Recursive search for nested layouts (any depth)
    if _SERVER_DIR.is_dir():
        for candidate in _SERVER_DIR.rglob(exe_name):
            if candidate.is_file():
                return str(candidate)
    return None


class LocalProvider(BaseProvider):
    """
    Built-in Local LLM Provider for LiquidAI/LFM2.5-2.6B-GGUF.
    Provides 100% offline, real local inference without cloud API dependencies.

    Inference paths (tried in order):
      1. GPU + torch + transformers (fastest)
      2. llama-cpp-python (Python binding, needs C++ build)
      3. llama-server subprocess (auto-downloaded binary, works on CPU — best for Windows)
      4. Ollama (external HTTP API, must be pre-installed)
    """
    is_local = True


    def __init__(self, model_id: str = None, filename: str = None, context_size: Optional[int] = None, max_output_tokens: Optional[int] = None):
        """Create a local provider for any HuggingFace GGUF repo.

        Priority: explicit args > NEXUS_AGENT_MODEL_REPO / NEXUS_AGENT_MODEL_FILENAME
        env vars > built-in LiquidAI LFM2.5-2.6B defaults.
        """
        self.model_id = model_id or os.getenv("NEXUS_AGENT_MODEL_REPO", _DEFAULT_REPO)
        self._explicit_context_size = context_size is not None
        limits = resolve_limits("local", self.model_id, context_size, max_output_tokens, local=True)
        self.context_size = limits.context_size
        self.max_output_tokens = limits.max_output_tokens
        self.model = self.model_id
        self.filename = filename or os.getenv("NEXUS_AGENT_MODEL_FILENAME", _DEFAULT_FILENAME)
        self._tokenizer = None
        self._model_instance = None
        self._llama = None  # llama-cpp-python instance
        self._model_path = None
        self._server_proc = None  # llama-server subprocess
        self._server_port = None
        self._server_alias = f"nexus-{uuid.uuid4().hex}"
        self._server_log_path = None
        self._server_model_path = None
        self._server_context_size = None

    def setup_model(self, verify_download: bool = False) -> str:
        if self._model_path is not None and not verify_download:
            return self._model_path
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print("🚀 Initializing nexus-agent...")
        try:
            from ..utils import model_download
            print(f"⚡ Downloading/Verifying Local Liquid LFM engine ({self.filename})...")
            model_path = model_download.find_cached(self.model_id, self.filename)
            if verify_download or not model_path:
                model_path = model_download.download(
                    self.model_id, self.filename, progress=model_download.console_progress())
                sys.stderr.write("\n")
            model_path = os.path.realpath(model_path)
            print("✅ Core engine ready! Booting up...")
            self._model_path = model_path
            return model_path
        except Exception as e:
            if verify_download:
                raise RuntimeError(f"Failed to pull model '{self.model_id}': {str(e)}") from e
            self._model_path = self.model_id
            return self.model_id

    def _ensure_loaded(self):
        if self._model_instance is not None or self._llama is not None:
            return
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        model_path = self.setup_model()

        # Path 1: GPU + torch + transformers (fastest)
        try:
            import torch
            if torch.cuda.is_available():
                if self._explicit_context_size:
                    raise RuntimeError("--context-size currently supports llama-server and llama-cpp-python, not the Transformers engine.")
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA GPU"
                print(f"🟢 Dedicated GPU Detected ({gpu_name}). Using Transformers GGUF Engine...")
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, gguf_file=self.filename, trust_remote_code=True)
                self._model_instance = AutoModelForCausalLM.from_pretrained(
                    self.model_id, gguf_file=self.filename, device_map="auto", trust_remote_code=True
                )
                return
        except ImportError:
            pass

        # Path 2: llama-cpp-python (CPU GGUF inference — needs C++ build on Windows)
        try:
            from llama_cpp import Llama
            if isinstance(model_path, str) and os.path.isfile(model_path):
                print(f"💻 CPU Mode: Loading Liquid LFM 2.6B via llama-cpp-python...")
                self._llama = Llama(
                    model_path=model_path,
                    n_ctx=self.context_size,
                    n_threads=4,
                    verbose=False,
                )
                print("✅ Liquid LFM engine loaded on CPU!")
                return
        except ImportError:
            pass

        # Path 3: llama-server subprocess (auto-downloads pre-built binary — best for Windows CPU)
        if isinstance(model_path, str) and os.path.isfile(model_path):
            # Only this instance's process can serve its selected model/context.
            if system_machine_is_arm_linux() and not _find_llama_server_exe():
                print("⚠️ No native llama-server found for ARM Linux (aarch64).")
                print("   The auto-downloader only ships x86_64 binaries.")
                print("   Build llama.cpp locally or install Ollama, then retry:")
                print("     https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md")
                raise RuntimeError(
                    "Local inference on ARM Linux requires a natively built llama-server "
                    "(or Ollama). See https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md"
                )
            try:
                self._start_llama_server(model_path)
                return
            except Exception as e:
                print(f"⚠️ llama-server failed: {e}")
                print(f"   Log: {self._server_log_path}")

        # Path 4: Ollama (external HTTP API)
        if self._explicit_context_size:
            raise RuntimeError("--context-size requires llama-server or llama-cpp-python; it cannot be applied to the Ollama fallback.")
        print("💻 CPU Mode: llama-server unavailable. Checking Ollama...")
        print("   If you don't have Ollama, install it from: https://ollama.com")
        self._model_instance = "cpu_ollama_or_fallback"

    # ── llama-server subprocess management ────────────────────────────────

    def _download_llama_server(self) -> str:
        """Download and extract the pre-built llama-server binary."""
        existing = _find_llama_server_exe()
        if existing:
            return existing

        _SERVER_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = _SERVER_DIR / "llama-server.zip"
        bin_url, exe_name = _get_llama_server_info()
        asset_name = bin_url.rsplit("/", 1)[-1]
        # Env var override first, then the pinned official digest table.
        expected_sha256 = os.getenv("NEXUS_AGENT_LLAMA_SERVER_SHA256", "").strip().lower()
        if not (expected_sha256 and len(expected_sha256) == 64 and all(c in "0123456789abcdef" for c in expected_sha256)):
            expected_sha256 = _LLAMA_SERVER_SHA256.get(asset_name, "")
        if not expected_sha256:
            raise RuntimeError(
                f"No pinned SHA-256 digest for {asset_name}. "
                "Set NEXUS_AGENT_LLAMA_SERVER_SHA256 to the official 64-character digest "
                "to allow this download."
            )

        if not zip_path.is_file():
            print(f"⬇️  Downloading llama-server (~50 MB) for CPU inference...")
            import urllib.request
            try:
                # Download to a temp name and rename only when complete, so an interrupted
                # download can never be mistaken for the finished file on the next run.
                tmp_zip = zip_path.with_name(zip_path.name + ".part")
                urllib.request.urlretrieve(bin_url, str(tmp_zip))
                os.replace(tmp_zip, zip_path)
            except Exception as dl_err:
                # Clean up partial download
                zip_path.unlink(missing_ok=True)
                zip_path.with_name(zip_path.name + ".part").unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to download llama-server from:\n  {bin_url}\n"
                    f"Error: {dl_err}\n\n"
                    f"This usually happens due to:\n"
                    f"  - College/corporate firewall blocking GitHub downloads\n"
                    f"  - No internet connection\n\n"
                    f"Alternatives:\n"
                    f"  1. Download manually from: {bin_url}\n"
                    f"     Extract to: {_SERVER_DIR}\n"
                    f"  2. Install Ollama instead: https://ollama.com\n"
                    f"  3. Use a cloud provider: -p openrouter (free tier available)"
                ) from dl_err
            print("✅ Download complete.")

        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError("llama-server download checksum mismatch; refusing to execute it.")

        print("📦 Extracting llama-server...")
        import zipfile
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            destination = _SERVER_DIR.resolve()
            for member in zf.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination):
                    raise RuntimeError(f"Unsafe archive member path: {member.filename}")
            zf.extractall(str(destination))

        exe_path = _find_llama_server_exe()
        if not exe_path:
            raise RuntimeError(f"llama-server binary not found in {_SERVER_DIR} after extraction.")

        # Make executable on Linux/macOS
        if sys.platform != "win32":
            os.chmod(exe_path, 0o755)

        # Clean up zip to save space
        zip_path.unlink(missing_ok=True)
        return exe_path

    def _start_llama_server(self, model_path: str):
        """Download llama-server if needed and start it with the GGUF model."""
        model_path = str(Path(model_path).resolve())
        if (self._server_model_path == model_path
                and self._server_context_size == self.context_size
                and self._is_server_alive()):
            self._model_instance = "llama_server"
            return
        self.stop()
        exe_path = self._download_llama_server()
        import socket
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self._server_port = reservation.getsockname()[1]
        port = self._server_port
        _SERVER_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _SERVER_DIR / f"server-{self._server_alias}.log"
        self._server_log_path = log_path
        log_file = open(str(log_path), "w", encoding="utf-8")
        cmd = [
            exe_path,
            "-m", model_path,
            "--alias", self._server_alias,
            "-c", str(self.context_size),
            "--parallel", "1",
            "--port", str(port),
            "--host", "127.0.0.1",
            "-t", "4",  # threads
            "--jinja",  # required for tool calling support
            "--temp", "0.1",  # recommended by model card
            "--top-k", "50",  # recommended by model card
            "--repeat-penalty", "1.1",  # recommended by model card
        ]
        print(f"🔧 Starting llama-server on port {port}...")
        try:
            self._server_proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except BaseException:
            log_file.close()
            raise
        self._server_log = log_file
        self._server_model_path = model_path
        self._server_context_size = self.context_size
        _ACTIVE_SERVER_PROCS.add(self._server_proc)

        # Wait for server to be ready. Model loading is CPU/IO bound and can
        # take well over a minute for ~2 GB quant files on slow laptop CPUs,
        # so scale the wait budget with model size instead of a fixed 30s.
        try:
            model_gb = os.path.getsize(model_path) / (1024 ** 3)
        except OSError:
            model_gb = 2.0
        timeout_s = _startup_timeout_s(model_gb)
        print(f"   Loading model ({model_gb:.1f} GB) — waiting up to {timeout_s}s...")
        import time
        deadline = time.monotonic() + timeout_s
        try:
            while time.monotonic() < deadline:
                time.sleep(1.0)
                if self._is_server_alive():
                    print(f"✅ llama-server ready on http://127.0.0.1:{port}")
                    self._model_instance = "llama_server"
                    return
                if self._server_proc.poll() is not None:
                    log_tail = ""
                    try:
                        if log_path.is_file():
                            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"llama-server exited immediately (exit code {self._server_proc.returncode}).\n"
                        f"Server log:\n{log_tail.strip()}"
                    )
            raise RuntimeError(
                f"llama-server did not become ready within {timeout_s} seconds.\n"
                f"Check the log at: {log_path}\n"
                f"Another program may have taken port {port}, antivirus may have blocked "
                "the binary, or the model file may be corrupted."
            )
        except BaseException:
            self.stop()
            raise

    def _is_server_alive(self) -> bool:
        """Check health and identity only for the process owned by this instance."""
        if self._server_proc is None or self._server_proc.poll() is not None:
            return False
        import urllib.request
        try:
            base = f"http://127.0.0.1:{self._server_port}"
            with urllib.request.urlopen(base + "/health", timeout=2.0) as resp:
                if resp.status != 200:
                    return False
            with urllib.request.urlopen(base + "/v1/models", timeout=2.0) as resp:
                models = json.loads(resp.read()).get("data", [])
            return any(model.get("id") == self._server_alias for model in models)
        except Exception:
            return False

    def _run_via_server(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        for chunk in self._stream_via_server(messages, tools, system):
            if isinstance(chunk, ProviderResponse):
                return chunk
        raise RuntimeError("Local server ended without a final response")

    def _stream_via_server(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str):
        """Yield SSE text immediately, then the assembled tool calls and usage."""
        import urllib.request

        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(openai_messages(messages))

        payload = {
            "model": self._server_alias,
            "messages": formatted_messages,
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": self.max_output_tokens,
            "stream": True,
            # Ask the server to include real token counts in the final chunk
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self._server_port}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        raw_tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
        raw_tool_call_order: List[str] = []
        usage: Dict[str, Any] = {}
        finish_reason = None
        native_error = None

        import urllib.error
        try:
            response = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"Local model request failed (HTTP {exc.code}): {detail}. "
                               "If context is exceeded, shorten the input/history or restart with --context-size <tokens> or increase NEXUS_CONTEXT_SIZE within your model/RAM limits. In REPL, /clear clears history.") from exc
        with response as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk_str = line[len("data:"):].strip()
                if chunk_str == "[DONE]":
                    break
                if not chunk_str:
                    continue
                try:
                    chunk = json.loads(chunk_str)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                reason = choices[0].get("finish_reason")
                if isinstance(reason, str):
                    finish_reason = reason

                content_delta = delta.get("content")
                if content_delta:
                    text_parts.append(content_delta)
                    yield content_delta

                reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)

                for rtc in delta.get("tool_calls") or []:
                    # Key strictly by index: the id may only appear in the first
                    # chunk, while name/arguments arrive as later fragments.
                    idx = rtc.get("index", 0)
                    key = f"idx_{idx}"
                    if key not in raw_tool_calls_by_id:
                        raw_tool_calls_by_id[key] = {"id": f"call_{uuid.uuid4().hex}", "name": "", "arguments": ""}
                        raw_tool_call_order.append(key)
                    entry = raw_tool_calls_by_id[key]
                    if rtc.get("id"):
                        entry["id"] = rtc["id"]
                    func = rtc.get("function", {}) or {}
                    name = func.get("name")
                    arguments = func.get("arguments")
                    if name is not None:
                        if not isinstance(name, str):
                            native_error = "Native tool call function name must be a string"
                        else:
                            entry["name"] += name
                    if arguments is not None:
                        if not isinstance(arguments, str):
                            native_error = "Native tool call argument fragments must be JSON text"
                        else:
                            entry["arguments"] += arguments

        text = "".join(text_parts)
        # Reasoning models (LFM2.5 etc.) may emit only thinking tokens when the
        # answer is cut short. Never leak the full chain-of-thought as the
        # answer — surface only the last line, which is usually the answer.
        if not text and reasoning_parts:
            lines = [l.strip() for l in "".join(reasoning_parts).splitlines() if l.strip()]
            text = lines[-1] if lines else ""

        tool_calls = []
        if finish_reason != "length":
            if native_error:
                raise ValueError(native_error)
            for key in raw_tool_call_order:
                entry = raw_tool_calls_by_id[key]
                tool_calls.append(decode_native_tool_call(entry["name"], entry["arguments"], entry["id"]))

            if not tool_calls:
                text, tool_calls = extract_json_tool_calls(text)
        raw_msg = assistant_message(text, tool_calls)

        yield ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            raw_assistant_message=raw_msg,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=finish_reason,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def _convert_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}} for t in tools]

    def complete(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        self._ensure_loaded()

        # llama-server subprocess path
        if self._model_instance == "llama_server":
            return self._run_via_server(messages, tools, system)

        # llama-cpp-python CPU inference path
        if self._llama is not None:
            return self._run_llama_cpp(messages, tools, system)

        # Ollama fallback path
        if self._model_instance in ["cpu_ollama_or_fallback", "cpu_distilled_fallback"]:
            try:
                import urllib.request
                req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        models = [m.get("name", "") for m in data.get("models", [])]
                        target_model = os.getenv("LOCAL_MODEL", "llama3.2:3b")
                        # Auto-detect Liquid LFM or any coding model
                        for m in models:
                            if "liquid" in m.lower() or "lfm" in m.lower():
                                target_model = m
                                break
                        from .openai_provider import OpenAIProvider
                        ollama_prov = OpenAIProvider(
                            model=target_model, base_url="http://localhost:11434/v1",
                            context_size=self.context_size, max_output_tokens=self.max_output_tokens,
                        )
                        return ollama_prov.complete(messages, tools, system)
            except Exception:
                pass
            raise RuntimeError(
                "Local model inference unavailable - no engine could start. "
                "Options: (A) install Ollama from https://ollama.com then retry with: "
                "nexus-agent -p ollama \"your question\"; "
                "(B) use a free cloud key from https://openrouter.ai (set OPENROUTER_API_KEY, then -p openrouter); "
                "(C) check the per-instance logs at ~/.nexus-agent/llama-server/server-nexus-*.log"
            )

        # GPU Engine processing logic (torch + transformers)
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(openai_messages(messages))

        prompt = self._tokenizer.apply_chat_template(
            formatted_messages, tools=self._convert_tools(tools) if tools else None, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model_instance.device)
        input_tokens = inputs.input_ids.shape[1]
        outputs = self._model_instance.generate(**inputs, max_new_tokens=self.max_output_tokens, temperature=0.2, top_p=0.95, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        
        response_text = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        clean_text, tool_calls = extract_json_tool_calls(response_text)
        raw_msg = assistant_message(clean_text, tool_calls)
        # generate() returns token IDs, not a reliable stopping reason; leave it unobserved.

        return ProviderResponse(text=clean_text, tool_calls=tool_calls, raw_assistant_message=raw_msg, input_tokens=input_tokens, output_tokens=len(generated_ids[0]))

    def _run_llama_cpp(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> ProviderResponse:
        """CPU inference via llama-cpp-python."""
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(openai_messages(messages))

        kwargs = {
            "messages": formatted_messages,
            "max_tokens": self.max_output_tokens,
            "temperature": 0.2,
            "top_p": 0.95,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            kwargs["tool_choice"] = "auto"

        response = self._llama.create_chat_completion(**kwargs)

        # Parse response
        choice = response["choices"][0]
        msg = choice.get("message", {})
        text = msg.get("content", "") or ""
        raw_tool_calls = msg.get("tool_calls", []) or []
        finish_reason = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None

        tool_calls = []
        if finish_reason != "length":
            for rtc in raw_tool_calls:
                func = rtc.get("function", {})
                tool_calls.append(decode_native_tool_call(
                    func.get("name"), func.get("arguments", ""), rtc.get("id"),
                ))

            if not tool_calls:
                text, tool_calls = extract_json_tool_calls(text)
        raw_msg = assistant_message(text, tool_calls)

        usage = response.get("usage", {})
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            raw_assistant_message=raw_msg,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=finish_reason,
        )

    def stream(self, messages: List[Dict[str, Any]], tools: List[Tool], system: str) -> Any:
        self._ensure_loaded()
        if self._model_instance == "llama_server":
            yield from self._stream_via_server(messages, tools, system)
        else:
            # Optional engines without a streaming implementation return only
            # the final response; never simulate token delivery with slices.
            yield self.complete(messages, tools, system)

    def format_tool_result_message(self, tool_call_id: str, result: str) -> Dict[str, Any]:
        return tool_result(tool_call_id, result)

    def stop(self):
        """Explicitly shut down the llama-server subprocess and release resources."""
        server_proc = getattr(self, "_server_proc", None)
        if server_proc is not None:
            _terminate_proc(server_proc)
            self._server_proc = None
            if server_proc in _ACTIVE_SERVER_PROCS:
                _ACTIVE_SERVER_PROCS.discard(server_proc)

        log_file = getattr(self, "_server_log", None)
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass
            self._server_log = None

        if getattr(self, "_model_instance", None) == "llama_server":
            self._model_instance = None
        self._server_model_path = None
        self._server_context_size = None

    def __del__(self):
        """Clean up llama-server subprocess on garbage collection."""
        try:
            self.stop()
        except Exception:
            pass
