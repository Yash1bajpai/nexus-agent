import json
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
import urllib.parse

import typer
from . import display
from ..utils.config import DEFAULT_PROVIDER
from ..agent.memory import ConversationMemory
from ..agent.core import Agent
from .app import get_provider_instance, commit

# Global memory per web session
_memory = ConversationMemory()

class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            html_path = Path(__file__).parent.parent / "web" / "index.html"
            if not html_path.exists():
                self.send_error(404, "Frontend index.html not found")
                return
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
        elif path == "/api/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready", "version": "2.2.1"}).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(raw_body)
            except Exception:
                data = {}

            query = data.get("query", "").strip()
            provider_name = data.get("provider", DEFAULT_PROVIDER)

            events: List[Dict[str, Any]] = []
            def on_event(evt: Dict[str, Any]):
                events.append(evt)

            try:
                prov, resolved_name = get_provider_instance(provider_name)
                from ..providers.mock_provider import MockProvider
                is_mock = isinstance(prov, MockProvider)

                lower_input = query.lower()
                processed_query = query

                if lower_input.startswith("review "):
                    file_to_rev = query[7:].strip().strip('"').strip("'")
                    if is_mock:
                        processed_query = f"Review {file_to_rev} and tell me if it's good code"
                    else:
                        try:
                            fpath = os.path.join(os.getcwd(), file_to_rev) if not os.path.isabs(file_to_rev) else file_to_rev
                            with open(fpath, "r", encoding="utf-8", errors="replace") as _f:
                                file_contents = _f.read()
                            processed_query = (
                                f"Here is the content of '{file_to_rev}':\n\n```\n{file_contents}\n```\n\n"
                                f"Please review this code. Identify bugs, bad practices, missing type hints, "
                                f"missing docstrings, security issues, and suggest improvements. Be concise."
                            )
                        except Exception as e:
                            processed_query = f"File not found: {file_to_rev}"
                elif lower_input.startswith("debug "):
                    parts = query[6:].strip().split("--error")
                    file_to_dbg = parts[0].strip().strip('"').strip("'")
                    err_msg = parts[1].strip().strip('"').strip("'") if len(parts) > 1 else "Error reported by user"
                    if is_mock:
                        processed_query = f"Debug {file_to_dbg} --error {err_msg}"
                    else:
                        try:
                            fpath = os.path.join(os.getcwd(), file_to_dbg) if not os.path.isabs(file_to_dbg) else file_to_dbg
                            with open(fpath, "r", encoding="utf-8", errors="replace") as _f:
                                file_contents = _f.read()
                            processed_query = (
                                f"Here is the content of '{file_to_dbg}':\n\n```\n{file_contents}\n```\n\n"
                                f"The user reports this error:\n{err_msg}\n\n"
                                f"Identify the root cause and provide the fixed version of the code."
                            )
                        except Exception:
                            processed_query = f"File not found: {file_to_dbg}"
                elif lower_input == "commit" or lower_input.startswith("commit"):
                    # For commit command in web UI
                    if is_mock:
                        from ..providers.mock_provider import COMMIT_FINAL_TEXT
                        events.append({"type": "response", "content": COMMIT_FINAL_TEXT, "tokens": 428, "cost": 0.0015})
                        self.send_json({"events": events, "status": "success"})
                        return

                agent = Agent(provider=prov, memory=_memory, verbose=False, event_callback=on_event)
                final_text = agent.run(processed_query, stream=False)

                self.send_json({"events": events, "status": "success"})
            except Exception as e:
                self.send_json({"error": str(e), "status": "error"}, code=500)
        else:
            self.send_error(404, "API Endpoint Not Found")

    def send_json(self, payload: Dict[str, Any], code: int = 200):
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress standard logging for clean console output
        pass

def run_web_server(port: int = 8000, host: str = "127.0.0.1"):
    server_address = (host, port)
    httpd = HTTPServer(server_address, WebHandler)
    typer.echo(f"\n  [OK] Nexus-Agent Web Dashboard running at: http://{host}:{port}")
    typer.echo("  Press Ctrl+C to stop the web server.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\n  Stopping web server...")
        httpd.server_close()
