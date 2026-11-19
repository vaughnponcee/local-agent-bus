"""Test doubles: an OpenAI-compatible HTTP server with scripted replies."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpenAI:
    """Serves POST /v1/chat/completions. `script` is a list of replies, each a
    dict like {"content": "..."} or {"tool_calls": [("fs_read", {"path": "a"})]}
    or {"status": 500}. Records every request body in `.requests`."""

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.requests: list[dict] = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                step = fake.script.pop(0) if fake.script else {"content": "(script exhausted)"}
                if "status" in step:
                    self.send_response(step["status"])
                    self.end_headers()
                    self.wfile.write(b'{"error": "boom"}')
                    return
                msg = {"role": "assistant", "content": step.get("content")}
                if step.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"id": f"call_{i}", "type": "function",
                         "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
                        for i, (n, a) in enumerate(step["tool_calls"])]
                resp = {"id": "x", "object": "chat.completion", "model": body["model"],
                        "choices": [{"index": 0, "message": msg,
                                     "finish_reason": "tool_calls" if step.get("tool_calls") else "stop"}],
                        "usage": step.get("usage", {"prompt_tokens": 100, "completion_tokens": 20})}
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
