"""Minimal MCP client over stdio (newline-delimited JSON-RPC 2.0)."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading

PROTOCOL = "2025-06-18"


class McpError(RuntimeError):
    pass


class McpClient:
    def __init__(self, name: str, command: list[str], env: dict | None = None, timeout: float = 30.0):
        self.name, self.timeout = name, timeout
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                     env={**os.environ, **(env or {})}, bufsize=1)
        self._id = 0
        self._q: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self.tools: list[dict] = []
        info = self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                           "clientInfo": {"name": "agentbus", "version": "0.1.0"}})
        self.server_info = info.get("serverInfo", {})
        self.notify("notifications/initialized")
        self.tools = self.request("tools/list").get("tools", [])

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                try:
                    self._q.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self._q.put(None)

    def notify(self, method: str, params: dict | None = None):
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def _send(self, msg: dict):
        if self.proc.poll() is not None:
            raise McpError(f"{self.name}: server exited with {self.proc.returncode}")
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:
            try:
                msg = self._q.get(timeout=self.timeout)
            except queue.Empty:
                raise McpError(f"{self.name}: no reply to {method} within {self.timeout}s") from None
            if msg is None:
                raise McpError(f"{self.name}: server closed the connection")
            if msg.get("id") != rid:
                continue  # notifications or stale replies
            if "error" in msg:
                raise McpError(f"{self.name}: {msg['error'].get('message')}")
            return msg.get("result", {})

    def call(self, tool: str, arguments: dict) -> tuple[str, bool]:
        res = self.request("tools/call", {"name": tool, "arguments": arguments})
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        return text, bool(res.get("isError"))

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
