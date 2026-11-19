"""Built-in tools: sandboxed filesystem + allow-listed shell, plus MCP proxies.

Every path is resolved and must stay inside the workspace (symlinks and
`..` are resolved before the check). Shell commands run without a shell
(no pipes, globs or redirection), from an allow-list, with a timeout and
an output cap.
"""
from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .mcp import McpClient


class ToolDenied(Exception):
    pass


@dataclass
class ToolResult:
    content: str
    ok: bool
    denied: bool = False
    meta: dict | None = None


BUILTIN_SCHEMAS = [
    {"name": "fs_list", "description": "List a directory inside the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}},
    {"name": "fs_read", "description": "Read a UTF-8 text file inside the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                     "max_chars": {"type": "integer"}}, "required": ["path"]}},
    {"name": "fs_write", "description": "Write a UTF-8 text file inside the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "shell", "description": "Run an allow-listed command (no shell syntax) in the workspace.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
]


class Toolbox:
    def __init__(self, cfg: Config, mcp_clients: list[McpClient] | None = None):
        self.cfg = cfg
        self.root = Path(cfg.workspace).resolve()
        self.mcp = {c.name: c for c in (mcp_clients or [])}
        self._mcp_tools: dict[str, tuple[McpClient, str]] = {}
        for c in self.mcp.values():
            for t in c.tools:
                self._mcp_tools[f"{c.name}__{t['name']}"] = (c, t["name"])

    def schemas(self) -> list[dict]:
        out = list(BUILTIN_SCHEMAS)
        for full, (client, name) in self._mcp_tools.items():
            t = next(t for t in client.tools if t["name"] == name)
            out.append({"name": full, "description": f"[{client.name}] {t.get('description', '')}",
                        "parameters": t.get("inputSchema", {"type": "object", "properties": {}})})
        return out

    # -------------------------------------------------------------- sandbox
    def _path(self, p: str | None) -> Path:
        target = (self.root / (p or ".")).resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolDenied(f"path escapes workspace: {p}")
        return target

    def _cap(self, s: str) -> tuple[str, bool]:
        cap = self.cfg.tool_output_cap
        if len(s) <= cap:
            return s, False
        return s[: cap - 200] + f"\n...[{len(s) - cap + 200} chars cut]...\n" + s[-200:], True

    # -------------------------------------------------------------- dispatch
    def call(self, name: str, args: dict) -> ToolResult:
        t0 = time.perf_counter()
        try:
            if name in self._mcp_tools:
                client, tool = self._mcp_tools[name]
                text, is_err = client.call(tool, args)
                text, cut = self._cap(text)
                res = ToolResult(text, not is_err, meta={"server": client.name, "truncated": cut})
            elif name == "fs_list":
                d = self._path(args.get("path"))
                entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                lines = [f"{'d' if e.is_dir() else 'f'} {e.name}" + ("" if e.is_dir() else f" {e.stat().st_size}")
                         for e in entries[:500]]
                res = ToolResult("\n".join(lines) or "(empty)", True, meta={"entries": len(entries)})
            elif name == "fs_read":
                f = self._path(args["path"])
                text = f.read_text(encoding="utf-8", errors="replace")
                limit = min(int(args.get("max_chars") or self.cfg.tool_output_cap), self.cfg.tool_output_cap)
                over = len(text) > limit
                text, cut = self._cap(text) if over else (text, False)
                res = ToolResult(text, True, meta={"bytes": f.stat().st_size, "truncated": over or cut})
            elif name == "fs_write":
                f = self._path(args["path"])
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(args["content"], encoding="utf-8")
                res = ToolResult(f"wrote {len(args['content'])} chars to {f.relative_to(self.root)}", True)
            elif name == "shell":
                argv = shlex.split(args["command"], posix=True)
                if not argv:
                    raise ToolDenied("empty command")
                exe = Path(argv[0]).name.lower().removesuffix(".exe")
                if exe not in [a.lower() for a in self.cfg.shell_allow]:
                    raise ToolDenied(f"command '{argv[0]}' is not in shell_allow {self.cfg.shell_allow}")
                for a in argv[1:]:  # path-like args must stay inside the workspace too
                    if a.startswith(("/", "\\")) or ".." in Path(a).parts or (len(a) > 1 and a[1] == ":"):
                        self._path(a)
                try:
                    p = subprocess.run(argv, cwd=self.root, capture_output=True, text=True,
                                       timeout=self.cfg.shell_timeout_s, encoding="utf-8", errors="replace")
                except FileNotFoundError:
                    return ToolResult(f"command not found: {argv[0]}", False)
                except subprocess.TimeoutExpired:
                    return ToolResult(f"timed out after {self.cfg.shell_timeout_s}s", False,
                                      meta={"timeout": True})
                out, cut = self._cap(p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr else ""))
                res = ToolResult(out or "(no output)", p.returncode == 0,
                                 meta={"exit": p.returncode, "truncated": cut})
            else:
                return ToolResult(f"unknown tool: {name}", False)
        except ToolDenied as e:
            return ToolResult(f"denied: {e}", False, denied=True)
        except (OSError, KeyError, ValueError) as e:
            return ToolResult(f"error: {type(e).__name__}: {e}", False)
        res.meta = {**(res.meta or {}), "ms": round((time.perf_counter() - t0) * 1000, 1)}
        return res
