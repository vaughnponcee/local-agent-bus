"""Configuration: models, VRAM budget, tool sandbox, MCP servers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SKILLS = ("code", "chat", "summarize", "reason", "tools")


@dataclass
class ModelSpec:
    name: str
    endpoint: str                 # OpenAI-compatible base URL, e.g. http://127.0.0.1:8080/v1
    model: str                    # id sent in the request body
    context_tokens: int
    vram_gb: float                # resident size incl. KV cache at context_tokens
    skills: list = field(default_factory=lambda: ["chat"])
    quality: int = 1              # higher = better; router breaks ties toward cheaper
    quant: str = ""
    max_output_tokens: int = 1024
    supports_tools: bool = True


@dataclass
class McpServerSpec:
    name: str
    command: list
    env: dict = field(default_factory=dict)


@dataclass
class Config:
    models: list
    vram_budget_gb: float
    workspace: str = "."
    shell_allow: list = field(default_factory=lambda: ["git", "python", "ls", "dir", "rg", "grep", "cat", "echo"])
    shell_timeout_s: float = 20.0
    tool_output_cap: int = 8_000      # characters kept from any tool result
    max_steps: int = 8
    max_total_tokens: int = 60_000
    reserve_output_tokens: int = 512
    audit_log: str = ".agentbus/audit.jsonl"
    mcp_servers: list = field(default_factory=list)
    request_timeout_s: float = 120.0

    def model(self, name: str) -> ModelSpec:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)


class ConfigError(ValueError):
    pass


def from_dict(d: dict, base: Path | None = None) -> Config:
    try:
        models = [ModelSpec(**m) for m in d["models"]]
    except (KeyError, TypeError) as e:
        raise ConfigError(f"bad model entry: {e}") from None
    if not models:
        raise ConfigError("at least one model is required")
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ConfigError("model names must be unique")
    for m in models:
        bad = set(m.skills) - set(SKILLS)
        if bad:
            raise ConfigError(f"{m.name}: unknown skills {sorted(bad)}")
        if m.context_tokens <= m.max_output_tokens:
            raise ConfigError(f"{m.name}: context_tokens must exceed max_output_tokens")
    budget = float(d.get("vram_budget_gb", 0))
    if budget <= 0:
        raise ConfigError("vram_budget_gb must be > 0")
    too_big = [m.name for m in models if m.vram_gb > budget]
    if len(too_big) == len(models):
        raise ConfigError(f"no model fits the {budget} GB VRAM budget")
    rest = {k: v for k, v in d.items() if k not in ("models", "vram_budget_gb", "mcp_servers")}
    cfg = Config(models=models, vram_budget_gb=budget,
                 mcp_servers=[McpServerSpec(**s) for s in d.get("mcp_servers", [])], **rest)
    if base and not Path(cfg.workspace).is_absolute():
        cfg.workspace = str((base / cfg.workspace).resolve())
    return cfg


def load(path: str | Path) -> Config:
    p = Path(path)
    return from_dict(json.loads(p.read_text(encoding="utf-8")), p.parent)
