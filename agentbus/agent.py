"""The agent loop: route -> fit context -> call model -> run tools -> repeat."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from . import llm
from .audit import AuditLog
from .budget import estimate_tokens, fit_context, total_tokens
from .config import Config
from .router import Router
from .tools import Toolbox

SYSTEM = ("You are a local coding assistant. Use the provided tools to inspect the workspace before "
          "answering questions about it. Be concise. When done, answer without calling tools.")


@dataclass
class RunResult:
    answer: str
    model: str
    steps: int
    stopped: str           # done | max_steps | token_budget | error
    tool_calls: int = 0
    tokens_used: int = 0
    run_id: str = ""
    events: list = field(default_factory=list)


class Agent:
    def __init__(self, cfg: Config, toolbox: Toolbox | None = None, router: Router | None = None,
                 audit: AuditLog | None = None, chat_fn=llm.chat):
        self.cfg = cfg
        self.tools = toolbox or Toolbox(cfg)
        self.router = router or Router(cfg)
        self.audit = audit or AuditLog(cfg.audit_log)
        self.chat = chat_fn

    def run(self, task: str, prefer_model: str | None = None) -> RunResult:
        a = self.audit
        a.write("task", task=task, chars=len(task))
        schemas = self.tools.schemas()
        tools_tokens = estimate_tokens(json.dumps(schemas))
        decision = self.router.route(task, needs_tools=True, prefer=prefer_model,
                                     history_tokens=tools_tokens + estimate_tokens(SYSTEM))
        a.write("route", **decision.as_dict())
        self.router.commit(decision)
        m = decision.model
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
        used = 0
        calls = 0
        # Our estimate vs the server's real tokenizer: once the server reports
        # usage, scale later estimates by the observed ratio (never below 1x),
        # so the budget check can't be fooled by an estimator that runs low.
        calib = 1.0
        window = m.context_tokens - m.max_output_tokens - tools_tokens
        for step in range(1, self.cfg.max_steps + 1):
            fitted, actions = fit_context(messages, int(window / calib))
            if actions:
                a.write("context_trim", step=step, limit=int(window / calib), actions=actions,
                        tokens_after=total_tokens(fitted), calibration=round(calib, 3))
            est_prompt = total_tokens(fitted) + tools_tokens
            prompt_tokens = int(est_prompt * calib)
            if used + prompt_tokens > self.cfg.max_total_tokens:
                a.write("stop", reason="token_budget", used=used, next_prompt=prompt_tokens,
                        budget=self.cfg.max_total_tokens)
                return RunResult("(stopped: token budget exhausted)", m.name, step - 1, "token_budget",
                                 calls, used, a.run)
            t0 = time.perf_counter()
            try:
                comp = self.chat(m.endpoint, m.model, fitted, schemas if m.supports_tools else None,
                                 max_tokens=m.max_output_tokens, timeout=self.cfg.request_timeout_s)
            except llm.LLMError as e:
                a.write("model_error", step=step, model=m.name, error=str(e))
                return RunResult(f"(model error: {e})", m.name, step, "error", calls, used, a.run)
            out_tokens = comp.usage.get("completion_tokens") or estimate_tokens(comp.content)
            in_tokens = comp.usage.get("prompt_tokens") or prompt_tokens
            if comp.usage.get("prompt_tokens"):
                calib = max(1.0, calib, in_tokens / max(1, est_prompt))
            used += in_tokens + out_tokens
            a.write("model_call", step=step, model=m.name, prompt_tokens=in_tokens, completion_tokens=out_tokens,
                    tokens_source="server" if comp.usage else "estimate", latency_s=round(comp.latency_s, 3),
                    finish=comp.finish_reason, tool_calls=[c["name"] for c in comp.tool_calls],
                    wall_s=round(time.perf_counter() - t0, 3))
            if not comp.tool_calls:
                a.write("final", step=step, answer_chars=len(comp.content), tokens_used=used, tool_calls=calls)
                return RunResult(comp.content, m.name, step, "done", calls, used, a.run)
            messages.append({"role": "assistant", "content": comp.content or None,
                             "tool_calls": [{"id": c["id"], "type": "function",
                                             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                            for c in comp.tool_calls]})
            for c in comp.tool_calls:
                calls += 1
                res = self.tools.call(c["name"], c["arguments"])
                a.write("tool_call", step=step, tool=c["name"], args=c["arguments"], ok=res.ok,
                        denied=res.denied, result_chars=len(res.content), meta=res.meta or {})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": res.content})
        a.write("stop", reason="max_steps", steps=self.cfg.max_steps, tokens_used=used)
        return RunResult("(stopped: step limit reached)", m.name, self.cfg.max_steps, "max_steps",
                         calls, used, a.run)
