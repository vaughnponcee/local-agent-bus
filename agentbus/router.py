"""Task classification, model selection under a VRAM budget, and residency
tracking (which models are 'loaded' and which to evict)."""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

from .budget import estimate_tokens
from .config import Config, ModelSpec

RULES = [  # (skill, weight, pattern)
    ("code", 3, r"\b(code|function|class|bug|stack ?trace|refactor|compile|test|regex|python|rust|typescript|"
                r"javascript|sql|diff|patch|traceback|exception|implement)\b|```|\.(py|rs|ts|js|go|cpp|c|h)\b"),
    ("summarize", 3, r"\b(summari[sz]e|tl;?dr|key points|condense|digest|recap|overview of)\b"),
    ("reason", 2, r"\b(prove|why|plan|compare|trade-?offs?|step by step|derive|estimate|design)\b"),
    ("tools", 3, r"\b(file|files|folder|directory|repo|read|list|run|execute|search|grep|open|ls|git)\b"),
]


def classify(task: str) -> dict[str, int]:
    """Keyword scores per skill. Cheap and transparent on purpose: the audit
    log records exactly which words caused a routing decision."""
    t = task.lower()
    scores = {"chat": 1}
    for skill, w, pat in RULES:
        hits = re.findall(pat, t)
        if hits:
            scores[skill] = scores.get(skill, 0) + w * len(hits)
    return scores


@dataclass
class RouteDecision:
    model: ModelSpec
    skill: str
    scores: dict
    prompt_tokens: int
    evict: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    rejected: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"model": self.model.name, "skill": self.skill, "scores": self.scores,
                "prompt_tokens": self.prompt_tokens, "evict": self.evict, "reasons": self.reasons,
                "rejected": self.rejected}


class Residency:
    """LRU set of models currently resident in VRAM (as far as we know)."""

    def __init__(self, budget_gb: float):
        self.budget = budget_gb
        self.loaded: OrderedDict[str, float] = OrderedDict()

    @property
    def used(self) -> float:
        return sum(self.loaded.values())

    def plan_load(self, m: ModelSpec) -> list[str] | None:
        """Names to evict (LRU first) so `m` fits; None if it can never fit."""
        if m.name in self.loaded:
            return []
        if m.vram_gb > self.budget:
            return None
        evict, free = [], self.budget - self.used
        for name, gb in self.loaded.items():
            if free >= m.vram_gb:
                break
            evict.append(name)
            free += gb
        return evict if free >= m.vram_gb else None

    def touch(self, m: ModelSpec, evict: list[str]):
        for n in evict:
            self.loaded.pop(n, None)
        self.loaded[m.name] = m.vram_gb
        self.loaded.move_to_end(m.name)


class Router:
    def __init__(self, cfg: Config, residency: Residency | None = None):
        self.cfg = cfg
        self.res = residency or Residency(cfg.vram_budget_gb)

    def route(self, task: str, needs_tools: bool = False, prefer: str | None = None,
              history_tokens: int = 0) -> RouteDecision:
        scores = classify(task)
        skill = max(scores, key=lambda s: (scores[s], s != "chat"))
        need = estimate_tokens(task) + history_tokens
        reasons = [f"skill={skill} from scores {scores}", f"prompt≈{need} tokens"]
        rejected: dict[str, str] = {}
        cands = []
        for m in self.cfg.models:
            if prefer and m.name != prefer:
                rejected[m.name] = "not the requested model"
                continue
            if m.vram_gb > self.cfg.vram_budget_gb:
                rejected[m.name] = f"{m.vram_gb} GB > budget {self.cfg.vram_budget_gb} GB"
                continue
            room = m.context_tokens - m.max_output_tokens - self.cfg.reserve_output_tokens
            if need > room:
                rejected[m.name] = f"context {m.context_tokens} too small for ~{need} tokens"
                continue
            if needs_tools and not m.supports_tools:
                rejected[m.name] = "no tool-calling support"
                continue
            cands.append(m)
        if not cands:
            raise RuntimeError(f"no model can take this task: {rejected}")

        def rank(m: ModelSpec):
            has_skill = skill in m.skills
            resident = m.name in self.res.loaded
            # skill match first; then for hard skills prefer quality, for chat prefer small;
            # then prefer already-resident (no load latency); then less VRAM
            q = m.quality if skill in ("code", "reason") else -m.vram_gb
            return (has_skill, q, resident, -m.vram_gb)

        cands.sort(key=rank, reverse=True)
        best = cands[0]
        evict = self.res.plan_load(best) or []
        reasons.append(f"chose {best.name} (skill match={skill in best.skills}, quality={best.quality}, "
                       f"{best.vram_gb} GB, resident={best.name in self.res.loaded})")
        if evict:
            reasons.append(f"evict {evict} to stay within {self.cfg.vram_budget_gb} GB")
        return RouteDecision(best, skill, scores, need, evict, reasons, rejected)

    def commit(self, d: RouteDecision):
        self.res.touch(d.model, d.evict)
