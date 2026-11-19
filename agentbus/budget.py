"""Token estimation and context-window fitting."""
from __future__ import annotations

import json
import re

_PIECES = re.compile(r"\w+|[^\w\s]")


def estimate_tokens(text: str) -> int:
    """~4 chars/token blended with a word/symbol count. Errs high on code,
    which is the safe direction for a hard context limit."""
    if not text:
        return 0
    pieces = _PIECES.findall(text)
    by_pieces = sum(max(1, (len(p) + 3) // 4) for p in pieces)
    return max(1, round((by_pieces + len(text) / 4) / 2))


def message_tokens(m: dict) -> int:
    n = 4  # role + framing overhead per message (OpenAI-style chat template)
    n += estimate_tokens(m.get("content") or "")
    if m.get("tool_calls"):
        n += estimate_tokens(json.dumps(m["tool_calls"]))
    return n


def total_tokens(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages) + 3


def fit_context(messages: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    """Shrink `messages` to fit `limit` tokens. Returns (fitted, actions).

    Order of sacrifice (cheapest information first):
      1. truncate the middle of the largest tool results to 400 chars
      2. drop the oldest complete exchange (assistant tool_calls + its tool results)
      3. truncate the oldest remaining user/assistant text
    The system prompt and the latest user message are never removed.
    """
    msgs = [dict(m) for m in messages]
    actions: list[dict] = []
    if total_tokens(msgs) <= limit:
        return msgs, actions

    # 1. squeeze tool outputs, biggest first
    tool_idx = sorted((i for i, m in enumerate(msgs) if m["role"] == "tool"),
                      key=lambda i: -len(msgs[i].get("content") or ""))
    for i in tool_idx:
        c = msgs[i].get("content") or ""
        if len(c) > 400:
            msgs[i]["content"] = c[:250] + f"\n...[{len(c) - 400} chars trimmed by agentbus]...\n" + c[-150:]
            actions.append({"action": "trim_tool_result", "index": i, "chars_before": len(c)})
            if total_tokens(msgs) <= limit:
                return msgs, actions

    # 2. drop oldest exchanges between the first user turn and the last user turn
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    while total_tokens(msgs) > limit:
        cand = next((i for i, m in enumerate(msgs) if i < last_user and m["role"] == "assistant"), None)
        if cand is None:
            break
        j = cand + 1
        while j < len(msgs) and msgs[j]["role"] == "tool":
            j += 1
        actions.append({"action": "drop_exchange", "messages": j - cand})
        del msgs[cand:j]
        last_user -= j - cand

    # 3. hard-truncate old user text (never the system prompt or the final user turn)
    for i, m in enumerate(msgs):
        if total_tokens(msgs) <= limit:
            break
        if m["role"] == "user" and i != last_user and len(m.get("content") or "") > 200:
            c = m["content"]
            m["content"] = c[:200] + "...[truncated]"
            actions.append({"action": "truncate_message", "index": i, "chars_before": len(c)})

    if total_tokens(msgs) > limit:  # last resort: the newest user message itself
        m = msgs[last_user]
        keep = max(200, len(m["content"]) * limit // max(1, total_tokens(msgs)) - 50)
        actions.append({"action": "truncate_latest_user", "chars_before": len(m["content"]), "chars_after": keep})
        m["content"] = m["content"][:keep]
    return msgs, actions
