"""OpenAI-compatible chat client (llama.cpp server, LM Studio, Ollama's /v1,
vLLM...). urllib only; retries on connection errors and 5xx."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    content: str
    tool_calls: list = field(default_factory=list)  # [{"id", "name", "arguments": dict}]
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)


def chat(endpoint: str, model: str, messages: list[dict], tools: list[dict] | None = None,
         max_tokens: int = 1024, temperature: float = 0.2, timeout: float = 120.0,
         retries: int = 2, backoff: float = 0.5) -> Completion:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        body["tools"] = [{"type": "function", "function": t} for t in tools]
        body["tool_choice"] = "auto"
    url = endpoint.rstrip("/") + "/chat/completions"
    data = json.dumps(body).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            last = LLMError(f"HTTP {e.code} from {url}: {detail}")
            if e.code < 500:
                raise last from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = LLMError(f"cannot reach {url}: {getattr(e, 'reason', e)}")
        except json.JSONDecodeError:
            last = LLMError(f"non-JSON response from {url}")
        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))
    else:
        raise last  # type: ignore[misc]
    latency = time.perf_counter() - t0
    try:
        choice = payload["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"unexpected response shape: {str(payload)[:200]}") from None
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        try:
            parsed = json.loads(args) if isinstance(args, str) else dict(args)
        except json.JSONDecodeError:
            parsed = {"__invalid_json__": args}
        calls.append({"id": tc.get("id") or f"call_{len(calls)}", "name": fn.get("name", ""), "arguments": parsed})
    return Completion(msg.get("content") or "", calls, choice.get("finish_reason", ""),
                      payload.get("usage") or {}, latency, payload)
