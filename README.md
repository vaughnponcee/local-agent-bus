# local-agent-bus

local-agent-bus is a lightweight router and agent loop for **local quantized models**. It talks to llama.cpp `server`, LM Studio, Ollama's `/v1`, vLLM, or anything else that serves the OpenAI chat API. It gives those models filesystem and shell tools, plus tools from any MCP server, under hard budgets:

- **VRAM.** It tracks which models are resident and plans LRU evictions to stay under budget.
- **Context.** Every request is trimmed to fit the chosen model's window.
- **Tokens and steps.** Each run stops once its token budget or step limit is used up.
- **Audit.** Every decision goes into a hash-chained JSONL audit log.

No Docker, no frameworks, no SDKs. It needs only the Python 3.10+ standard library (`urllib`, `subprocess`, `http.server` for the test doubles).

```bash
python -m agentbus route "why does test_parser fail? see traceback" -c config.example.json   # dry-run routing
python -m agentbus run   "summarize the key points of docs/design.md" -c config.json
python -m agentbus tools -c config.json
python -m agentbus audit verify .agentbus/audit.jsonl
python -m agentbus audit show   .agentbus/audit.jsonl --run 3f9a1c2b7d10
python -m unittest discover -s tests -t .     # 32 tests, all against mocked endpoints
```

## Architecture

```
task ──► Router ──────────────► Agent loop (≤ max_steps) ───────────────────────────► answer
         ├ classify(task)         ├ fit_context(messages, model window × calibration)
         ├ filter: VRAM, context, ├ budget check: used + next prompt ≤ max_total_tokens
         │  tool support          ├ llm.chat()  ── OpenAI-compatible HTTP (urllib, retries)
         ├ rank: skill, quality,  └ tool calls ─► Toolbox
         │  residency, size                         ├ fs_list / fs_read / fs_write  (workspace jail)
         └ Residency (LRU, GB)                      ├ shell (allow-list, no shell syntax, timeout)
                                                    └ <server>__<tool> ─► McpClient (stdio JSON-RPC)
every step ─► AuditLog (JSONL, sha256 hash chain, fsync per line)
```

### Routing

`classify()` scores the task against keyword patterns for `code`, `summarize`, `reason` and `tools`; `chat` always scores 1. The patterns are deliberately simple, because the audit log records *why* a model was chosen and a reviewer can re-check the reasoning. Candidate models are filtered, then ranked:

1. **Filter.** Drop any model whose `vram_gb` exceeds the budget, whose context can't hold the prompt plus output reserve, or that can't call tools when tools are needed. Each rejection reason is logged.
2. **Rank.** First, does the model have the needed skill? Then quality: the best model for `code`/`reason`, the smallest for chat and summaries. Then prefer a model that is already resident (no load time). Then less VRAM.
3. **Plan residency.** If loading the chosen model would exceed `vram_budget_gb`, evict least-recently-used models until it fits. The plan is logged. The router *plans* evictions but does not unload anything itself; use separate llama.cpp servers per model, or LM Studio's JIT loading, and let the plan guide which to stop.

With the example config (12 GB budget):

| task | routed to | why |
|---|---|---|
| "refactor this python function" | `qwen-coder-7b` (6.1 GB) | code skill, highest quality for code |
| "summarize the key points of this note" | `llama-3b` (2.9 GB) | summarize skill, smallest |
| a 9,000-word document to summarize | `mistral-nemo-12b` (9.4 GB) | `llama-3b`'s 8k context is too small; evicts both resident models (6.1 + 2.9 + 9.4 > 12) |

### Context budget

Before each model call, the message list is fitted to `context_tokens − max_output_tokens − tool_schema_tokens`, giving up the cheapest information first:

1. trim the middle of the largest tool results to about 400 characters;
2. drop the oldest complete exchange (the assistant's tool calls plus their results), so no tool result is left orphaned;
3. truncate old user messages;
4. as a last resort, truncate the newest user message.

The system prompt is never touched.

**Token estimates vs. the real tokenizer.** Tokens are estimated as the average of a characters/4 count and a per-word count. Local servers report their real `usage.prompt_tokens`. After every call, the agent computes `calibration = max(1, real/estimated)` and multiplies later estimates by it, for both the context fit and the budget check.

This came from a failing test. A fake server reported 5,000 prompt tokens per call, far above the local estimate. The uncalibrated loop let a second call through and spent 10,020 tokens against a 9,000 budget.

### Tools and the sandbox

| tool | safeguards |
|---|---|
| `fs_list`, `fs_read`, `fs_write` | path resolved (`..` and symlinks) and required to stay inside `workspace`; reads capped at `tool_output_cap` characters |
| `shell` | `shlex.split` and run **without a shell** (no pipes, globs or `$()`); executable must be in `shell_allow`; path-like arguments must stay in the workspace; timeout; output cap |
| `<server>__<tool>` | proxied to an MCP server started from `mcp_servers[].command` |

A denied call is returned to the model as a normal tool result (`denied: ...`). It is also logged with `denied: true`, so you can find attempted escapes.

### Audit log

Each line is `{"seq", "ts", "run", "type", ..., "prev", "hash"}` with `hash = sha256(prev_hash + canonical_json(line))`. The event types are:

- `task`, `route` (with every rejected model and the reason)
- `context_trim` (what was removed)
- `model_call` (tokens, and whether the count came from the server or an estimate; latency; tool names)
- `tool_call` (arguments, ok/denied, result size, exit code, milliseconds)
- `model_error`, `stop`, `final`

`audit verify` recomputes the chain and reports the first edited, deleted or reordered line. This makes tampering *evident*, not impossible: someone who can rewrite the whole file can rebuild the chain. Copy the last hash somewhere else if that matters to you.

## Config

See [config.example.json](config.example.json). Per model: `endpoint`, `model`, `context_tokens`, `vram_gb` (weights plus KV cache at that context, which you measure once with `nvidia-smi`), `quant`, `skills`, `quality`, `max_output_tokens`, `supports_tools`. Global settings: `vram_budget_gb`, `workspace`, `shell_allow`, `shell_timeout_s`, `tool_output_cap`, `max_steps`, `max_total_tokens`, `audit_log`, `mcp_servers: [{"name", "command": [...], "env": {}}]`.

## Tests

All 32 tests run offline:

- `tests/fakes.py` is an OpenAI-compatible HTTP server on a random port that replays scripted replies: tool calls, 500s, bad JSON arguments, custom `usage`.
- `tests/fake_mcp_server.py` is a real stdio MCP server with `add` and `fail` tools.

They cover:
- routing, VRAM eviction order, and preferring resident models
- context fitting: tool results trimmed first, no orphaned tool messages
- the agent loop end to end (the model reads a file, tries `../../etc/passwd`, gets denied, then answers)
- the step limit and the token budget with calibration
- retry on 5xx but not 4xx
- sandbox escapes, the shell allow-list and timeouts
- detecting edited and deleted audit-log lines

Not tested here: a live llama.cpp or LM Studio server. Nothing was running on this machine while the tests were written, so behaviour against real models has only been checked against the fake server.

## Limitations

- **The classifier is keyword-based.** It is transparent and fast, but it will misroute vague tasks. Pass `--model` to force a choice.
- **VRAM is bookkeeping.** It relies on the `vram_gb` values you configure and does not query the GPU or load and unload models itself.
- **Single-agent and sequential.** No parallel tool calls and no streaming.
- **Tool calls need a server that supports them,** such as llama.cpp's `--jinja` with a tool-capable chat template. Models without tool support (`"supports_tools": false`) get no tools.

## License

MIT
