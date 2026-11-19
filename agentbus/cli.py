"""agentbus CLI.

  python -m agentbus route "summarize README.md" -c config.example.json     # dry run
  python -m agentbus run   "why does test_x fail?" -c config.json
  python -m agentbus tools -c config.json
  python -m agentbus audit verify .agentbus/audit.jsonl
  python -m agentbus audit show   .agentbus/audit.jsonl [--run ID]
"""
from __future__ import annotations

import argparse
import json
import sys

from . import audit, config
from .agent import Agent
from .mcp import McpClient
from .router import Router
from .tools import Toolbox


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agentbus", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("route", "run", "tools"):
        p = sub.add_parser(name)
        if name != "tools":
            p.add_argument("task")
            p.add_argument("--model", help="force a model by name")
        p.add_argument("-c", "--config", default="config.json")
    au = sub.add_parser("audit")
    au.add_argument("action", choices=["verify", "show"])
    au.add_argument("log")
    au.add_argument("--run")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if a.cmd == "audit":
        if a.action == "verify":
            ok, n, msg = audit.verify(a.log)
            print(("OK  " if ok else "FAIL ") + msg)
            return 0 if ok else 1
        for r in audit.read(a.log, a.run):
            extra = {k: v for k, v in r.items() if k not in ("seq", "ts", "run", "type", "prev", "hash")}
            print(f"{r['seq']:>5} {r['run']} {r['type']:<13} {json.dumps(extra, ensure_ascii=False)[:150]}")
        return 0

    cfg = config.load(a.config)
    if a.cmd == "route":
        d = Router(cfg).route(a.task, needs_tools=True, prefer=a.model)
        print(json.dumps(d.as_dict(), indent=2, ensure_ascii=False))
        return 0

    clients = [McpClient(s.name, s.command, s.env) for s in cfg.mcp_servers]
    try:
        box = Toolbox(cfg, clients)
        if a.cmd == "tools":
            for t in box.schemas():
                print(f"{t['name']:28} {t.get('description', '')[:80]}")
            return 0
        res = Agent(cfg, box).run(a.task, a.model)
        print(res.answer)
        print(f"\n[{res.stopped}] model={res.model} steps={res.steps} tool_calls={res.tool_calls} "
              f"tokens={res.tokens_used} run={res.run_id}", file=sys.stderr)
        return 0 if res.stopped == "done" else 2
    finally:
        for c in clients:
            c.close()


if __name__ == "__main__":
    raise SystemExit(main())
