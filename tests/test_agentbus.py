import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agentbus import audit, budget, config, llm, router
from agentbus.agent import Agent
from agentbus.cli import main
from agentbus.mcp import McpClient, McpError
from agentbus.tools import Toolbox

from .fakes import FakeOpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def make_cfg(tmp: Path, endpoint="http://127.0.0.1:9/v1", **over) -> config.Config:
    d = json.loads((ROOT / "config.example.json").read_text())
    for m in d["models"]:
        m["endpoint"] = endpoint
    d["workspace"] = str(tmp / "ws")
    d["audit_log"] = str(tmp / "audit.jsonl")
    d.update(over)
    (tmp / "ws").mkdir(exist_ok=True)
    return config.from_dict(d)


class Tmp(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()


class ConfigTests(Tmp):
    def test_example_loads(self):
        cfg = config.load(ROOT / "config.example.json")
        self.assertEqual(len(cfg.models), 3)

    def test_validation(self):
        base = json.loads((ROOT / "config.example.json").read_text())
        for mutate, msg in [
            (lambda d: d.update(vram_budget_gb=0), "vram_budget_gb"),
            (lambda d: d.update(vram_budget_gb=1), "no model fits"),
            (lambda d: d["models"][0].update(skills=["juggling"]), "unknown skills"),
            (lambda d: d["models"][1].update(name=d["models"][0]["name"]), "unique"),
            (lambda d: d["models"][0].update(context_tokens=100), "context_tokens"),
        ]:
            d = json.loads(json.dumps(base))
            mutate(d)
            with self.assertRaises(config.ConfigError) as cm:
                config.from_dict(d)
            self.assertIn(msg, str(cm.exception))


class BudgetTests(unittest.TestCase):
    def test_estimate(self):
        self.assertEqual(budget.estimate_tokens(""), 0)
        text = "def f(x):\n    return x * 2\n" * 50
        self.assertTrue(len(text) / 8 < budget.estimate_tokens(text) < len(text) / 2)

    def test_fit_trims_tool_output_first(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "x" * 20000}, {"role": "user", "content": "latest question"}]
        out, actions = budget.fit_context(msgs, 1000)
        self.assertLessEqual(budget.total_tokens(out), 1000)
        self.assertEqual(actions[0]["action"], "trim_tool_result")
        self.assertEqual(out[-1]["content"], "latest question")
        self.assertEqual(out[0]["content"], "sys")

    def test_fit_drops_old_exchanges(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "old " * 50}]
        for i in range(10):
            msgs += [{"role": "assistant", "content": "thinking " * 40, "tool_calls": [{"id": str(i)}]},
                     {"role": "tool", "content": "r" * 300}]
        msgs.append({"role": "user", "content": "now"})
        out, actions = budget.fit_context(msgs, 300)
        self.assertLessEqual(budget.total_tokens(out), 300)
        self.assertIn("drop_exchange", {a["action"] for a in actions})
        self.assertEqual([m["role"] for m in out][-1], "user")
        # never leaves an orphaned tool message without its assistant call
        for i, m in enumerate(out):
            if m["role"] == "tool":
                self.assertIn(out[i - 1]["role"], ("assistant", "tool"))

    def test_fit_noop_when_small(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(budget.fit_context(msgs, 100), (msgs, []))


class RouterTests(Tmp):
    def test_classify(self):
        self.assertEqual(max(router.classify("fix the bug in parser.py traceback"), key=lambda k: router.classify(
            "fix the bug in parser.py traceback")[k]), "code")
        s = router.classify("summarize this meeting transcript")
        self.assertGreater(s.get("summarize", 0), s.get("code", 0))

    def test_routes_by_skill(self):
        r = router.Router(make_cfg(self.tmp))
        self.assertEqual(r.route("refactor this python function").model.name, "qwen-coder-7b")
        self.assertEqual(r.route("summarize the key points of this note").model.name, "llama-3b")
        self.assertEqual(r.route("hello there").model.name, "llama-3b")  # chat prefers smallest

    def test_context_forces_bigger_model(self):
        r = router.Router(make_cfg(self.tmp))
        d = r.route("summarize: " + "word " * 9000)
        self.assertEqual(d.model.name, "mistral-nemo-12b")
        self.assertIn("context", d.rejected["llama-3b"])

    def test_vram_eviction_lru(self):
        r = router.Router(make_cfg(self.tmp))  # 12 GB budget
        d1 = r.route("refactor this python function")       # 6.1 GB
        r.commit(d1)
        d2 = r.route("summarize the key points")             # 2.9 GB, fits alongside
        r.commit(d2)
        self.assertEqual(d2.evict, [])
        self.assertAlmostEqual(r.res.used, 9.0)
        d3 = r.route("summarize: " + "word " * 9000)         # 9.4 GB: must evict
        self.assertEqual(d3.evict, ["qwen-coder-7b", "llama-3b"])
        r.commit(d3)
        self.assertLessEqual(r.res.used, 12)
        self.assertEqual(list(r.res.loaded), ["mistral-nemo-12b"])

    def test_prefers_resident_on_tie(self):
        cfg = make_cfg(self.tmp)
        cfg.models.append(config.ModelSpec("llama-3b-copy", cfg.models[1].endpoint, "m", 8192, 2.9,
                                           ["chat", "summarize", "tools"], 1, max_output_tokens=768))
        r = router.Router(cfg)
        r.res.touch(cfg.models[-1], [])
        self.assertEqual(r.route("summarize this").model.name, "llama-3b-copy")

    def test_nothing_fits(self):
        r = router.Router(make_cfg(self.tmp))
        with self.assertRaises(RuntimeError):
            r.route("x " * 60000)


class AuditTests(Tmp):
    def test_chain_and_tamper_detection(self):
        p = self.tmp / "a.jsonl"
        log = audit.AuditLog(p, "run1")
        for i in range(5):
            log.write("event", i=i, note="ok")
        self.assertEqual(audit.verify(p), (True, 5, "5 lines, chain intact"))
        lines = p.read_text().splitlines()
        edited = lines[:]
        edited[2] = edited[2].replace('"note": "ok"', '"note": "OK"')
        p.write_text("\n".join(edited) + "\n")
        ok, n, msg = audit.verify(p)
        self.assertFalse(ok)
        self.assertIn("line 3", msg)
        p.write_text("\n".join(lines[:2] + lines[3:]) + "\n")  # delete a line
        ok, _, msg = audit.verify(p)
        self.assertFalse(ok)
        self.assertIn("chain broken", msg)

    def test_resume_appends_to_chain(self):
        p = self.tmp / "a.jsonl"
        audit.AuditLog(p, "r1").write("x")
        audit.AuditLog(p, "r2").write("y")
        self.assertTrue(audit.verify(p)[0])
        self.assertEqual([r["run"] for r in audit.read(p)], ["r1", "r2"])


class ToolTests(Tmp):
    def setUp(self):
        super().setUp()
        self.cfg = make_cfg(self.tmp, tool_output_cap=300)
        self.box = Toolbox(self.cfg)
        ws = Path(self.cfg.workspace)
        (ws / "notes.txt").write_text("hello\nworld\n")
        (ws / "big.txt").write_text("z" * 5000)
        (self.tmp / "secret.txt").write_text("outside")

    def test_fs(self):
        self.assertIn("f notes.txt", self.box.call("fs_list", {}).content)
        self.assertEqual(self.box.call("fs_read", {"path": "notes.txt"}).content, "hello\nworld\n")
        r = self.box.call("fs_write", {"path": "sub/new.txt", "content": "abc"})
        self.assertTrue(r.ok)
        self.assertEqual((Path(self.cfg.workspace) / "sub/new.txt").read_text(), "abc")

    def test_sandbox_escape_denied(self):
        for args in ({"path": "../secret.txt"}, {"path": str(self.tmp / "secret.txt")}, {"path": "sub/../../secret.txt"}):
            r = self.box.call("fs_read", args)
            self.assertTrue(r.denied, args)
            self.assertNotIn("outside", r.content)

    def test_output_cap(self):
        r = self.box.call("fs_read", {"path": "big.txt"})
        self.assertLessEqual(len(r.content), 350)
        self.assertTrue(r.meta["truncated"])

    def test_shell_allowlist(self):
        r = self.box.call("shell", {"command": f'"{sys.executable}" -c "print(6*7)"'})
        self.assertTrue(r.ok, r.content)
        self.assertIn("42", r.content)
        for cmd in ("rm -rf /", "curl http://example.com", "powershell -c whoami"):
            self.assertTrue(self.box.call("shell", {"command": cmd}).denied, cmd)
        self.assertTrue(self.box.call("shell", {"command": "python ../secret.txt"}).denied)

    def test_shell_timeout(self):
        self.cfg.shell_timeout_s = 0.5
        r = self.box.call("shell", {"command": f'"{sys.executable}" -c "import time; time.sleep(5)"'})
        self.assertFalse(r.ok)
        self.assertTrue(r.meta["timeout"])

    def test_unknown_and_bad_args(self):
        self.assertFalse(self.box.call("nope", {}).ok)
        self.assertIn("KeyError", self.box.call("fs_read", {}).content)


class McpTests(Tmp):
    def test_client_against_fake_server(self):
        c = McpClient("fake", [sys.executable, str(HERE / "fake_mcp_server.py")])
        try:
            self.assertEqual(c.server_info["name"], "fake-mcp")
            self.assertEqual([t["name"] for t in c.tools], ["add", "fail"])
            self.assertEqual(c.call("add", {"a": 2, "b": 40}), ("42", False))
            self.assertEqual(c.call("fail", {}), ("it failed", True))
            with self.assertRaises(McpError):
                c.request("bogus/method")
            box = Toolbox(make_cfg(self.tmp), [c])
            self.assertIn("fake__add", [t["name"] for t in box.schemas()])
            self.assertEqual(box.call("fake__add", {"a": 1, "b": 1}).content, "2")
        finally:
            c.close()


class LlmTests(unittest.TestCase):
    def test_parse_tool_calls_and_usage(self):
        with FakeOpenAI([{"tool_calls": [("fs_read", {"path": "a.py"})], "usage": {"prompt_tokens": 7,
                                                                                  "completion_tokens": 3}}]) as f:
            c = llm.chat(f.url, "m", [{"role": "user", "content": "hi"}], tools=[{"name": "fs_read"}])
        self.assertEqual(c.tool_calls, [{"id": "call_0", "name": "fs_read", "arguments": {"path": "a.py"}}])
        self.assertEqual(c.usage["prompt_tokens"], 7)
        self.assertEqual(f.requests[0]["tools"][0]["type"], "function")

    def test_retry_on_500_then_success(self):
        with FakeOpenAI([{"status": 500}, {"content": "ok"}]) as f:
            c = llm.chat(f.url, "m", [{"role": "user", "content": "hi"}], backoff=0.01)
        self.assertEqual(c.content, "ok")
        self.assertEqual(len(f.requests), 2)

    def test_4xx_not_retried(self):
        with FakeOpenAI([{"status": 400}, {"content": "never"}]) as f:
            with self.assertRaises(llm.LLMError):
                llm.chat(f.url, "m", [], backoff=0.01)
        self.assertEqual(len(f.requests), 1)

    def test_unreachable(self):
        with self.assertRaises(llm.LLMError):
            llm.chat("http://127.0.0.1:9/v1", "m", [], retries=0, timeout=1)

    def test_bad_json_arguments_survive(self):
        with FakeOpenAI([{"tool_calls": [("fs_read", "{not json")]}]) as f:
            c = llm.chat(f.url, "m", [])
        self.assertIn("__invalid_json__", c.tool_calls[0]["arguments"])


class AgentTests(Tmp):
    def test_tool_loop_end_to_end(self):
        script = [{"tool_calls": [("fs_list", {}), ("fs_read", {"path": "app.py"})]},
                  {"tool_calls": [("fs_read", {"path": "../../etc/passwd"})]},
                  {"content": "app.py defines main() which prints hello."}]
        with FakeOpenAI(script) as f:
            cfg = make_cfg(self.tmp, endpoint=f.url)
            (Path(cfg.workspace) / "app.py").write_text("def main():\n    print('hello')\n")
            res = Agent(cfg).run("explain what app.py does in this repo")
        self.assertEqual(res.stopped, "done")
        self.assertEqual(res.model, "qwen-coder-7b")
        self.assertEqual(res.tool_calls, 3)
        self.assertIn("main()", res.answer)
        # the model saw the file content in the second request
        tool_msgs = [m for m in f.requests[1]["messages"] if m["role"] == "tool"]
        self.assertIn("print('hello')", tool_msgs[1]["content"])
        events = audit.read(cfg.audit_log)
        types = [e["type"] for e in events]
        self.assertEqual(types[:2], ["task", "route"])
        self.assertEqual(types[-1], "final")
        denied = [e for e in events if e["type"] == "tool_call" and e["denied"]]
        self.assertEqual(len(denied), 1)
        self.assertTrue(audit.verify(cfg.audit_log)[0])

    def test_max_steps(self):
        with FakeOpenAI([{"tool_calls": [("fs_list", {})]}] * 10) as f:
            cfg = make_cfg(self.tmp, endpoint=f.url, max_steps=3)
            res = Agent(cfg).run("list files")
        self.assertEqual(res.stopped, "max_steps")
        self.assertEqual(len(f.requests), 3)

    def test_token_budget(self):
        script = [{"tool_calls": [("fs_list", {})], "usage": {"prompt_tokens": 5000, "completion_tokens": 10}}] * 5
        with FakeOpenAI(script) as f:
            cfg = make_cfg(self.tmp, endpoint=f.url, max_total_tokens=9000)
            res = Agent(cfg).run("list files")
        self.assertEqual(res.stopped, "token_budget")
        self.assertLessEqual(res.tokens_used, 9000)

    def test_context_window_respected(self):
        """Huge tool output must be trimmed before the next request so the
        prompt fits the model's window."""
        with FakeOpenAI([{"tool_calls": [("fs_read", {"path": "huge.log"})]}, {"content": "done"}]) as f:
            cfg = make_cfg(self.tmp, endpoint=f.url, tool_output_cap=200_000)
            (Path(cfg.workspace) / "huge.log").write_text("ERROR something happened\n" * 8000)
            res = Agent(cfg).run("summarize huge.log key points")
            self.assertEqual(res.model, "llama-3b")
            m = cfg.model("llama-3b")
            sent = f.requests[1]["messages"]
            self.assertLessEqual(budget.total_tokens(sent), m.context_tokens - m.max_output_tokens)
            self.assertIn("context_trim", [e["type"] for e in audit.read(cfg.audit_log)])

    def test_model_error_logged(self):
        cfg = make_cfg(self.tmp, endpoint="http://127.0.0.1:9/v1", request_timeout_s=1)
        res = Agent(cfg, chat_fn=lambda *a, **k: (_ for _ in ()).throw(llm.LLMError("down"))).run("hi")
        self.assertEqual(res.stopped, "error")
        self.assertEqual(audit.read(cfg.audit_log)[-1]["type"], "model_error")


class CliTests(Tmp):
    def test_route_and_audit_commands(self):
        cfg_path = self.tmp / "c.json"
        d = json.loads((ROOT / "config.example.json").read_text())
        d["workspace"] = str(self.tmp)
        cfg_path.write_text(json.dumps(d))
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["route", "fix this python traceback", "-c", str(cfg_path)]), 0)
        self.assertEqual(json.loads(buf.getvalue())["model"], "qwen-coder-7b")
        log = self.tmp / "a.jsonl"
        audit.AuditLog(log).write("x")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["audit", "verify", str(log)]), 0)
        log.write_text(log.read_text().replace('"x"', '"y"'))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["audit", "verify", str(log)]), 1)


if __name__ == "__main__":
    unittest.main()
