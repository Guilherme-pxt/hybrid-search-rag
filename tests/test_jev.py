import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path

from hybridrag import pipeline as P
from hybridrag.jev import (CRITERIA, BudgetExceeded, JevClient, JevError, JevReranker, cost_usd, load_api_key,
                           rerank_request)
from hybridrag.jev_eval import ALL_VARIANTS, JEV_VARIANTS, cascade, step_jev_eval
from hybridrag.providers import CachedProvider, ChatResult, ProviderError, ScriptedProvider
from hybridrag.report import write_report
from hybridrag.rerank import LLMReranker
from tests.test_synth_pipeline import fake_llm

KEY = "test-key-0123456789-do-not-leak"
ENV = {"JEV_API_KEY": KEY}


def words(text):
    return set(re.findall(r"[a-z0-9]{4,}", text.lower()))


class FakeTransport:
    """Answers each noul with the share of the user question's words found in the passage."""

    def __init__(self, script=None):
        self.calls = []
        self.script = list(script or [])  # optional (status, headers, body) replies before the normal ones

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": json.loads(body)})
        if self.script:
            return self.script.pop(0)
        req = json.loads(body)
        q = words(req["state"]["user_question"])
        answers = {qid: {"type": "noul", "noul": round(len(q & words(spec["instructions"]["passage"])) / max(len(q), 1), 4)}
                   for qid, spec in req["questions"].items()}
        resp = {"model": "jev-1.14.0", "answers": answers, "usage": {"input_tokens": len(body) // 4, "output_tokens": 3 * len(answers)}}
        return 200, {}, json.dumps(resp).encode()


class JevClientTests(unittest.TestCase):
    def test_key_from_env_then_file(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(JevError) as cm:
                load_api_key(d, env={})
            self.assertIn(".env.jev", str(cm.exception))
            Path(d, ".env.jev").write_text("# comment\nOTHER=1\nexport JEV_API_KEY='from-file'\n")
            self.assertEqual(load_api_key(d, env={}), "from-file")
            self.assertEqual(load_api_key(d, env={"JEV_API_KEY": "from-env"}), "from-env")

    def test_request_shape(self):
        body = rerank_request("jev-1.14.0", "What is the EU meal per diem?", ["passage one", "passage two"])
        self.assertEqual(body["model"], "jev-1.14.0")
        self.assertEqual(body["state"], {"user_question": "What is the EU meal per diem?"})
        self.assertEqual(list(body["questions"]), ["c01", "c02"])
        q = body["questions"]["c02"]
        self.assertEqual(q["type"], "noul")
        self.assertEqual(q["instructions"]["passage"], "passage two")
        self.assertIn("`passage`", q["instructions"]["question"])
        self.assertEqual(q["criteria"], CRITERIA)

    def test_bearer_header_cache_and_no_key_anywhere(self):
        with tempfile.TemporaryDirectory() as d:
            tr = FakeTransport()
            c = JevClient(f"{d}/c.sqlite", key_root=d, transport=tr, env=ENV)
            self.addCleanup(c.close)
            body = rerank_request("jev-1.14.0", "battery life of the sensor", ["the sensor battery life is 2 years", "other"])
            r1 = c.ask(body)
            self.assertFalse(r1.cached)
            self.assertEqual(tr.calls[0]["headers"]["Authorization"], f"Bearer {KEY}")
            self.assertTrue(tr.calls[0]["url"].endswith("/v1/systemone"))
            self.assertGreater(r1.nouls["c01"], r1.nouls["c02"])
            r2 = c.ask(body)
            self.assertTrue(r2.cached)
            self.assertEqual(len(tr.calls), 1)
            self.assertEqual((r2.nouls, r2.latency_s, r2.input_tokens), (r1.nouls, r1.latency_s, r1.input_tokens))
            self.assertNotIn(KEY, repr(c))
            c.close()
            db = sqlite3.connect(f"{d}/c.sqlite")
            dump = "\n".join(v for (v,) in db.execute("SELECT value FROM cache"))
            db.close()
            self.assertNotIn(KEY, dump)
            # a cache hit needs no key at all
            c2 = JevClient(f"{d}/c.sqlite", key_root=d, transport=FakeTransport(), env={})
            self.addCleanup(c2.close)
            self.assertTrue(c2.ask(body).cached)

    def test_retry_then_error_without_key(self):
        sleeps = []
        tr = FakeTransport(script=[(429, {"Retry-After": "2"}, b"slow down"), (529, {}, b"overloaded")])
        c = JevClient(None, transport=tr, env=ENV, sleep=sleeps.append, backoff_s=0.5)
        res = c.ask(rerank_request("jev-1.14.0", "q words here", ["p"]))
        self.assertEqual(len(tr.calls), 3)
        self.assertEqual(sleeps, [2.0, 1.0])
        self.assertEqual(c.retries, 2)
        self.assertIn("c01", res.nouls)
        bad = JevClient(None, transport=FakeTransport(script=[(422, {}, b'{"detail":"bad question"}')]), env=ENV)
        with self.assertRaises(JevError) as cm:
            bad.ask(rerank_request("jev-1.14.0", "q", ["p"]))
        self.assertIn("422", str(cm.exception))
        self.assertNotIn(KEY, str(cm.exception))

    def test_missing_answer_is_an_error(self):
        reply = json.dumps({"model": "m", "answers": {"c01": {"type": "noul", "noul": 0.5}},
                            "usage": {"input_tokens": 1, "output_tokens": 1}}).encode()
        c = JevClient(None, transport=FakeTransport(script=[(200, {}, reply)]), env=ENV)
        with self.assertRaises(JevError):
            c.ask(rerank_request("m", "q", ["a", "b"]))

    def test_reranker_sorts_ties_by_fused_rank_and_counts_tokens(self):
        reply = json.dumps({"model": "jev-1.14.0", "answers": {
            "c01": {"type": "noul", "noul": 0.2}, "c02": {"type": "noul", "noul": 0.9},
            "c03": {"type": "noul", "noul": 0.2}}, "usage": {"input_tokens": 1000, "output_tokens": 9}}).encode()
        jev = JevReranker(JevClient(None, transport=FakeTransport(script=[(200, {}, reply)]), env=ENV), "jev-1.14.0", 0.042, 1.0)
        scored, res = jev.score("q", [("a#1", "x"), ("b#1", "y"), ("c#1", "z")])
        self.assertEqual([s.chunk_id for s in scored], ["b#1", "a#1", "c#1"])
        self.assertEqual((jev.input_tokens, jev.billed_input_tokens, jev.output_tokens), (1000, 1000, 9))
        self.assertAlmostEqual(cost_usd(1_000_000, 0.042), 0.042)

    def test_budget_cap_stops_before_sending(self):
        tr = FakeTransport()
        jev = JevReranker(JevClient(None, transport=tr, env=ENV), "jev-1.14.0", 0.042, max_usd=1e-9)
        with self.assertRaises(BudgetExceeded):
            jev.score("q", [("a#1", "some passage text")])
        self.assertEqual(tr.calls, [])

    def test_cascade_grades_only_the_top_n(self):
        prov = ScriptedProvider(lambda m, msgs: ChatResult(text="3" if "gold" in msgs[-1]["content"] else "1", latency_s=0.1))
        texts = {c: ("gold passage" if c == "c#1" else "other") for c in ("a#1", "b#1", "c#1", "d#1", "e#1")}
        ranked, llm_s, pairs = cascade("q", ["a#1", "c#1", "b#1", "d#1"], ["d#1", "b#1", "a#1", "c#1", "e#1"], 4, 2,
                                       texts, LLMReranker(prov, "m", use_logprobs=False))
        self.assertEqual(pairs, 2)
        self.assertEqual(len(prov.calls), 2)
        self.assertAlmostEqual(llm_s, 0.2)
        self.assertEqual(ranked, ["c#1", "a#1", "b#1", "d#1", "e#1"])


class JevEvalTests(unittest.TestCase):
    def test_end_to_end_offline(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = P.load_config("config.toml")
            cfg["run"].update({"data_dir": f"{d}/data", "index_dir": f"{d}/index", "results_dir": f"{d}/results",
                               "cache": f"{d}/cache.sqlite"})
            cfg["retrieval"]["rerank_top_n"] = 6
            cfg["jev"].update({"cascade_n": 3, "cascade_sweep": [2, 3, 6]})
            quiet = lambda *_: None  # noqa: E731
            P.step_synth(cfg, log=quiet)
            P.step_index(cfg, log=quiet)
            prov = CachedProvider(ScriptedProvider(fake_llm), cfg["run"]["cache"])
            self.addCleanup(prov.close)

            # without the reference run in the cache, the LLM replay refuses to call the model
            tr = FakeTransport()
            jev = JevReranker(JevClient(cfg["run"]["cache"], transport=tr, env=ENV), cfg["jev"]["model"], 0.042, 1.0)
            self.addCleanup(jev.client.close)
            with self.assertRaises(ProviderError):
                step_jev_eval(cfg, prov, jev, log=quiet, limit=20, answers=False)
            self.assertTrue(prov.allow_miss)

            ref = P.step_eval(cfg, prov, log=quiet, answer_variants=["hybrid", "hybrid_rerank"], limit=20)
            calls_before = len(prov.inner.calls)

            # a cap below the estimate refuses before sending anything
            capped = JevReranker(JevClient(cfg["run"]["cache"], transport=FakeTransport(), env=ENV), cfg["jev"]["model"],
                                 0.042, 1e-9)
            self.addCleanup(capped.client.close)
            with self.assertRaises(BudgetExceeded):
                step_jev_eval(cfg, prov, capped, log=quiet, limit=20, answers=False)
            self.assertEqual(capped.client.transport.calls, [])

            out = step_jev_eval(cfg, prov, jev, log=quiet, limit=20, answers=True)
            rm = out["retrieval"]
            self.assertEqual(set(rm["variants"]), set(ALL_VARIANTS))
            for v in ("bm25", "dense", "hybrid", "hybrid_rerank"):
                self.assertEqual(rm["variants"][v]["recall@1"], ref["retrieval"]["variants"][v]["recall@1"], v)
            self.assertIn("hybrid_jev_llm_vs_hybrid_rerank", rm["comparisons"])
            self.assertIn("ndcg@10", rm["comparisons"]["hybrid_jev_rrf_vs_hybrid_jev"])
            # one Jev request per question, each carrying the fused top 6
            self.assertEqual(len(tr.calls), 20)
            self.assertTrue(all(len(c["body"]["questions"]) == 6 for c in tr.calls))
            # retrieval made no new LLM call (cascade grades are a subset of the reference top 6); only new answers did
            gen_calls = [c for c in prov.inner.calls[calls_before:] if c["max_tokens"] == 1]
            self.assertEqual(gen_calls, [])
            self.assertEqual([s["n"] for s in out["sweep"]], [2, 3, 6])
            full = out["sweep"][-1]
            self.assertEqual(full["recall@1"], rm["variants"]["hybrid_rerank"]["recall@1"])
            cost = out["cost"]
            self.assertEqual(cost["input_tokens"], sum(len(json.dumps(c["body"]).encode()) // 4 for c in tr.calls))
            self.assertAlmostEqual(cost["cost_usd"], round(cost["input_tokens"] * 0.042 / 1e6, 6))
            for v in ("hybrid", "hybrid_rerank") + JEV_VARIANTS:
                self.assertIn(v, out["answers"])
            report = write_report(cfg).read_text()
            self.assertIn("## Jev (TypeSafe) reranking", report)
            self.assertIn("Hybrid (RRF) + Jev prune + LLM rerank", report)
            self.assertIn("### Cascade", report)

            # a rerun replays Jev from the cache, with no key and no request
            jev2 = JevReranker(JevClient(cfg["run"]["cache"], transport=FakeTransport(), env={}, key_root=d),
                               cfg["jev"]["model"], 0.042, 1.0)
            self.addCleanup(jev2.client.close)
            out2 = step_jev_eval(cfg, prov, jev2, log=quiet, limit=20, answers=False)
            self.assertEqual(jev2.client.transport.calls, [])
            self.assertEqual(out2["retrieval"]["variants"]["hybrid_jev"], rm["variants"]["hybrid_jev"])
            self.assertEqual(out2["cost"]["input_tokens"], cost["input_tokens"])
            self.assertEqual(out2["cost"]["billed_this_process_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
