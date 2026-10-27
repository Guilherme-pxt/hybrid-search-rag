import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from hybridrag import pipeline as P
from hybridrag.chunking import chunk_corpus, gold_chunk_ids
from hybridrag.cli import main as cli_main
from hybridrag.evalgen_bridge import to_eval_items, to_predictions
from hybridrag.generate import IDK
from hybridrag.metrics import fact_present
from hybridrag.providers import ChatResult, ScriptedProvider
from hybridrag.report import write_report
from hybridrag.synth import generate, write_dataset


class SynthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.questions = generate(42)
        cls.ddicts = [d.to_dict() for d in cls.docs]

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            write_dataset(a, 42)
            write_dataset(b, 42)
            for f in ("corpus.jsonl", "questions.jsonl"):
                self.assertEqual(Path(a, f).read_bytes(), Path(b, f).read_bytes())
        _, other = generate(7)
        self.assertNotEqual([q.question for q in other], [q.question for q in self.questions])

    def test_sizes_and_types(self):
        self.assertTrue(80 <= len(self.docs) <= 150)
        self.assertTrue(100 <= len(self.questions) <= 150)
        types = {q.qtype for q in self.questions}
        self.assertEqual(types, {"exact_code", "lexical", "paraphrase", "unanswerable"})
        self.assertEqual(len({q.qid for q in self.questions}), len(self.questions))

    def test_every_gold_fact_is_in_its_evidence_and_its_chunk(self):
        chunks = chunk_corpus(self.ddicts, 50)
        by_doc = {}
        for c in chunks:
            by_doc.setdefault(c.doc_id, []).append(c)
        for q in self.questions:
            qd = q.to_dict()
            if q.qtype == "unanswerable":
                self.assertEqual(gold_chunk_ids(qd, by_doc), [])
                continue
            ids = gold_chunk_ids(qd, by_doc)
            self.assertTrue(ids, q.qid)
            chunk_text = {c.chunk_id: c.text for c in chunks}
            self.assertTrue(any(fact_present(q.gold_fact, chunk_text[i]) for i in ids), q.qid)

    def test_siblings_disagree_on_the_gold_fact(self):
        """The near-duplicate distractors must state a different value, or wrong-entity errors would be invisible."""
        text = {d.doc_id: d.text for d in self.docs}
        for q in self.questions:
            if q.qtype == "unanswerable" or not q.distractor_doc_ids:
                continue
            for sib in q.distractor_doc_ids:
                sib_sents = [s for s in re.split(r"(?<=\.)\s+", text[sib]) if s and q.evidence.split(" ")[0] in s]
                self.assertFalse(any(s.strip() == q.evidence for s in sib_sents), (q.qid, sib))

    def test_unanswerable_entities_absent(self):
        corpus = " ".join(d.text + d.title for d in self.docs)
        for word in ("Brazil", "Canada", "India", "LATAM", "QV-PU400", "geo-indexer", "8.1", "Business plan", "Premium",
                     "Berlin", "dental", "sabbatical"):
            self.assertNotIn(word, corpus)


def fake_llm(model, messages):
    """Reranker: grade 3 when the passage repeats a question word with a digit, else 1. Generator: cite passage 1."""
    system = messages[0]["content"]
    user = messages[-1]["content"]
    if system.startswith("You grade"):
        return ChatResult(text="2", latency_s=0.01, top_logprobs=[{"2": -0.1, "3": -2.5}])
    passages = re.findall(r"\[([^\]]+#\d+)\]\n(.*?)(?=\n\n\[|\n\nQuestion:)", user, re.S)
    if not passages:
        return ChatResult(text=IDK, latency_s=0.02)
    cid, text = passages[0]
    body = text.split("\n", 1)[-1]
    sent = re.split(r"(?<=\.)\s+", body.split(": ", 1)[-1])[0].rstrip(".")
    return ChatResult(text=f"{sent} [{cid}]. It also costs EUR 12345 [{cid}].", latency_s=0.05)


class PipelineTests(unittest.TestCase):
    def test_end_to_end_offline(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = P.load_config("config.toml")
            cfg["run"].update({"data_dir": f"{d}/data", "index_dir": f"{d}/index", "results_dir": f"{d}/results"})
            cfg["retrieval"]["rerank_top_n"] = 5
            P.step_synth(cfg, log=lambda *_: None)
            meta = P.step_index(cfg, log=lambda *_: None)
            self.assertGreater(meta["chunks"], meta["documents"])
            prov = ScriptedProvider(fake_llm)
            out = P.step_eval(cfg, prov, log=lambda *_: None, answer_variants=["bm25", "hybrid_rerank"], limit=30)
            rm = out["retrieval"]
            self.assertEqual(set(rm["variants"]), set(P.VARIANTS))
            for v in P.VARIANTS:
                for m in ("recall@1", "recall@5", "mrr@10", "ndcg@10"):
                    self.assertTrue(0.0 <= rm["variants"][v][m] <= 1.0)
            self.assertIn("hybrid_rerank_vs_hybrid", rm["comparisons"])
            am = out["answers"]["hybrid_rerank"]
            # the fake generator invents "EUR 12345" in every answer: the verifier must remove it everywhere
            self.assertGreater(am["gold_unsupported_raw"], 0.9)
            self.assertLess(am["gold_unsupported_verified"], am["gold_unsupported_raw"])
            recs = P.read_jsonl(f"{d}/results/answers_hybrid_rerank.jsonl")
            self.assertTrue(all("12345" not in r["answer_verified"] for r in recs))
            # rerank calls come first and ask for one token with logprobs
            self.assertEqual(prov.calls[0]["max_tokens"], 1)
            self.assertGreater(prov.calls[0]["top_logprobs"], 0)
            report = write_report(cfg).read_text()
            self.assertIn("Hybrid (RRF) + LLM rerank", report)
            self.assertIn("## Latency", report)
            for f in ("retrieval_metrics.json", "answer_metrics.json", "latency.json", "retrieval_per_question.jsonl"):
                self.assertTrue(Path(d, "results", f).exists(), f)

            items = to_eval_items(P.read_jsonl(f"{d}/data/questions.jsonl"))
            self.assertEqual({i["expected"]["action"] for i in items}, {"answer", "decline"})
            preds = to_predictions(recs, "hybrid_rerank", "fake")
            self.assertTrue(all(p["response"]["action"] in ("answer", "decline") for p in preds))
            self.assertFalse(any(re.search(r"\[[^\]]*#\d+\]", p["response"]["message"]) for p in preds))

    def test_cli_synth_and_index(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "c.toml"
            text = Path("config.toml").read_text()
            text = text.replace('data_dir = "data"', f'data_dir = "{d}/data"').replace(
                'index_dir = "index"', f'index_dir = "{d}/index"')
            cfg_path.write_text(text)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main(["--config", str(cfg_path), "synth"]), 0)
                self.assertEqual(cli_main(["--config", str(cfg_path), "index"]), 0)
            meta = json.loads(Path(d, "index", "meta.json").read_text())
            self.assertEqual(meta["embedder"], "lsa-128")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(cli_main(["--config", str(cfg_path), "retrieve", "QV-PU200 battery", "--variant", "bm25"]), 0)
            self.assertIn("[spec-pulse-pro#", buf.getvalue().splitlines()[0])


if __name__ == "__main__":
    unittest.main()
