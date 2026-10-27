import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from hybridrag.bm25 import BM25
from hybridrag.dense import (EmbedderError, LSAEmbedder, OllamaEmbedder, OpenAICompatibleEmbedder, VectorIndex,
                             build_embedder)
from hybridrag.fusion import rrf

DOCS = ["The meal per diem in the EU is EUR 90 per travel day.",
        "The meal per diem in the US is USD 75 per travel day.",
        "Error ING-5031 means the upload queue is full; run qvctl drain ingest-gateway.",
        "Battery life of the QV-PU200 is 12 months."]


def fake_response(status, payload):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = payload
    r.content = b"x"
    r.text = str(payload)
    return r


class BM25Tests(unittest.TestCase):
    def test_exact_code_wins(self):
        bm = BM25().fit(DOCS)
        self.assertEqual(bm.search("what does ING-5031 mean", 4)[0][0], 2)
        self.assertEqual(bm.search("QV-PU200 battery", 4)[0][0], 3)

    def test_score_matches_formula(self):
        bm = BM25(k1=1.2, b=0.75).fit(["apple banana", "apple apple cherry", "date"])
        n, df, tf, dl = 3, 2, 2, 3
        avgdl = (2 + 3 + 1) / 3
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        want = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))
        self.assertAlmostEqual(float(bm.scores("apple")[1]), want, places=5)
        self.assertEqual(bm.search("zzz", 3), [])


class FusionTests(unittest.TestCase):
    def test_rrf_scores_and_order(self):
        fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]], k=60))
        self.assertAlmostEqual(fused["a"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(fused["d"], 1 / 63)
        order = [x for x, _ in rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)]
        self.assertEqual(order[:2], ["a", "b"])  # tie on score: best single rank first, then list order
        self.assertEqual(set(order), {"a", "b", "c", "d"})


class DenseTests(unittest.TestCase):
    def test_lsa_roundtrip_and_search(self):
        e = LSAEmbedder(dims=3)
        e.fit(DOCS)
        v = e.embed(DOCS)
        self.assertEqual(v.shape, (4, 3))
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-5)
        idx = VectorIndex(v)
        self.assertEqual(idx.search(e.embed(["upload queue full ingest-gateway"], "query")[0], 2)[0][0], 2)
        with tempfile.TemporaryDirectory() as d:
            e.save(Path(d) / "emb")
            e2 = LSAEmbedder.load(Path(d) / "emb", 3)
            np.testing.assert_allclose(e2.embed(DOCS), v, rtol=1e-5)
            idx.save(Path(d) / "v.npy")
            np.testing.assert_allclose(VectorIndex.load(Path(d) / "v.npy").vectors, idx.vectors)

    def test_lsa_is_deterministic(self):
        a, b = LSAEmbedder(3), LSAEmbedder(3)
        a.fit(DOCS)
        b.fit(DOCS)
        np.testing.assert_array_equal(a.proj, b.proj)

    def test_ollama_embedder_success_and_refusal(self):
        e = OllamaEmbedder("nomic-embed-text", "http://h:1", batch_size=2, query_prefix="q: ")
        with mock.patch("hybridrag.dense.requests.post",
                        return_value=fake_response(200, {"embeddings": [[3.0, 4.0], [1.0, 0.0]]})) as post:
            v = e.embed(["a", "b"], kind="query")
        np.testing.assert_allclose(v[0], [0.6, 0.8])
        self.assertEqual(post.call_args.kwargs["json"], {"model": "nomic-embed-text", "input": ["q: a", "q: b"]})
        self.assertTrue(post.call_args.args[0].endswith("/api/embed"))
        refusal = {"error": "This server does not support embeddings. Start it with `--embeddings`"}
        with mock.patch("hybridrag.dense.requests.post", return_value=fake_response(500, refusal)):
            with self.assertRaises(EmbedderError) as ctx:
                e.embed(["a"])
        self.assertIn("does not support embeddings", str(ctx.exception))

    def test_openai_compatible_embedder(self):
        with mock.patch.dict("os.environ", {"HYBRIDRAG_OPENAI_BASE_URL": "http://x/v1", "HYBRIDRAG_OPENAI_API_KEY": "k"}):
            e = OpenAICompatibleEmbedder("emb-model")
        data = {"data": [{"index": 1, "embedding": [0.0, 2.0]}, {"index": 0, "embedding": [2.0, 0.0]}]}
        with mock.patch("hybridrag.dense.requests.post", return_value=fake_response(200, data)) as post:
            v = e.embed(["first", "second"])
        np.testing.assert_allclose(v, [[1.0, 0.0], [0.0, 1.0]])  # reordered by index
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer k")
        self.assertEqual(post.call_args.args[0], "http://x/v1/embeddings")
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(EmbedderError):
                OpenAICompatibleEmbedder("m")

    def test_build_embedder(self):
        self.assertIsInstance(build_embedder({"kind": "lsa", "dims": 8}), LSAEmbedder)
        self.assertIsInstance(build_embedder({"kind": "ollama", "model": "m"}), OllamaEmbedder)
        with self.assertRaises(EmbedderError):
            build_embedder({"kind": "nope"})


if __name__ == "__main__":
    unittest.main()
