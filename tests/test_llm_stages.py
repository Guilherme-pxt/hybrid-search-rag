import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hybridrag.generate import IDK, answer_messages
from hybridrag.providers import (CachedProvider, ChatResult, OllamaProvider, OpenAICompatibleProvider, ProviderError,
                                 ScriptedProvider)
from hybridrag.rerank import LLMReranker, grade_from
from hybridrag.verify import verify

CHUNKS = {
    "policy-travel-eu#1": "Travel and Expense Policy (EU)\nMeals: The meal per diem is EUR 90 per travel day.",
    "policy-travel-us#1": "Travel and Expense Policy (US)\nMeals: The meal per diem is USD 75 per travel day.",
}


def fake_response(status, payload):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = payload
    r.text = str(payload)
    return r


class ProviderTests(unittest.TestCase):
    def test_ollama_payload_and_logprobs(self):
        p = OllamaProvider("http://h:1", temperature=0.0, seed=7, num_ctx=2048)
        body = p.payload("m", [{"role": "user", "content": "hi"}], 1, 5)
        self.assertEqual(body["options"], {"temperature": 0.0, "seed": 7, "num_ctx": 2048, "num_predict": 1})
        self.assertIs(body["think"], False)
        self.assertTrue(body["logprobs"])
        data = {"message": {"content": "3"}, "logprobs": [{"token": "3", "logprob": -0.1,
                                                           "top_logprobs": [{"token": "3", "logprob": -0.1},
                                                                            {"token": "2", "logprob": -2.4}]}],
                "prompt_eval_count": 200, "eval_count": 1}
        with mock.patch("hybridrag.providers.requests.post", return_value=fake_response(200, data)):
            res = p.chat("m", [{"role": "user", "content": "hi"}], 1, 5)
        self.assertEqual(res.text, "3")
        self.assertEqual(res.top_logprobs, [{"3": -0.1, "2": -2.4}])
        with mock.patch("hybridrag.providers.requests.post", return_value=fake_response(404, {"error": "no model"})):
            with self.assertRaises(ProviderError):
                p.chat("m", [], 1)

    def test_openai_compatible_chat(self):
        p = OpenAICompatibleProvider(base_url="http://x/v1", api_key="secret")
        data = {"choices": [{"message": {"content": "2"}, "logprobs": {"content": [
            {"token": "2", "logprob": -0.2, "top_logprobs": [{"token": "2", "logprob": -0.2}]}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
        with mock.patch("hybridrag.providers.requests.post", return_value=fake_response(200, data)) as post:
            res = p.chat("gpt-x", [{"role": "user", "content": "q"}], max_tokens=1, top_logprobs=3)
        kw = post.call_args.kwargs
        self.assertEqual(post.call_args.args[0], "http://x/v1/chat/completions")
        self.assertEqual(kw["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(kw["json"]["temperature"], 0.0)
        self.assertEqual(kw["json"]["top_logprobs"], 3)
        self.assertEqual(res.text, "2")
        self.assertEqual(res.top_logprobs, [{"2": -0.2}])
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ProviderError):
                OpenAICompatibleProvider()

    def test_cache_hit_keeps_original_latency(self):
        inner = ScriptedProvider(lambda m, msgs: ChatResult(text="ok", latency_s=1.5))
        with tempfile.TemporaryDirectory() as d:
            c = CachedProvider(inner, Path(d) / "c.sqlite")
            a = c.chat("m", [{"role": "user", "content": "x"}])
            b = c.chat("m", [{"role": "user", "content": "x"}])
            c.chat("m", [{"role": "user", "content": "y"}])
            c.close()
        self.assertFalse(a.cached)
        self.assertTrue(b.cached)
        self.assertEqual(b.latency_s, 1.5)
        self.assertEqual(len(inner.calls), 2)
        self.assertEqual((c.hits, c.misses), (1, 2))


class RerankTests(unittest.TestCase):
    def test_grade_from_logprobs_is_expected_value(self):
        g = grade_from("3", [{"3": math.log(0.75), "2": math.log(0.25), "x": -1.0}])
        self.assertAlmostEqual(g, 2.75)
        self.assertEqual(grade_from("Grade: 2", []), 2.0)
        self.assertIsNone(grade_from("n/a", []))

    def test_rerank_orders_by_grade_then_fused_rank(self):
        def fn(model, msgs):
            return "3" if "EUR 90" in msgs[-1]["content"] else "1"

        r = LLMReranker(ScriptedProvider(fn), "m", use_logprobs=False)
        graded, _ = r.rerank("EU meal per diem?", [("policy-travel-us#1", CHUNKS["policy-travel-us#1"]),
                                                   ("other#0", "x"), ("policy-travel-eu#1", CHUNKS["policy-travel-eu#1"])])
        self.assertEqual([g.chunk_id for g in graded], ["policy-travel-eu#1", "policy-travel-us#1", "other#0"])
        self.assertEqual(graded[0].fused_rank, 3)
        self.assertIn("Question: EU meal per diem?", r.provider.calls[0]["messages"][1]["content"])
        self.assertEqual(r.provider.calls[0]["max_tokens"], 1)


class VerifierTests(unittest.TestCase):
    def test_supported_answer_passes_unchanged(self):
        v = verify("The meal per diem is EUR 90 per travel day [policy-travel-eu#1].", CHUNKS)
        self.assertEqual((v.flagged, v.claims, v.removed), (0, 1, 0))
        self.assertEqual(v.final_text, "The meal per diem is EUR 90 per travel day [policy-travel-eu#1].")

    def test_flags_wrong_figure_missing_and_unknown_citations(self):
        ans = ("The meal per diem is EUR 95 per travel day [policy-travel-eu#1]. "
               "Hotels are capped at EUR 150 per night. "
               "Receipts are needed for taxis above EUR 20 [policy-travel-eu#7].")
        v = verify(ans, CHUNKS, remove=False)
        reasons = [s.reasons for s in v.sentences]
        self.assertIn("figures not in the cited chunks: 95", reasons[0])
        self.assertIn("no citation", reasons[1])
        self.assertTrue(any("not in the context" in r for r in reasons[2]))
        self.assertEqual(v.flagged, 3)
        self.assertEqual(v.final_text, ans)

    def test_wrong_entity_with_valid_citation_is_not_caught(self):
        # documented blind spot: the claim is faithful to the (wrong) chunk it cites
        v = verify("The meal per diem is USD 75 per travel day [policy-travel-us#1].", CHUNKS)
        self.assertEqual(v.flagged, 0)

    def test_removal_and_caveat(self):
        ans = "The meal per diem is EUR 90 [policy-travel-eu#1]. It rises to EUR 120 in capitals [policy-travel-eu#1]."
        v = verify(ans, CHUNKS)
        self.assertEqual(v.final_text, "The meal per diem is EUR 90 [policy-travel-eu#1].")
        self.assertIn("1 statement removed", v.caveat)
        self.assertIn("1 statement removed", v.shown_text)

    def test_everything_removed_becomes_idk_and_idk_is_exempt(self):
        v = verify("The meal per diem is EUR 99 [policy-travel-eu#1].", CHUNKS)
        self.assertEqual(v.final_text, IDK)
        v = verify(IDK, CHUNKS)
        self.assertEqual((v.flagged, v.claims), (0, 0))

    def test_low_overlap_claim_without_figures(self):
        v = verify("Employees must always fly business class on long routes [policy-travel-eu#1].", CHUNKS, remove=False)
        self.assertEqual(v.flagged, 1)
        self.assertIn("low word overlap", v.sentences[0].reasons[0])

    def test_answer_prompt_lists_chunk_ids(self):
        msgs = answer_messages("q?", list(CHUNKS.items()))
        self.assertIn("[policy-travel-eu#1]\nTravel and Expense Policy (EU)", msgs[1]["content"])
        self.assertIn(IDK, msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()
