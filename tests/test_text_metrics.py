import unittest

from hybridrag.metrics import (contaminated_at, fact_present, mrr_at, ndcg_at, paired_bootstrap, recall_at,
                               retrieval_row, unsupported_figures)
from hybridrag.text import extract_figures, find_citations, is_idk, split_sentences, strip_citations, tokenize


class TextTests(unittest.TestCase):
    def test_tokenize_keeps_codes_whole_and_emits_parts(self):
        toks = tokenize("What is the battery life of the QV-PU200 for ingest-gateway policies?")
        for t in ("qv-pu200", "qv", "pu200", "ingest-gateway", "ingest", "gateway", "battery", "policy"):
            self.assertIn(t, toks)
        self.assertNotIn("the", toks)

    def test_figures_numbers_and_identifiers(self):
        f = extract_figures("The QV-PU200 costs USD 1,250.00, is rated IP67, runs from -25 °C to 60 °C; colour #2E5E8C, "
                            "release 7.9, prefix RMA-EU.")
        self.assertEqual(f, {"QV-PU200", "1250", "IP67", "25", "60", "2E5E8C", "7.9", "RMA-EU"})

    def test_currency_and_unit_wording_do_not_matter(self):
        self.assertEqual(extract_figures("EUR 65 per day"), extract_figures("65 euros"))

    def test_citations(self):
        s = "The cap is EUR 150 [policy-travel-eu#1, policy-travel-eu#2]. Book early [policy-travel-eu#3]."
        self.assertEqual(find_citations(s), ["policy-travel-eu#1", "policy-travel-eu#2", "policy-travel-eu#3"])
        self.assertEqual(strip_citations("The cap is EUR 150 [policy-travel-eu#1]."), "The cap is EUR 150.")

    def test_split_sentences_keeps_decimals_and_trailing_citations(self):
        parts = split_sentences("The rate is EUR 0.35 per km. [policy-travel-eu#1] Version 7.3 fixed it [release-7-3#2].")
        self.assertEqual(parts, ["The rate is EUR 0.35 per km. [policy-travel-eu#1]", "Version 7.3 fixed it [release-7-3#2]."])

    def test_idk(self):
        self.assertTrue(is_idk("I don't know based on the provided documents."))
        self.assertTrue(is_idk("This is not mentioned in the provided documents."))
        self.assertFalse(is_idk("The meal per diem is EUR 90."))


class MetricTests(unittest.TestCase):
    def test_rank_metrics(self):
        ranked = ["a#0", "b#1", "g#2", "c#0"]
        gold = {"g#2"}
        self.assertEqual(recall_at(ranked, gold, 1), 0.0)
        self.assertEqual(recall_at(ranked, gold, 3), 1.0)
        self.assertAlmostEqual(mrr_at(ranked, gold), 1 / 3)
        self.assertAlmostEqual(ndcg_at(ranked, gold), 0.5)  # 1/log2(4) / 1

    def test_contamination(self):
        self.assertTrue(contaminated_at(["sib#0", "gold#1"], {"gold#1"}, {"sib"}, 5))
        self.assertFalse(contaminated_at(["gold#1", "sib#0"], {"gold#1"}, {"sib"}, 5))
        self.assertTrue(contaminated_at(["x#0", "sib#3"], {"gold#1"}, {"sib"}, 5))  # gold missing, sibling present
        self.assertFalse(contaminated_at(["x#0", "y#3"], {"gold#1"}, {"sib"}, 5))
        self.assertNotIn("contam@1", retrieval_row(["a#0"], {"a#0"}, set()))

    def test_fact_present(self):
        self.assertTrue(fact_present("90", "The per diem is EUR 90 per day [policy-travel-eu#1]."))
        self.assertFalse(fact_present("90", "The per diem is EUR 95 per day."))
        self.assertTrue(fact_present("31 March", "Use them by March 31."))
        self.assertFalse(fact_present("31 March", "Use them by 31 May."))
        self.assertTrue(fact_present("qvctl drain ingest-gateway", "Run `qvctl drain ingest-gateway` first."))
        self.assertFalse(fact_present("qvctl drain ingest-gateway", "Run qvctl restart ingest-gateway."))
        self.assertFalse(fact_present("90", "The cap is [policy-travel-eu#90]."))  # citation ids are not figures

    def test_unsupported_figures(self):
        doc = "Hotel bookings are capped at EUR 150 per night."
        self.assertEqual(unsupported_figures("The cap is EUR 150 [d#1].", doc), [])
        self.assertEqual(unsupported_figures("The cap is EUR 180 [d#1].", doc), ["180"])
        self.assertEqual(unsupported_figures("I don't know based on the provided documents.", ""), [])

    def test_paired_bootstrap(self):
        same = paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1])
        self.assertEqual(same["delta"], 0.0)
        self.assertEqual(same["ci95"], [0.0, 0.0])
        better = paired_bootstrap([0] * 30, [1] * 30)
        self.assertTrue(better["excludes_zero"])
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [1, 0])


if __name__ == "__main__":
    unittest.main()
