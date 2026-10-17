import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from basket_recommender import fit, export_model, recommend, score_basket, rank_indices, regular_ids, load_orders


def example():
    orders = pd.DataFrame([
        dict(date=pd.Timestamp("2026-01-01"), branch="A", customer="one", daypart="morning", products=("a", "b")),
        dict(date=pd.Timestamp("2026-01-02"), branch="A", customer="one", daypart="morning", products=("a", "b")),
        dict(date=pd.Timestamp("2026-01-03"), branch="A", customer="two", daypart="evening", products=("a", "c")),
        dict(date=pd.Timestamp("2026-01-04"), branch="A", customer=None, daypart="evening", products=("d",)),
    ])
    rows = []
    for o in orders.itertuples():
        for p in o.products:
            rows.append(dict(Date=o.date, Branch="A", Product=p, Category="Bakery", Quantity=1,
                **{"Line Gross Profit (USD)": {"a":1,"b":2,"c":3,"d":4}[p], "Line Revenue (USD)":5}))
    return pd.DataFrame(rows), orders


class BasketTests(unittest.TestCase):
    def test_training_boundary_rejects_future_sales(self):
        raw, orders = example()
        raw.loc[0,"Date"] = pd.Timestamp("2026-04-01")
        with self.assertRaisesRegex(ValueError,"April-June"):
            fit(raw, orders)

    def test_exported_ranking_matches_training_engine(self):
        raw, orders = example()
        model = fit(raw,orders,"shrunk_pairs",100)
        payload = json.loads(json.dumps(export_model(model)))
        for basket in [[],["a"],["a","b"]]:
            scores,_ = score_basket(model,"A","morning","all",basket)
            expected = [model["names"][i] for i in rank_indices(model["names"],scores,basket)][:3]
            actual = recommend(payload,"A",basket,"morning")["recommendations"]
            self.assertEqual(expected,[r["product"] for r in actual])

    def test_exclusions_empty_and_unknown_anchors(self):
        raw,orders = example()
        payload = export_model(fit(raw,orders))
        self.assertEqual(recommend(payload,"A",["a","b"],unavailable=["c","d"])["recommendations"],[])
        result = recommend(payload,"A",["unknown"],part="unknown")
        self.assertTrue(result["context_fallback"])
        self.assertEqual(len(result["recommendations"]),3)
        with self.assertRaises(ValueError):
            recommend(payload,"unknown")

    def test_probabilities_and_sparse_contexts(self):
        raw,orders = example()
        payload = export_model(fit(raw,orders))
        for ctx in payload["contexts"].values():
            for rows in ctx["by_anchor"].values():
                for r in rows:
                    self.assertGreaterEqual(r["conditional_probability"],0)
                    self.assertLessEqual(r["conditional_probability"],1)
                    self.assertTrue(np.isfinite(r["score"]))

    def test_complementary_mode_is_explicit_business_filter(self):
        raw,orders = example()
        raw.loc[raw.Product.eq("a"),"Category"]="Hot Coffee"
        raw.loc[raw.Product.eq("b"),"Category"]="Cold Coffee"
        payload=export_model(fit(raw,orders))
        rows=recommend(payload,"A",["a"],mode="complementary")["recommendations"]
        self.assertEqual({r["product"] for r in rows},{"c","d"})
        rows=recommend(payload,"A",["c"],mode="complementary")["recommendations"]
        self.assertEqual({r["product"] for r in rows},{"a","b"})

    def test_regular_definition_uses_distinct_dates_and_no_exported_ids(self):
        raw,orders = example()
        repeated = pd.concat([orders.iloc[[0]]]*10,ignore_index=True)
        self.assertEqual(regular_ids(repeated,threshold=2),set())
        payload = export_model(fit(raw,orders))
        self.assertNotIn('"customer"',json.dumps(payload))
        self.assertNotIn('"regular_ids"',json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
