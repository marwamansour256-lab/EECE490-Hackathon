from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from wejha_layout import distance_score, optimize, validate_layout


class WejhaTests(unittest.TestCase):
    def setUp(self):
        self.products=["a","b","c","d"]
        self.pairs=[dict(item_a="a",item_b="b",weight=10),dict(item_a="c",item_b="d",weight=1)]

    def test_separation_and_adjacency_have_expected_geometry(self):
        low,_=optimize(self.products,self.pairs,False)
        high,_=optimize(self.products,self.pairs,True)
        self.assertAlmostEqual(distance_score(low,self.pairs),1/3)
        self.assertAlmostEqual(distance_score(high,self.pairs),(10+1/3)/11)
        self.assertEqual(abs(high.index("a")-high.index("b")),3)

    def test_fixed_slots_and_storage_zones_survive_optimization(self):
        constraints=dict(fixed={0:"a"},zones=["ambient","ambient","cold","cold"],
            eligible_zones={"a":["ambient"],"b":["ambient"],"c":["cold"],"d":["cold"]})
        layout,_=optimize(self.products,self.pairs,True,**constraints)
        self.assertTrue(validate_layout(layout,self.products,**constraints))
        self.assertEqual(layout[0],"a")
        self.assertEqual(set(layout[:2]),{"a","b"})

    def test_invalid_duplicate_and_infeasible_constraints_fail(self):
        with self.assertRaises(ValueError):
            validate_layout(["a","a","c","d"],self.products)
        with self.assertRaises(ValueError):
            optimize(self.products,self.pairs,fixed={0:"unknown"})
        with self.assertRaises(ValueError):
            optimize(self.products,self.pairs,zones=["cold"]*4,eligible_zones={p:["ambient"] for p in self.products})

    def test_score_is_scale_invariant_and_bounded(self):
        scaled=[dict(p,weight=p["weight"]*100) for p in self.pairs]
        self.assertAlmostEqual(distance_score(self.products,self.pairs),distance_score(self.products,scaled))
        self.assertGreaterEqual(distance_score(self.products,self.pairs),0)
        self.assertLessEqual(distance_score(self.products,self.pairs),1)


if __name__ == "__main__":
    unittest.main()
