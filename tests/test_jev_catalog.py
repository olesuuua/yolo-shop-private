"""Compact Jev request construction without making network calls."""

import unittest

from jev_catalog import build_request, load_catalog, select_candidates, validate_answer


class JevCatalogTests(unittest.TestCase):
    def test_removed_saint_spring_sku_is_not_selectable(self):
        products = load_catalog()
        self.assertNotIn("saint-spring-0-33l", {p["sku"] for p in products})
        self.assertIn("saint-spring-0-75l", {p["sku"] for p in products})
        payload = build_request(
            [{"text": "СВЯТОЙ ИСТОЧНИК", "score": 0.9}], products, "Bottle")
        criteria = payload["questions"]["product"]["criteria"]
        self.assertNotIn("saint-spring-0-33l", criteria)
        self.assertIn("saint-spring-0-75l", criteria)
        probabilities = dict.fromkeys(criteria, 0.0)
        probabilities["saint-spring-0-33l"] = 1.0
        response = {"model": "test", "answers": {"product": {
            "choice": "saint-spring-0-33l", "confidence": 1.0,
            "probabilities": probabilities}}}
        with self.assertRaises(ValueError):
            validate_answer(response, payload)

    def test_request_contains_only_ocr_comparable_product_fields(self):
        payload = build_request(
            [{"text": "СЕНЕЖСКАЯ", "score": 0.9}], load_catalog(), "Bottle")
        criteria = payload["questions"]["product"]["criteria"]
        expected = {"name", "brand", "category", "variant", "size", "aliases", "verified_label_text"}
        for sku, product in criteria.items():
            if sku in {"other_product", "insufficient_evidence"}:
                continue
            self.assertEqual(set(product), expected)
            self.assertNotIn("packaging", product)
            self.assertNotIn("identity_clues", product)
            self.assertNotIn("cautions", product)

    def test_shortlist_ranks_text_within_the_whole_packaging_class(self):
        products=[]
        for index in range(25):
            products.append({
                "sku":f"water-{index}", "name":f"Water {index}", "brand":"WaterCo",
                "category":"Water", "variant":f"flavor {index}", "size":"500 ml",
                "aliases":[], "verified_label_text":[f"WATER FLAVOR {index}"],
                "object_classes":["Bottle"],
            })
        products.append({
            "sku":"milk-special", "name":"Farm Milk", "brand":"MooFarm",
            "category":"Milk", "variant":"banana", "size":"500 ml",
            "aliases":[], "verified_label_text":["MOOFARM BANANA"],
            "object_classes":["Bottle"],
        })
        products.append({
            "sku":"cat-can", "name":"Cat Food", "brand":"CatCo",
            "category":"Cat food", "variant":"salmon", "size":"80 g",
            "aliases":[], "verified_label_text":["CATCO SALMON"],
            "object_classes":["Tin can"],
        })
        selected=select_candidates(
            [{"text":"MOOFARM BANANA","score":0.8}],products,"Bottle")
        self.assertEqual(len(selected),20)
        self.assertEqual(selected[0]["sku"],"milk-special")
        self.assertNotIn("cat-can",{product["sku"] for product in selected})

    def test_incomplete_class_metadata_is_not_excluded(self):
        products=[
            {"sku":"bottle","object_classes":["Bottle"]},
            {"sku":"unclassified"},
            {"sku":"can","object_classes":["Tin can"]},
        ]
        selected=select_candidates([],products,"Bottle",limit=20)
        self.assertEqual([product["sku"] for product in selected],
                         ["bottle","unclassified"])


if __name__ == "__main__":
    unittest.main()
