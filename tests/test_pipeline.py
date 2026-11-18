"""Verify the synthetic data contract and reusable analysis."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from src.generate_data import IMAGE_SHAPE, RAW_DATA_PATH, generate_transactions
from src.pipeline import DataAnalyzer


class TestDataAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generated = generate_transactions()

    def setUp(self):
        self.analyzer = DataAnalyzer(RAW_DATA_PATH)
        self.raw = self.analyzer.load_data()

    def test_generated_csv_contract_and_reproducibility(self):
        expected = self.generated.copy()
        expected["product_name"] = expected["product_name"].astype("string")
        actual = self.raw.copy()
        actual["product_name"] = actual["product_name"].astype("string")
        assert_frame_equal(actual, expected, check_dtype=False)
        assert_frame_equal(self.generated, generate_transactions())
        self.assertEqual(self.raw["customer_id"].nunique(), 500)
        self.assertEqual(self.raw["product_id"].nunique(), 100)
        self.assertGreaterEqual(self.raw["order_id"].nunique(), 3000)
        self.assertGreaterEqual(len(self.raw), 1000)
        self.assertGreaterEqual(len(self.raw.columns), 8)
        self.assertEqual(self.raw["category"].nunique(), 5)
        self.assertTrue(pd.api.types.is_numeric_dtype(self.raw["amount"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.raw["order_date"]))
        self.assertEqual(self.raw["order_date"].min(), pd.Timestamp("2025-01-01"))
        self.assertEqual(self.raw["order_date"].max(), pd.Timestamp("2025-12-31"))
        self.assertTrue(self.raw["amount"].eq(self.raw["price"] * self.raw["quantity"]).all())
        self.assertTrue(self.raw.groupby("order_id")["customer_id"].nunique().eq(1).all())

    def test_missing_names_are_recovered_by_product(self):
        before = self.analyzer.missing_counts()
        self.assertEqual(before["product_name"], 100)
        self.analyzer.handle_missing_values()
        self.assertEqual(self.analyzer.missing_counts().sum(), 0)
        self.assertTrue(self.analyzer.df.groupby("product_id")["product_name"].nunique().eq(1).all())
        self.assertEqual(self.analyzer.raw["product_name"].isna().sum(), 100)

    def test_numeric_imputation_with_global_fallback(self):
        self.analyzer.df = pd.DataFrame({
            "group": ["a", "a", "b", "b", "c"],
            "value": [10.0, np.nan, 30.0, 50.0, np.nan],
        })
        filled = self.analyzer.handle_missing_values("value", "group", "median")
        np.testing.assert_allclose(filled, [10, 10, 30, 50, 30])

    def test_handle_missing_matches_existing_method(self):
        original = DataAnalyzer(RAW_DATA_PATH)
        original.load_data()
        received = self.analyzer.handle_missing(
            column="product_name", group_col="product_id", strategy="mode"
        )
        assert_series_equal(received, original.handle_missing_values())
        assert_frame_equal(self.analyzer.df, original.df)
        self.assertEqual(self.analyzer.missing_counts().sum(), 0)

    def test_iqr_and_clipping_preserve_observed_values(self):
        self.assertEqual(self.analyzer.iqr_bounds("quantity"), (-2, 6))
        outliers = self.analyzer.detect_outliers()
        self.assertEqual(len(outliers), 24)
        self.assertTrue(outliers["quantity"].ge(20).all())
        self.assertEqual(self.analyzer.add_clipped_feature().max(), 6)
        assert_series_equal(self.analyzer.df["quantity"], self.analyzer.raw["quantity"])

    def test_image_and_text_features(self):
        self.analyzer.handle_missing_values()
        words = self.analyzer.add_text_features()
        self.assertTrue(words.between(3, 4).all())

        features = self.analyzer.add_image_features(image_shape=IMAGE_SHAPE)
        pixels = np.asarray(self.raw["image_array"].map(json.loads).tolist())
        self.assertEqual(pixels.shape, (len(self.raw), *IMAGE_SHAPE))
        np.testing.assert_allclose(features["image_mean"], pixels.mean(axis=(1, 2)))
        np.testing.assert_allclose(features["image_std"], pixels.std(axis=(1, 2)))

    def test_image_rejects_wrong_dimensions(self):
        self.analyzer.df.loc[0, "image_array"] = "[[1, 2]]"
        with self.assertRaises(ValueError):
            self.analyzer.add_image_features()

    def test_statistics_correlations_and_sales(self):
        stats = self.analyzer.descriptive_statistics()
        self.assertEqual(set(stats.columns), {"count", "mean", "std", "min", "25%", "median", "75%", "max"})
        self.assertAlmostEqual(stats.loc["amount", "mean"], self.raw["amount"].mean())
        self.assertEqual(stats.loc["price", "median"], self.raw["price"].median())
        corr = self.analyzer.correlations()
        self.assertAlmostEqual(corr.loc["quantity", "price"], self.raw["quantity"].corr(self.raw["price"]))
        self.assertAlmostEqual(corr.loc["quantity", "amount"], self.raw["quantity"].corr(self.raw["amount"]))
        monthly = self.analyzer.monthly_sales()
        self.assertEqual(len(monthly), 12)
        self.assertEqual(monthly.sum(), self.raw["amount"].sum())

    def test_rfm_covers_customers_and_reconciles_with_transactions(self):
        rfm = self.analyzer.calculate_rfm()
        segments = self.analyzer.summarize_segments(rfm)
        self.assertTrue(rfm.index.is_unique)
        self.assertEqual(set(rfm.index), set(self.raw["customer_id"]))
        self.assertEqual(segments["customers"].to_dict(), {
            "VIP": 89, "Loyal": 161, "New": 125, "Churned": 125,
        })
        self.assertEqual(rfm["frequency"].sum(), self.raw["order_id"].nunique())
        self.assertEqual(rfm["monetary"].sum(), self.raw["amount"].sum())
        self.assertEqual(rfm["recency"].min(), 0)
        self.assertTrue(rfm[["r_score", "f_score", "m_score"]].isin(range(1, 6)).all().all())
        with self.assertRaises(ValueError):
            self.analyzer.calculate_rfm(reference_date="2025-01-01")

    def test_rfm_dates_and_rules_at_boundaries(self):
        self.analyzer.df = pd.DataFrame({
            "client": ["old", "fresh", "active", "active", "active", "active"],
            "date": ["2025-10-02", "2025-12-01", "2025-11-30", "2025-12-04", "2025-12-31", "2025-12-31"],
            "order": ["1", "2", "3", "4", "5", "5"],
            "revenue": [1, 2, 10, 20, 30, 40],
        })
        rfm = self.analyzer.calculate_rfm(
            customer_col="client", date_col="date", amount_col="revenue", order_col="order"
        )
        self.assertEqual(rfm.loc["old", "recency"], 90)
        self.assertEqual(rfm.loc["old", "segment"], "Churned")
        self.assertEqual(rfm.loc["fresh", "segment"], "New")
        self.assertEqual(rfm.loc["active", "frequency"], 3)
        self.assertEqual(rfm.loc["active", "monetary"], 100)
        self.assertEqual(rfm.loc["active", "segment"], "VIP")

    def test_processed_file_keeps_raw_file_unchanged(self):
        before = RAW_DATA_PATH.read_bytes()
        with TemporaryDirectory() as temp:
            destination = Path(temp) / "processed.csv"
            self.analyzer.save_processed(destination)
            self.assertEqual(len(pd.read_csv(destination)), len(self.raw))
        with self.assertRaises(ValueError):
            self.analyzer.save_processed(RAW_DATA_PATH)
        self.assertEqual(RAW_DATA_PATH.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
