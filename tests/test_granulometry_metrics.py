from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from granulometria.metrics import compute_granulometry_metrics


class GranulometryMetricsTests(unittest.TestCase):
    def test_pure_classes_have_zero_dispersion(self) -> None:
        expected_indices = {"fino": 1.0, "medio": 2.0, "grueso": 3.0}
        for label, expected_index in expected_indices.items():
            with self.subTest(label=label):
                metrics = compute_granulometry_metrics({label: 1.0})
                self.assertAlmostEqual(metrics["indice_granulometrico"], expected_index)
                self.assertAlmostEqual(metrics["dispersion_granulometrica"], 0.0)

    def test_balanced_extremes_have_maximum_dispersion(self) -> None:
        metrics = compute_granulometry_metrics({"fino": 0.5, "grueso": 0.5})
        self.assertAlmostEqual(metrics["indice_granulometrico"], 2.0)
        self.assertAlmostEqual(metrics["dispersion_granulometrica"], 1.0)

    def test_mixed_distribution(self) -> None:
        metrics = compute_granulometry_metrics(
            {"fino": 0.2, "medio": 0.5, "grueso": 0.3}
        )
        self.assertAlmostEqual(metrics["indice_granulometrico"], 2.1)
        self.assertAlmostEqual(metrics["dispersion_granulometrica"], 0.7)

    def test_input_is_normalized(self) -> None:
        proportions = compute_granulometry_metrics(
            {"fino": 0.2, "medio": 0.5, "grueso": 0.3}
        )
        counts = compute_granulometry_metrics(
            {"fino": 20, "medio": 50, "grueso": 30}
        )
        self.assertEqual(proportions, counts)

    def test_empty_distribution_returns_none(self) -> None:
        metrics = compute_granulometry_metrics({})
        self.assertIsNone(metrics["indice_granulometrico"])
        self.assertIsNone(metrics["dispersion_granulometrica"])


if __name__ == "__main__":
    unittest.main()
