import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts/demo_portfolio.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class PublicDemoTests(unittest.TestCase):
    def test_recorded_metrics_match_confusion_counts(self):
        data = json.loads((ROOT / "docs/public/results.json").read_text())
        for experiment in data["experiments"]:
            with self.subTest(experiment=experiment["id"]):
                m = experiment["matrix"]
                total = sum(map(sum, m))
                f1 = sum(2*m[i][i] / (sum(m[i]) + sum(row[i] for row in m))
                         for i in range(3)) / 3
                self.assertAlmostEqual(experiment["accuracy"], sum(m[i][i] for i in range(3))/total)
                self.assertAlmostEqual(experiment["test_f1_macro"], f1)
        ds = data["dataset"]
        self.assertEqual(ds["annotations"]-ds["invalid"]-ds["low_confidence_excluded"], ds["usable_patches"])
        self.assertEqual(sum(ds["split_patches"]), ds["usable_patches"])

    def test_pooling_weights_by_patch_count(self):
        # One fine cell and three coarse cells: pooled IG 2.5, not frame mean 2.
        stats = demo.summarize(["fino"] + ["grueso"]*3)
        self.assertEqual(stats["shares"]["grueso"], .75)
        self.assertEqual(stats["indice_granulometrico"], 2.5)
        self.assertIsNone(demo.summarize([])["indice_granulometrico"])

    def test_unknown_class_is_rejected(self):
        with self.assertRaises(ValueError):
            demo.summarize(["not_a_class"])

    def test_cli_from_different_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.html"
            subprocess.run([sys.executable, str(ROOT / "scripts/demo_portfolio.py"),
                            "--output", str(target)], cwd=tmp, check=True, capture_output=True)
            text = target.read_text(encoding="utf-8")
            self.assertIn("Invented labels, not model predictions", text)
            self.assertIn("76.7%", text)
            self.assertIn("Not evaluable", text)


if __name__ == "__main__":
    unittest.main()
