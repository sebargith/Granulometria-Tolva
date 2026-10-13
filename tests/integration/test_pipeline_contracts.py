"""Synthetic contract tests against real helpers; model calls are stubbed."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modelos granulometria/clasificador_patches"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inference = load("inference", "modelos granulometria/inferencia_tolva/predict_granulometria_tolva.py")
rf = load("rf", "modelos granulometria/clasificador_patches/train_random_forest.py")
compiler = load("compiler", "scripts/labeling/compile_grid_ai_labels.py")
CLASSES = ["fino", "medio", "grueso"]


class PipelineContractTests(unittest.TestCase):
    def test_normalization_preserves_aspect_and_excludes_background(self):
        image = np.full((100, 200, 3), 180, dtype=np.uint8)
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[20:80, 20:180] = 255
        mask[20:35, 20:40] = 0
        bbox = inference.mask_bbox(mask)
        normalized, resized_mask, scale = inference.crop_and_normalize(image, mask, bbox, 320, "black")
        self.assertEqual(normalized.shape, (120, 320, 3))
        self.assertEqual(scale, 2)
        self.assertTrue(np.all(normalized[resized_mask == 0] == 0))
        self.assertTrue(np.all(normalized[resized_mask > 0] == 180))
        self.assertIsNone(inference.mask_bbox(np.zeros_like(mask)))

    def test_grid_boundary_behavior(self):
        self.assertEqual(inference.sliding_positions(95, 96, 96), [])
        self.assertEqual(inference.sliding_positions(192, 96, 96), [0, 96])
        positions = inference.sliding_positions(896, 96, 96)
        self.assertEqual(positions[-1], 800)
        self.assertLess(positions[-1] - positions[-2], 96)

    def test_validation_catches_incomplete_and_invalid_responses(self):
        expected = pd.DataFrame({"key": ["f|001", "f|002"]})
        responses = pd.DataFrame([
            dict(key="f|001", frame_folder="f", cell_id="001", response_file="labels.csv",
                 label="fino", confidence="alta"),
            dict(key="f|001", frame_folder="f", cell_id="001", response_file="labels.csv",
                 label="bad", confidence="bad"),
        ])
        report = compiler.validation_report(expected, responses)
        self.assertEqual(report["missing_cells"], 1)
        self.assertEqual(report["duplicate_rows"], 2)
        self.assertEqual(report["invalid_labels"], 1)
        self.assertEqual(report["invalid_confidence"], 1)

    def test_response_ids_and_dataset_confidence_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pd.DataFrame([dict(frame_folder="f", image_name="f.png", cell_id="1", label="FINO",
                               confidence="ALTA", notes="")]).to_csv(directory / "labels_batch004.csv", index=False)
            response = compiler.load_responses(directory)
            self.assertEqual(response.iloc[0]["key"], "f|001")
            cv2.imwrite(str(directory / "patch.png"), np.zeros((96,96,3), dtype=np.uint8))
            pd.DataFrame([dict(patch_id=str(i), patch_path="patch.png", source_image="f.png")
                          for i in range(3)]).to_csv(directory / "patches.csv", index=False)
            pd.DataFrame([dict(patch_id="0", label="fino", confidence="alta"),
                          dict(patch_id="1", label="medio", confidence="baja"),
                          dict(patch_id="2", label="invalido", confidence="alta")]
                         ).to_csv(directory / "labels.csv", index=False)
            rows = rf.load_dataset(directory, CLASSES, ["alta", "media"])
            self.assertEqual(rows["patch_id"].tolist(), ["0"])

    def test_group_split_and_reuse(self):
        rows = pd.DataFrame([dict(group=f"frame_{i}", label=label)
                             for i in range(20) for label in CLASSES])
        assignments = rf.create_group_split(rows, CLASSES, 42, .15, .15, 3)
        rows["split"] = assignments
        self.assertTrue((rows.groupby("group")["split"].nunique() == 1).all())
        for split in ("train", "val", "test"):
            self.assertEqual(set(rows.loc[rows["split"] == split, "label"]), set(CLASSES))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.csv"
            rows[["group", "split"]].drop_duplicates().to_csv(path, index=False)
            self.assertEqual(rf.reuse_group_split(rows, path).tolist(), assignments.tolist())
            self.assertEqual(rf.reuse_group_split(pd.DataFrame({"group": ["new_frame"]}), path).tolist(), ["train"])

    def test_image_summary_and_empty_frame(self):
        kwargs = dict(image_path=Path("synthetic.png"), width=192, height=96, detections=1,
                      max_yolo_conf=.9, mean_yolo_conf=.9, mask_area_px=10000,
                      candidate_count=1, rejected_patches=1, model_classes=CLASSES)
        result = inference.summary_for_image(patch_rows=[dict(prediction="fino", confidence=.8),
                                                         dict(prediction="grueso", confidence=.9)], **kwargs)
        self.assertEqual(result["share_fino"], .5)
        self.assertEqual(result["dispersion_granulometrica"], 1)
        self.assertEqual(result["mean_patch_confidence"], .85)
        empty = inference.summary_for_image(patch_rows=[], **kwargs)
        self.assertEqual(empty["dominant_label"], "")
        self.assertEqual(empty["indice_granulometrico"], "")

    def test_pipeline_coverage_gate_and_exports(self):
        # Two 96 px windows: the first straddles the 80% gate, the second is empty.
        for polygon_right, expected_count in ((76, 1), (75, 0)):
            with self.subTest(polygon_right=polygon_right), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                image = np.full((96, 192, 3), 120, dtype=np.uint8)
                source = root / "synthetic.png"
                cv2.imwrite(str(source), image)
                weights = root / "stub.pt"
                weights.touch()
                argv = ["predict", "--source", str(source), "--patch-model", str(weights),
                        "--yolo-model", str(weights), "--output-dir", str(root / "runs"),
                        "--run-name", "test", "--normalize-width", "0"]
                polygon = np.array([[0,0], [polygon_right,0], [polygon_right,95], [0,95]])
                result = SimpleNamespace(orig_img=image, masks=SimpleNamespace(xy=[polygon]), boxes=None)
                yolo = SimpleNamespace(predict=lambda **kwargs: iter([result]))
                stub = SimpleNamespace(YOLO=lambda path: yolo)
                bundle = {"kind": "cnn", "classes": CLASSES}
                def predict(bundle, patches):
                    return np.array(["medio"] * len(patches)), np.tile([.1,.8,.1], (len(patches),1))
                with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"ultralytics": stub}), \
                     patch.object(inference, "load_patch_model", return_value=bundle), \
                     patch.object(inference, "predict_patch_batch", side_effect=predict):
                    inference.main()
                summary = pd.read_csv(root / "runs/test/summary.csv")
                self.assertEqual(int(summary.iloc[0]["patch_count"]), expected_count)
                self.assertEqual(int(summary.iloc[0]["rejected_patches"]), 2-expected_count)
                self.assertTrue((root / "runs/test/details_por_patch.csv").is_file())
                self.assertEqual(len(list((root / "runs/test/overlays_granulometria").glob("*.jpg"))), 1)


if __name__ == "__main__":
    unittest.main()
