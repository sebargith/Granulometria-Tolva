from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_VERSION, extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from granulometria.metrics import compute_granulometry_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clasifica patches con un modelo entrenado.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    dataset_dir = args.dataset.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset_dir / "predictions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"Version de features incompatible: {bundle.get('feature_version')} != {FEATURE_VERSION}"
        )

    rows = pd.read_csv(dataset_dir / "patches.csv", encoding="utf-8-sig")
    feature_rows = []
    for index, row in enumerate(rows.itertuples(index=False), start=1):
        image_path = (dataset_dir / Path(str(row.patch_path))).resolve()
        feature_rows.append(extract_features(image_path))
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"Features: {index}/{len(rows)}")

    features = pd.DataFrame(feature_rows)[bundle["feature_names"]]
    model = bundle["model"]
    prediction = model.predict(features.to_numpy(dtype=np.float32))
    probabilities = model.predict_proba(features.to_numpy(dtype=np.float32))
    result = rows.copy()
    result["prediction"] = prediction
    for class_index, label in enumerate(model.classes_):
        result[f"prob_{label}"] = probabilities[:, class_index]
    result.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    probability_columns = [f"prob_{label}" for label in model.classes_]
    summary = result.groupby("source_image")[probability_columns].mean().reset_index()
    shares = (
        pd.crosstab(result["source_image"], result["prediction"], normalize="index")
        .add_prefix("share_")
        .reset_index()
    )
    summary = summary.merge(shares, on="source_image", how="left")
    patch_counts = result.groupby("source_image").size().rename("patch_count").reset_index()
    summary = summary.merge(patch_counts, on="source_image", how="left")
    summary["predominant"] = summary[probability_columns].idxmax(axis=1).str.removeprefix("prob_")
    for label in ("fino", "medio", "grueso"):
        column = f"share_{label}"
        if column not in summary.columns:
            summary[column] = 0.0
    if {"fino", "medio", "grueso"}.issubset(set(str(label) for label in model.classes_)):
        metrics = summary.apply(
            lambda row: pd.Series(
                compute_granulometry_metrics(
                    {
                        "fino": row["share_fino"],
                        "medio": row["share_medio"],
                        "grueso": row["share_grueso"],
                    }
                )
            ),
            axis=1,
        )
        summary["indice_granulometrico"] = metrics["indice_granulometrico"]
        summary["dispersion_granulometrica"] = metrics["dispersion_granulometrica"]
    summary.to_csv(output_dir / "summary_by_frame.csv", index=False, encoding="utf-8-sig")
    print(f"Predicciones: {output_dir / 'predictions.csv'}")
    print(f"Resumen por frame: {output_dir / 'summary_by_frame.csv'}")


if __name__ == "__main__":
    main()
