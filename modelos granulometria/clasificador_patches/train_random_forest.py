from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

from features import FEATURE_VERSION, extract_features


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_DATASET = PROJECT_DIR / "outputs" / "patches" / "data_parches_test_carga_128"
DEFAULT_OUTPUT = SCRIPT_DIR / "runs"
DEFAULT_CLASSES = ("fino", "medio", "grueso")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un Random Forest para clasificar patches granulometricos."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        default=[DEFAULT_DATASET],
        help="Una o mas carpetas con patches.csv, labels.csv e images/.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument(
        "--confidence",
        nargs="+",
        default=["alta", "media"],
        help="Niveles de confianza admitidos.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument(
        "--min-class-holdout",
        type=int,
        default=3,
        help="Minimo de patches por clase exigido tanto en val como en test.",
    )
    parser.add_argument(
        "--reuse-split",
        type=Path,
        default=None,
        help="group_split.csv anterior. Los grupos nuevos se asignan a train.",
    )
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def next_run_dir(output_dir: Path, run_name: str | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        candidate = output_dir / run_name
        if candidate.exists():
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            candidate = output_dir / f"{run_name}_{suffix}"
        candidate.mkdir(parents=True)
        return candidate

    versions = []
    for path in output_dir.glob("v*_random_forest"):
        try:
            versions.append(int(path.name[1:4]))
        except ValueError:
            continue
    version = max(versions, default=0) + 1
    candidate = output_dir / f"v{version:03d}_random_forest"
    candidate.mkdir(parents=True)
    return candidate


def load_dataset(dataset_dir: Path, classes: list[str], confidence: list[str]) -> pd.DataFrame:
    dataset_dir = dataset_dir.expanduser().resolve()
    manifest_path = dataset_dir / "patches.csv"
    labels_path = dataset_dir / "labels.csv"
    if not manifest_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"El dataset requiere patches.csv y labels.csv: {dataset_dir}"
        )

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype={"patch_id": str})
    labels = pd.read_csv(labels_path, encoding="utf-8-sig", dtype={"patch_id": str})
    label_columns = ["patch_id", "label", "confidence", "labeled_at"]
    labels = labels[[column for column in label_columns if column in labels.columns]]
    labels = labels.drop_duplicates("patch_id", keep="last")
    rows = manifest.merge(labels, on="patch_id", how="inner")
    rows = rows[rows["label"].isin(classes) & rows["confidence"].isin(confidence)].copy()
    rows["dataset_dir"] = str(dataset_dir)
    rows["dataset_name"] = dataset_dir.name
    rows["patch_file"] = rows["patch_path"].map(
        lambda value: str((dataset_dir / Path(str(value))).resolve())
    )
    rows["group"] = rows["source_image"].astype(str)
    missing = [path for path in rows["patch_file"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan {len(missing)} patches. Ejemplo: {missing[0]}")
    return rows


def split_score(
    rows: pd.DataFrame,
    assignments: np.ndarray,
    classes: list[str],
    target_fraction: dict[str, float],
    min_class_holdout: int,
) -> float:
    total_distribution = rows["label"].value_counts(normalize=True)
    score = 0.0
    for split in ("train", "val", "test"):
        subset = rows[assignments == split]
        if set(subset["label"]) != set(classes):
            return float("inf")
        if split in {"val", "test"}:
            counts = subset["label"].value_counts()
            if any(counts.get(label, 0) < min_class_holdout for label in classes):
                return float("inf")
        score += abs(len(subset) / len(rows) - target_fraction[split]) * 3.0
        distribution = subset["label"].value_counts(normalize=True)
        score += sum(abs(distribution.get(label, 0.0) - total_distribution[label]) for label in classes)
    return score


def create_group_split(
    rows: pd.DataFrame,
    classes: list[str],
    seed: int,
    val_size: float,
    test_size: float,
    min_class_holdout: int,
) -> np.ndarray:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("--val-size y --test-size deben ser positivos y sumar menos de 1.")

    best_assignments = None
    best_score = float("inf")
    holdout_size = val_size + test_size
    target_fraction = {
        "train": 1.0 - holdout_size,
        "val": val_size,
        "test": test_size,
    }
    for offset in range(1000):
        outer = GroupShuffleSplit(
            n_splits=1, test_size=holdout_size, random_state=seed + offset
        )
        train_indices, holdout_indices = next(
            outer.split(rows, rows["label"], groups=rows["group"])
        )
        holdout = rows.iloc[holdout_indices]
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size / holdout_size,
            random_state=seed + 10_000 + offset,
        )
        val_relative, test_relative = next(
            inner.split(holdout, holdout["label"], groups=holdout["group"])
        )
        assignments = np.full(len(rows), "train", dtype=object)
        assignments[holdout_indices[val_relative]] = "val"
        assignments[holdout_indices[test_relative]] = "test"
        score = split_score(
            rows,
            assignments,
            classes,
            target_fraction,
            min_class_holdout,
        )
        if score < best_score:
            best_assignments = assignments.copy()
            best_score = score

    if best_assignments is None:
        raise RuntimeError(
            "No fue posible crear train/val/test con todas las clases. "
            "Se necesitan mas frames independientes por clase."
        )
    return best_assignments


def reuse_group_split(rows: pd.DataFrame, split_file: Path) -> np.ndarray:
    previous = pd.read_csv(split_file.expanduser().resolve(), encoding="utf-8-sig")
    if not {"group", "split"}.issubset(previous.columns):
        raise ValueError("--reuse-split debe contener columnas group y split.")
    mapping = previous.drop_duplicates("group").set_index("group")["split"].to_dict()
    return rows["group"].map(mapping).fillna("train").to_numpy(dtype=object)


def extract_feature_table(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_rows = []
    feature_names: list[str] | None = None
    total = len(rows)
    for index, row in enumerate(rows.itertuples(index=False), start=1):
        extracted = extract_features(Path(row.patch_file))
        if feature_names is None:
            feature_names = list(extracted)
        feature_rows.append(extracted)
        if index == 1 or index % 25 == 0 or index == total:
            print(f"Features: {index}/{total}")
    return pd.DataFrame(feature_rows), feature_names or []


def model_candidates(seed: int, jobs: int) -> list[tuple[str, RandomForestClassifier]]:
    common = dict(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=jobs,
    )
    return [
        ("rf_leaf1_sqrt", RandomForestClassifier(min_samples_leaf=1, max_features="sqrt", **common)),
        ("rf_leaf2_sqrt", RandomForestClassifier(min_samples_leaf=2, max_features="sqrt", **common)),
        ("rf_leaf3_sqrt", RandomForestClassifier(min_samples_leaf=3, max_features="sqrt", **common)),
        ("rf_depth12_leaf2", RandomForestClassifier(max_depth=12, min_samples_leaf=2, max_features=0.5, **common)),
    ]


def main() -> None:
    args = parse_args()
    classes = list(dict.fromkeys(args.classes))
    frames = [load_dataset(path, classes, args.confidence) for path in args.dataset]
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.drop_duplicates(["dataset_dir", "patch_id"], keep="last").reset_index(drop=True)
    if rows.empty:
        raise ValueError("No quedaron patches con las clases y confianza seleccionadas.")

    if args.reuse_split:
        assignments = reuse_group_split(rows, args.reuse_split)
    else:
        assignments = create_group_split(
            rows,
            classes,
            args.seed,
            args.val_size,
            args.test_size,
            args.min_class_holdout,
        )
    rows["split"] = assignments

    for split in ("train", "val", "test"):
        if set(rows.loc[rows["split"] == split, "label"]) != set(classes):
            raise ValueError(f"El split {split} no contiene todas las clases: revisa la division.")

    run_dir = next_run_dir(args.output_dir.expanduser().resolve(), args.run_name)
    print(f"Muestras: {len(rows)}")
    print(f"Frames: {rows['group'].nunique()}")
    print(f"Salida: {run_dir}")

    feature_table, feature_names = extract_feature_table(rows)
    x = feature_table.to_numpy(dtype=np.float32)
    y = rows["label"].to_numpy()
    train_mask = rows["split"].eq("train").to_numpy()
    val_mask = rows["split"].eq("val").to_numpy()
    test_mask = rows["split"].eq("test").to_numpy()

    candidate_metrics = []
    best_name = None
    best_model = None
    best_score = -1.0
    best_accuracy = -1.0
    for name, model in model_candidates(args.seed, args.jobs):
        model.fit(x[train_mask], y[train_mask])
        prediction = model.predict(x[val_mask])
        score = f1_score(y[val_mask], prediction, labels=classes, average="macro", zero_division=0)
        accuracy = accuracy_score(y[val_mask], prediction)
        candidate_metrics.append({"name": name, "val_f1_macro": score, "val_accuracy": accuracy})
        print(f"{name}: val_f1_macro={score:.3f} val_accuracy={accuracy:.3f}")
        if (score, accuracy) > (best_score, best_accuracy):
            best_name = name
            best_model = model
            best_score = score
            best_accuracy = accuracy

    assert best_model is not None and best_name is not None
    validation_prediction = best_model.predict(x[val_mask])
    validation_report = classification_report(
        y[val_mask], validation_prediction, labels=classes, output_dict=True, zero_division=0
    )

    final_training_mask = train_mask | val_mask
    best_model.fit(x[final_training_mask], y[final_training_mask])
    test_prediction = best_model.predict(x[test_mask])
    test_probabilities = best_model.predict_proba(x[test_mask])
    test_report = classification_report(
        y[test_mask], test_prediction, labels=classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y[test_mask], test_prediction, labels=classes)

    bundle = {
        "model": best_model,
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "classes": classes,
        "selected_candidate": best_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, run_dir / "model.joblib")

    metadata_columns = [
        "patch_id", "dataset_name", "patch_file", "source_image", "source_relpath",
        "group", "label", "confidence", "split", "x", "y", "width", "height",
    ]
    combined_features = pd.concat(
        [rows[[column for column in metadata_columns if column in rows.columns]], feature_table], axis=1
    )
    combined_features.to_csv(run_dir / "features.csv", index=False, encoding="utf-8-sig")
    rows.to_csv(run_dir / "split.csv", index=False, encoding="utf-8-sig")
    rows[["group", "split"]].drop_duplicates().sort_values("group").to_csv(
        run_dir / "group_split.csv", index=False, encoding="utf-8-sig"
    )

    test_rows = rows.loc[test_mask, ["patch_id", "dataset_name", "patch_file", "source_image", "label"]].copy()
    test_rows["prediction"] = test_prediction
    class_to_column = {label: index for index, label in enumerate(best_model.classes_)}
    for label in classes:
        test_rows[f"prob_{label}"] = test_probabilities[:, class_to_column[label]]
    test_rows.to_csv(run_dir / "predictions_test.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        run_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(cmap="Blues", colorbar=False)
    plt.title("Random Forest - conjunto test")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    pd.DataFrame(
        {"feature": feature_names, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        run_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    split_summary = {
        split: {
            "patches": int((rows["split"] == split).sum()),
            "frames": int(rows.loc[rows["split"] == split, "group"].nunique()),
            "classes": dict(Counter(rows.loc[rows["split"] == split, "label"])),
        }
        for split in ("train", "val", "test")
    }
    save_json(
        run_dir / "metrics.json",
        {
            "selected_candidate": best_name,
            "candidate_validation": candidate_metrics,
            "validation_report_before_refit": validation_report,
            "test_report": test_report,
            "test_f1_macro": f1_score(
                y[test_mask], test_prediction, labels=classes, average="macro", zero_division=0
            ),
            "test_accuracy": accuracy_score(y[test_mask], test_prediction),
        },
    )
    save_json(
        run_dir / "dataset_summary.json",
        {
            "datasets": [str(path.expanduser().resolve()) for path in args.dataset],
            "classes": classes,
            "confidence": args.confidence,
            "feature_version": FEATURE_VERSION,
            "total_patches": len(rows),
            "total_frames": rows["group"].nunique(),
            "class_distribution": dict(Counter(rows["label"])),
            "splits": split_summary,
            "seed": args.seed,
            "min_class_holdout": args.min_class_holdout,
            "reuse_split": str(args.reuse_split) if args.reuse_split else None,
        },
    )

    print(f"Modelo seleccionado: {best_name}")
    print(f"Test F1 macro: {f1_score(y[test_mask], test_prediction, labels=classes, average='macro', zero_division=0):.3f}")
    print(f"Test accuracy: {accuracy_score(y[test_mask], test_prediction):.3f}")
    print(f"Modelo guardado: {run_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
