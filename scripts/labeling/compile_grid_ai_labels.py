from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DEFAULT_BATCH_DIR = (
    PROJECT_ROOT
    / "data"
    / "labeling"
    / "batches"
    / "batch003_grilla96_data3_mixto_20260702"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "patches" / "grid96_batch003_data3_mixto_ai"
VALID_LABELS = {"fino", "medio", "grueso", "invalido", "ambiguo"}
VALID_CONFIDENCE = {"alta", "media", "baja"}
REQUIRED_RESPONSE_COLUMNS = ["frame_folder", "image_name", "cell_id", "label", "confidence", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida respuestas IA de grilla normalizada y crea dataset de patches."
    )
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--output-dataset", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Identificador trazable del dataset. Por defecto usa el nombre de output-dataset.",
    )
    parser.add_argument("--annotator", default="chatgpt_grid96")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_csv_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if stripped.lower().startswith("frame:"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned) + "\n"


def read_response_csv(path: Path) -> pd.DataFrame:
    text = clean_csv_text(path)
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    frame.columns = [column.strip() for column in frame.columns]
    missing = [column for column in REQUIRED_RESPONSE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} no contiene columnas requeridas: {', '.join(missing)}")
    frame = frame[REQUIRED_RESPONSE_COLUMNS].copy()
    frame["response_file"] = path.name
    return frame


def load_expected(batch_dir: Path) -> pd.DataFrame:
    frames_dir = batch_dir / "frames_a_etiquetar"
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"No existe frames_a_etiquetar: {frames_dir}")

    rows: list[dict] = []
    for frame_dir in sorted(path for path in frames_dir.iterdir() if path.is_dir()):
        cells_path = frame_dir / "cells.csv"
        metadata_path = frame_dir / "metadata.json"
        normalized_path = frame_dir / "01_carga_normalizada.jpg"
        if not cells_path.is_file():
            raise FileNotFoundError(f"Falta cells.csv: {frame_dir}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Falta metadata.json: {frame_dir}")
        if not normalized_path.is_file():
            raise FileNotFoundError(f"Falta 01_carga_normalizada.jpg: {frame_dir}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cells = pd.read_csv(cells_path, encoding="utf-8-sig", dtype={"cell_id": str})
        for row in cells.itertuples(index=False):
            cell_id = str(row.cell_id).zfill(3)
            rows.append(
                {
                    "frame_folder": frame_dir.name,
                    "image_name": str(metadata["image_name"]),
                    "cell_id": cell_id,
                    "x": int(row.x_norm),
                    "y": int(row.y_norm),
                    "width": int(row.width_norm),
                    "height": int(row.height_norm),
                    "valid_fraction": float(row.valid_fraction),
                    "normalized_image": str(normalized_path),
                }
            )
    expected = pd.DataFrame(rows)
    expected["key"] = expected["frame_folder"].astype(str) + "|" + expected["cell_id"].astype(str)
    return expected


def load_responses(responses_dir: Path) -> pd.DataFrame:
    files = sorted(
        path for path in responses_dir.glob("labels_batch*.csv") if path.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No se encontraron labels_batch*.csv en: {responses_dir}")
    frames = [read_response_csv(path) for path in files]
    responses = pd.concat(frames, ignore_index=True)
    for column in ("frame_folder", "image_name", "cell_id", "label", "confidence", "notes"):
        responses[column] = responses[column].astype(str).str.strip()
    responses["cell_id"] = responses["cell_id"].map(lambda value: str(value).zfill(3))
    responses["label"] = responses["label"].str.lower()
    responses["confidence"] = responses["confidence"].str.lower()
    responses["key"] = responses["frame_folder"] + "|" + responses["cell_id"]
    return responses


def validation_report(expected: pd.DataFrame, responses: pd.DataFrame) -> dict:
    expected_keys = set(expected["key"])
    response_keys = set(responses["key"])
    duplicate_rows = responses[responses.duplicated("key", keep=False)]
    invalid_labels = responses[~responses["label"].isin(VALID_LABELS)]
    invalid_confidence = responses[~responses["confidence"].isin(VALID_CONFIDENCE)]
    missing_keys = sorted(expected_keys - response_keys)
    extra_keys = sorted(response_keys - expected_keys)
    return {
        "expected_cells": int(len(expected)),
        "response_rows": int(len(responses)),
        "unique_response_cells": int(len(response_keys)),
        "missing_cells": len(missing_keys),
        "extra_cells": len(extra_keys),
        "duplicate_rows": int(len(duplicate_rows)),
        "invalid_labels": int(len(invalid_labels)),
        "invalid_confidence": int(len(invalid_confidence)),
        "missing_examples": missing_keys[:20],
        "extra_examples": extra_keys[:20],
        "duplicate_examples": duplicate_rows[["frame_folder", "cell_id", "response_file"]]
        .head(20)
        .to_dict(orient="records"),
        "invalid_label_examples": invalid_labels[
            ["frame_folder", "cell_id", "label", "response_file"]
        ]
        .head(20)
        .to_dict(orient="records"),
        "invalid_confidence_examples": invalid_confidence[
            ["frame_folder", "cell_id", "confidence", "response_file"]
        ]
        .head(20)
        .to_dict(orient="records"),
    }


def reset_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Ya existe output-dataset. Usa --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for item in output_dir.iterdir():
        if item.is_dir():
            for child in sorted(item.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            item.rmdir()
        else:
            item.unlink()


def crop_patch(image_path: Path, x: int, y: int, width: int, height: int):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer imagen normalizada: {image_path}")
    return image[y : y + height, x : x + width], image.shape[1], image.shape[0]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_dir.expanduser().resolve()
    output_dir = args.output_dataset.expanduser().resolve()
    dataset_id = args.dataset_id.strip() if args.dataset_id else output_dir.name
    responses_dir = batch_dir / "respuestas_ia"
    reviewed_dir = batch_dir / "labels_revisadas"
    reviewed_dir.mkdir(parents=True, exist_ok=True)

    expected = load_expected(batch_dir)
    responses = load_responses(responses_dir)
    report = validation_report(expected, responses)
    (reviewed_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    blocking = [
        "missing_cells",
        "extra_cells",
        "duplicate_rows",
        "invalid_labels",
        "invalid_confidence",
    ]
    if any(report[name] for name in blocking):
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit("Validacion fallida. Corrige respuestas antes de crear dataset.")

    responses = responses.drop_duplicates("key", keep="last")
    merged = expected.merge(
        responses[["key", "label", "confidence", "notes", "response_file"]],
        on="key",
        how="left",
        validate="one_to_one",
    )
    merged.to_csv(reviewed_dir / "labels_compiladas.csv", index=False, encoding="utf-8-sig")

    reset_output(output_dir, args.overwrite)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    patch_rows: list[dict] = []
    label_rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    image_size_cache: dict[str, tuple[int, int]] = {}
    for row in merged.itertuples(index=False):
        patch_id = f"{row.frame_folder}__cell_{row.cell_id}"
        patch_path = images_dir / f"{patch_id}.jpg"
        normalized_image = Path(str(row.normalized_image))
        patch, image_width, image_height = crop_patch(
            normalized_image,
            int(row.x),
            int(row.y),
            int(row.width),
            int(row.height),
        )
        if patch.size == 0:
            raise ValueError(f"Patch vacio: {patch_id}")
        cv2.imwrite(str(patch_path), patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_size_cache[str(normalized_image)] = (image_width, image_height)
        patch_rows.append(
            {
                "patch_id": patch_id,
                "patch_path": str(patch_path.relative_to(output_dir)),
                "source_image": str(row.image_name),
                "source_relpath": str(row.image_name),
                "sequence": dataset_id,
                "x": int(row.x),
                "y": int(row.y),
                "width": int(row.width),
                "height": int(row.height),
                "image_width": image_width,
                "image_height": image_height,
                "valid_fraction": float(row.valid_fraction),
            }
        )
        label_rows.append(
            {
                "patch_id": patch_id,
                "annotator": args.annotator,
                "label": row.label,
                "confidence": row.confidence,
                "notes": row.notes,
                "labeled_at": now,
                "patch_path": str(patch_path.relative_to(output_dir)),
                "source_image": str(row.image_name),
                "x": int(row.x),
                "y": int(row.y),
                "width": int(row.width),
                "height": int(row.height),
            }
        )

    write_csv(
        output_dir / "patches.csv",
        patch_rows,
        [
            "patch_id",
            "patch_path",
            "source_image",
            "source_relpath",
            "sequence",
            "x",
            "y",
            "width",
            "height",
            "image_width",
            "image_height",
            "valid_fraction",
        ],
    )
    write_csv(
        output_dir / "labels.csv",
        label_rows,
        [
            "patch_id",
            "annotator",
            "label",
            "confidence",
            "notes",
            "labeled_at",
            "patch_path",
            "source_image",
            "x",
            "y",
            "width",
            "height",
        ],
    )
    (output_dir / "params.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "batch_dir": str(batch_dir),
                "responses_dir": str(responses_dir),
                "source": "grilla_normalizada_96",
                "normalized_width": 896,
                "patch_size": 96,
                "labels_total": len(label_rows),
                "created_at": now,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    distribution = merged["label"].value_counts().to_dict()
    print(f"Dataset: {output_dir}")
    print(f"Patches: {len(patch_rows)}")
    print(f"Distribucion: {distribution}")
    print(f"Validacion: {reviewed_dir / 'validation_report.json'}")


if __name__ == "__main__":
    main()
