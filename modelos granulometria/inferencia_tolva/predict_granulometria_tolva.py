from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from granulometria.paths import (  # noqa: E402
    CLASSIFIER_DIR,
    FINAL_CNN_MODEL,
    OUTPUTS_DIR,
    PERCEPTION_MODEL,
)

if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from features import FEATURE_VERSION, extract_features_from_image  # noqa: E402
from granulometria.metrics import (  # noqa: E402
    compute_granulometry_metrics,
    metric_definitions,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PATCH_MODEL = FINAL_CNN_MODEL
DEFAULT_YOLO_MODEL = PERCEPTION_MODEL
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "granulometria_tolva"
LABEL_COLORS_BGR = {
    "fino": (80, 210, 90),
    "medio": (0, 210, 255),
    "grueso": (55, 80, 240),
    "mixto": (210, 120, 255),
    "invalido": (135, 135, 135),
    "ambiguo": (255, 180, 60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predice granulometria global en frames completos de tolva: "
            "YOLO segmenta la carga, se generan patches dentro de la mascara, "
            "se clasifican con el modelo de patches y se agregan porcentajes por frame."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="Imagen o carpeta de frames.")
    parser.add_argument(
        "--patch-model",
        type=Path,
        default=DEFAULT_PATCH_MODEL,
        help="Modelo del clasificador de patches: .joblib para RF o .pt para CNN ResNet.",
    )
    parser.add_argument(
        "--patch-model-type",
        choices=("auto", "rf", "cnn"),
        default="auto",
        help="Tipo de clasificador de patches. auto infiere por extension del archivo.",
    )
    parser.add_argument(
        "--patch-device",
        default=None,
        help="Dispositivo para CNN: cpu, cuda, cuda:0. Si se omite, usa CUDA si esta disponible.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=DEFAULT_YOLO_MODEL,
        help="Modelo YOLO-seg para segmentar carga visible en tolva.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta base de salidas.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre del experimento. Si se omite, usa el nombre del source.",
    )
    parser.add_argument("--recursive", action="store_true", help="Buscar imagenes recursivamente.")
    parser.add_argument("--max-images", type=int, default=None, help="Limite de frames para pruebas.")
    parser.add_argument("--imgsz", type=int, default=736, help="Tamano de entrada YOLO.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianza minima YOLO.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU para NMS de YOLO.")
    parser.add_argument(
        "--device",
        default=None,
        help="Dispositivo YOLO: cpu, 0, 0,1, etc. Si se omite, Ultralytics decide.",
    )
    parser.add_argument(
        "--class-ids",
        default="1",
        help="IDs de clase YOLO separados por coma. El modelo de carga usa 1 para material.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("union", "largest"),
        default="union",
        help="Como combinar multiples mascaras detectadas en un frame.",
    )
    parser.add_argument(
        "--boxes-as-masks",
        action="store_true",
        help="Si YOLO no entrega segmentacion, usar cajas como mascaras rectangulares.",
    )
    parser.add_argument("--patch-size", type=int, default=96, help="Tamano del patch en pixeles.")
    parser.add_argument("--stride", type=int, default=96, help="Paso entre patches.")
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.8,
        help="Fraccion minima del patch cubierta por la mascara de carga.",
    )
    parser.add_argument(
        "--min-mask-area-ratio",
        type=float,
        default=0.0,
        help="Descarta mascaras con area menor a esta fraccion del frame.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.38,
        help="Transparencia del color de patches en el overlay.",
    )
    parser.add_argument(
        "--no-save-patches",
        action="store_true",
        help="No guardar crops de patches clasificados.",
    )
    parser.add_argument(
        "--normalize-width",
        type=int,
        default=896,
        help=(
            "Si es mayor que 0, recorta la mascara de carga y la redimensiona a este ancho "
            "antes de generar patches. Usar 896 para modelos entrenados con grilla normalizada."
        ),
    )
    parser.add_argument(
        "--normalized-background",
        choices=("black", "dim", "original"),
        default="black",
        help="Fondo fuera de la mascara cuando se usa --normalize-width.",
    )
    return parser.parse_args()


def parse_class_ids(raw_value: str | None) -> list[int] | None:
    if raw_value is None or raw_value.strip() == "":
        return None
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def sanitize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "source"


def default_run_name(source: Path) -> str:
    return sanitize_name(source.stem if source.is_file() else source.name)


def unique_run_dir(output_dir: Path, run_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / sanitize_name(run_name)
    if not candidate.exists():
        return candidate

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{sanitize_name(run_name)}_{timestamp}"
    if not candidate.exists():
        return candidate

    suffix = 2
    while (output_dir / f"{candidate.name}_{suffix}").exists():
        suffix += 1
    return output_dir / f"{candidate.name}_{suffix}"


def list_images(source: Path, recursive: bool, max_images: int | None) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Archivo no soportado como imagen: {source}")
        return [source.resolve()]

    if not source.is_dir():
        raise FileNotFoundError(f"No existe source: {source}")

    iterator: Iterable[Path] = source.rglob("*") if recursive else source.glob("*")
    images = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:max_images] if max_images is not None else images


def output_stem(image_path: Path, source: Path) -> Path:
    image_path = image_path.resolve()
    source = source.resolve()
    if source.is_dir():
        return image_path.relative_to(source).with_suffix("")
    return Path(image_path.stem)


def polygons_to_masks(result, height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if result.masks is None:
        return masks

    for polygon in result.masks.xy:
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
        masks.append(mask)
    return masks


def boxes_to_masks(result, height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if result.boxes is None or result.boxes.xyxy is None:
        return masks

    for x1, y1, x2, y2 in result.boxes.xyxy.cpu().numpy():
        left = int(np.clip(round(x1), 0, width - 1))
        top = int(np.clip(round(y1), 0, height - 1))
        right = int(np.clip(round(x2), 0, width - 1))
        bottom = int(np.clip(round(y2), 0, height - 1))
        if right <= left or bottom <= top:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (left, top), (right, bottom), 255, thickness=-1)
        masks.append(mask)
    return masks


def combine_masks(masks: list[np.ndarray], mode: str) -> np.ndarray:
    if not masks:
        raise ValueError("No hay mascaras para combinar.")

    if mode == "largest":
        return max(masks, key=lambda item: int(np.count_nonzero(item)))

    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined = cv2.bitwise_or(combined, mask)
    return combined


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = cv2.findNonZero((mask > 0).astype(np.uint8))
    if coords is None:
        return None
    x, y, width, height = cv2.boundingRect(coords)
    return x, y, width, height


def crop_and_normalize(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_width: int,
    background: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    x, y, width, height = bbox
    crop = image[y : y + height, x : x + width]
    crop_mask = mask[y : y + height, x : x + width]
    scale = target_width / float(width)
    target_height = max(1, int(round(height * scale)))
    resized = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(
        crop_mask,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )

    if background == "black":
        normalized = np.zeros_like(resized)
        normalized[resized_mask > 0] = resized[resized_mask > 0]
    elif background == "dim":
        normalized = (resized * 0.28).astype(np.uint8)
        normalized[resized_mask > 0] = resized[resized_mask > 0]
    else:
        normalized = resized

    return normalized, resized_mask, scale


def confidence_summary(result) -> tuple[int, float | None, float | None]:
    if result.boxes is None or result.boxes.conf is None:
        return 0, None, None
    confidences = result.boxes.conf.cpu().numpy()
    if len(confidences) == 0:
        return 0, None, None
    return int(len(confidences)), float(confidences.max()), float(confidences.mean())


def sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def patch_id_for(stem: Path, x: int, y: int) -> str:
    prefix = "__".join(stem.parts)
    return f"{prefix}__x{x:05d}_y{y:05d}"


def label_color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS_BGR.get(label, (220, 220, 220))


def make_mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[:, :] = (0, 180, 255)
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(image, 0.55, color_layer, 0.45, 0)[mask_bool]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def draw_patch_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    patch_rows: list[dict],
    alpha: float,
) -> np.ndarray:
    overlay = image.copy()
    fill = image.copy()

    for row in patch_rows:
        x = int(row["x"])
        y = int(row["y"])
        width = int(row["width"])
        height = int(row["height"])
        color = label_color(str(row["prediction"]))
        cv2.rectangle(fill, (x, y), (x + width, y + height), color, thickness=-1)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, thickness=2)

    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        overlay,
        1.0 - alpha,
        fill,
        alpha,
        0,
    )[mask_bool]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    add_legend(overlay)
    add_summary_text(overlay, patch_rows)
    return overlay


def add_legend(image: np.ndarray) -> None:
    x0, y0 = 14, 18
    for index, label in enumerate(["fino", "medio", "grueso"]):
        y = y0 + index * 24
        color = label_color(label)
        cv2.rectangle(image, (x0, y), (x0 + 18, y + 14), color, thickness=-1)
        cv2.putText(
            image,
            label,
            (x0 + 26, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def add_summary_text(image: np.ndarray, patch_rows: list[dict]) -> None:
    if not patch_rows:
        return
    counts = Counter(str(row["prediction"]) for row in patch_rows)
    total = max(len(patch_rows), 1)
    shares = {
        label: counts.get(label, 0) / total
        for label in ("fino", "medio", "grueso")
    }
    metrics = compute_granulometry_metrics(shares)
    share_text = "  ".join(
        f"{label}:{shares[label]:.0%}" for label in ("fino", "medio", "grueso")
    )
    metrics_text = (
        f"IG:{metrics['indice_granulometrico']:.2f}  "
        f"dispersion:{metrics['dispersion_granulometrica']:.2f}"
    )
    first_y = max(105, image.shape[0] - 45)
    second_y = max(128, image.shape[0] - 18)
    max_width = max(
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
        for text in (share_text, metrics_text)
    )
    cv2.rectangle(
        image,
        (8, first_y - 22),
        (min(image.shape[1] - 1, max_width + 22), second_y + 7),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        image,
        share_text,
        (14, first_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        metrics_text,
        (14, second_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_fieldnames: list[str] = list(fieldnames or [])
    for row in rows:
        for key in row.keys():
            if key not in ordered_fieldnames:
                ordered_fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_patch_model_type(model_path: Path, requested_type: str) -> str:
    if requested_type != "auto":
        return requested_type
    if model_path.suffix.lower() == ".pt":
        return "cnn"
    return "rf"


def choose_torch_device(raw_device: str | None):
    import torch

    if raw_device:
        if raw_device.isdigit():
            return torch.device(f"cuda:{raw_device}")
        return torch.device(raw_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_patch_model(model_path: Path, model_type: str, patch_device: str | None) -> dict:
    model_type = resolve_patch_model_type(model_path, model_type)
    if model_type == "cnn":
        import torch
        from torch import nn
        from torchvision import models

        device = choose_torch_device(patch_device)
        checkpoint = torch.load(model_path, map_location=device)
        classes = [str(label) for label in checkpoint["classes"]]
        architecture = checkpoint.get("architecture", "resnet18")
        if architecture != "resnet18":
            raise ValueError(f"Arquitectura CNN no soportada: {architecture}")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return {
            "kind": "cnn",
            "model": model,
            "classes": classes,
            "image_size": int(checkpoint.get("image_size", 96)),
            "mean": np.asarray(checkpoint.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32),
            "std": np.asarray(checkpoint.get("std", [0.229, 0.224, 0.225]), dtype=np.float32),
            "device": device,
            "architecture": architecture,
        }

    bundle = joblib.load(model_path)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"Version de features incompatible: {bundle.get('feature_version')} != {FEATURE_VERSION}"
        )
    bundle["kind"] = "rf"
    bundle["classes"] = [str(label) for label in bundle["model"].classes_]
    return bundle


def predict_patch_batch(bundle: dict, patches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if bundle.get("kind") == "cnn":
        import torch

        image_size = int(bundle["image_size"])
        mean = bundle["mean"].reshape(1, 1, 3)
        std = bundle["std"].reshape(1, 1, 3)
        tensors = []
        for patch in patches:
            resized = cv2.resize(patch, (image_size, image_size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            normalized = (rgb - mean) / std
            tensors.append(np.transpose(normalized, (2, 0, 1)))

        device = bundle["device"]
        model = bundle["model"]
        batch_size = 64
        probabilities_list = []
        with torch.inference_mode():
            for start in range(0, len(tensors), batch_size):
                batch = torch.from_numpy(np.stack(tensors[start : start + batch_size])).to(
                    device=device,
                    dtype=torch.float32,
                )
                logits = model(batch)
                probabilities_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        probabilities = np.concatenate(probabilities_list, axis=0)
        classes = np.asarray(bundle["classes"], dtype=object)
        predictions = classes[np.argmax(probabilities, axis=1)]
        return predictions, probabilities

    feature_rows = [extract_features_from_image(patch) for patch in patches]
    features = pd.DataFrame(feature_rows)[bundle["feature_names"]]
    values = features.to_numpy(dtype=np.float32)
    classifier = bundle["model"]
    predictions = classifier.predict(values)
    probabilities = classifier.predict_proba(values)
    return predictions, probabilities


def summary_for_image(
    image_path: Path,
    width: int,
    height: int,
    detections: int,
    max_yolo_conf: float | None,
    mean_yolo_conf: float | None,
    mask_area_px: int,
    candidate_count: int,
    rejected_patches: int,
    patch_rows: list[dict],
    model_classes: list[str],
) -> dict:
    total_patches = len(patch_rows)
    summary: dict[str, object] = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "width": width,
        "height": height,
        "detections": detections,
        "mask_candidates": candidate_count,
        "has_mask": mask_area_px > 0,
        "mask_area_px": mask_area_px,
        "mask_area_ratio": round(mask_area_px / float(width * height), 6),
        "patch_count": total_patches,
        "rejected_patches": rejected_patches,
        "max_yolo_conf": "" if max_yolo_conf is None else round(max_yolo_conf, 6),
        "mean_yolo_conf": "" if mean_yolo_conf is None else round(mean_yolo_conf, 6),
    }

    if total_patches == 0:
        for label in model_classes:
            summary[f"share_{label}"] = 0.0
            summary[f"prob_{label}_mean"] = 0.0
        summary["dominant_label"] = ""
        summary["dominant_share"] = 0.0
        summary["mean_patch_confidence"] = 0.0
        summary["dominant_margin"] = 0.0
        summary["indice_granulometrico"] = ""
        summary["dispersion_granulometrica"] = ""
        return summary

    counts = Counter(str(row["prediction"]) for row in patch_rows)
    for label in model_classes:
        summary[f"share_{label}"] = round(counts.get(label, 0) / total_patches, 6)
        prob_values = [float(row.get(f"prob_{label}", 0.0)) for row in patch_rows]
        summary[f"prob_{label}_mean"] = round(float(np.mean(prob_values)), 6)

    ordered_counts = counts.most_common()
    dominant_label, dominant_count = ordered_counts[0]
    second_share = ordered_counts[1][1] / total_patches if len(ordered_counts) > 1 else 0.0
    dominant_share = dominant_count / total_patches
    summary["dominant_label"] = dominant_label
    summary["dominant_share"] = round(dominant_share, 6)
    summary["dominant_margin"] = round(dominant_share - second_share, 6)
    summary["mean_patch_confidence"] = round(
        float(np.mean([float(row["confidence"]) for row in patch_rows])),
        6,
    )
    if {"fino", "medio", "grueso"}.issubset(set(model_classes)):
        metrics = compute_granulometry_metrics(
            {
                "fino": float(summary["share_fino"]),
                "medio": float(summary["share_medio"]),
                "grueso": float(summary["share_grueso"]),
            }
        )
        summary["indice_granulometrico"] = round(
            float(metrics["indice_granulometrico"]),
            6,
        )
        summary["dispersion_granulometrica"] = round(
            float(metrics["dispersion_granulometrica"]),
            6,
        )
    else:
        summary["indice_granulometrico"] = ""
        summary["dispersion_granulometrica"] = ""
    return summary


def ensure_paths(args: argparse.Namespace) -> None:
    args.source = args.source.expanduser().resolve()
    args.patch_model = args.patch_model.expanduser().resolve()
    args.yolo_model = args.yolo_model.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.patch_model.is_file():
        raise FileNotFoundError(f"No existe patch-model: {args.patch_model}")
    if not args.yolo_model.is_file():
        raise FileNotFoundError(f"No existe yolo-model: {args.yolo_model}")
    if args.patch_size <= 0:
        raise ValueError("--patch-size debe ser mayor que 0")
    if args.stride <= 0:
        raise ValueError("--stride debe ser mayor que 0")
    if args.normalize_width < 0:
        raise ValueError("--normalize-width debe ser 0 o mayor")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise ValueError("--min-valid-fraction debe estar entre 0 y 1")


def main() -> None:
    args = parse_args()
    ensure_paths(args)
    class_ids = parse_class_ids(args.class_ids)
    run_name = args.run_name or default_run_name(args.source)
    run_dir = unique_run_dir(args.output_dir, run_name)
    masks_dir = run_dir / "masks"
    yolo_masks_dir = run_dir / "masks_yolo_original"
    normalized_dir = run_dir / "normalized_inputs"
    yolo_overlays_dir = run_dir / "overlays_yolo"
    granulometry_overlays_dir = run_dir / "overlays_granulometria"
    patches_dir = run_dir / "patches_clasificados"
    for directory in (masks_dir, yolo_overlays_dir, granulometry_overlays_dir):
        directory.mkdir(parents=True, exist_ok=False)
    if args.normalize_width > 0:
        yolo_masks_dir.mkdir(parents=True, exist_ok=False)
        normalized_dir.mkdir(parents=True, exist_ok=False)
    if not args.no_save_patches:
        patches_dir.mkdir(parents=True, exist_ok=False)

    image_paths = list_images(args.source, args.recursive, args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imagenes en: {args.source}")

    bundle = load_patch_model(args.patch_model, args.patch_model_type, args.patch_device)
    model_classes = [str(label) for label in bundle["classes"]]

    save_json(
        run_dir / "params.json",
        {
            "source": str(args.source),
            "patch_model": str(args.patch_model),
            "patch_model_type": bundle["kind"],
            "patch_device": str(bundle.get("device", "")),
            "patch_architecture": bundle.get("architecture", ""),
            "yolo_model": str(args.yolo_model),
            "feature_version": FEATURE_VERSION if bundle["kind"] == "rf" else "",
            "model_classes": model_classes,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "class_ids": class_ids,
            "mask_mode": args.mask_mode,
            "boxes_as_masks": args.boxes_as_masks,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "min_valid_fraction": args.min_valid_fraction,
            "min_mask_area_ratio": args.min_mask_area_ratio,
            "recursive": args.recursive,
            "max_images": args.max_images,
            "save_patches": not args.no_save_patches,
            "normalize_width": args.normalize_width,
            "normalized_background": args.normalized_background,
            "granulometry_metrics": metric_definitions(),
        },
    )

    from ultralytics import YOLO

    yolo = YOLO(str(args.yolo_model))
    results = yolo.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=class_ids,
        device=args.device,
        stream=True,
        verbose=False,
    )

    all_patch_rows: list[dict] = []
    summary_rows: list[dict] = []

    print(f"Frames: {len(image_paths)}")
    print(f"Salida: {run_dir}")

    for index, result in enumerate(results, start=1):
        image_path = image_paths[index - 1]
        image = result.orig_img
        height, width = image.shape[:2]
        stem = output_stem(image_path, args.source)
        stem_key = "__".join(stem.parts)

        candidate_masks = polygons_to_masks(result, height, width)
        if not candidate_masks and args.boxes_as_masks:
            candidate_masks = boxes_to_masks(result, height, width)

        if candidate_masks:
            mask = combine_masks(candidate_masks, args.mask_mode)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        mask_area_px = int(np.count_nonzero(mask))
        if mask_area_px / float(width * height) < args.min_mask_area_ratio:
            mask[:] = 0
            mask_area_px = 0

        analysis_image = image
        analysis_mask = mask
        analysis_height, analysis_width = height, width
        normalization_enabled = False
        normalization_scale = 1.0
        normalization_bbox = (0, 0, width, height)
        normalized_input_path = ""
        yolo_mask_path = ""

        if args.normalize_width > 0 and mask_area_px > 0:
            bbox = mask_bbox(mask)
            if bbox is not None:
                analysis_image, analysis_mask, normalization_scale = crop_and_normalize(
                    image=image,
                    mask=mask,
                    bbox=bbox,
                    target_width=args.normalize_width,
                    background=args.normalized_background,
                )
                analysis_height, analysis_width = analysis_image.shape[:2]
                normalization_enabled = True
                normalization_bbox = bbox
                mask_area_px = int(np.count_nonzero(analysis_mask))

        mask_path = masks_dir / stem.parent / f"{stem.name}_mask.png"
        yolo_overlay_path = yolo_overlays_dir / stem.parent / f"{stem.name}_yolo_overlay.jpg"
        granulometry_overlay_path = (
            granulometry_overlays_dir / stem.parent / f"{stem.name}_granulometria_overlay.jpg"
        )
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        yolo_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        granulometry_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), analysis_mask)
        cv2.imwrite(str(yolo_overlay_path), make_mask_overlay(image, mask))
        if args.normalize_width > 0:
            yolo_mask_output = yolo_masks_dir / stem.parent / f"{stem.name}_yolo_mask.png"
            yolo_mask_output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(yolo_mask_output), mask)
            yolo_mask_path = str(yolo_mask_output)
        if normalization_enabled:
            normalized_output = normalized_dir / stem.parent / f"{stem.name}_normalized.jpg"
            normalized_output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(normalized_output), analysis_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            normalized_input_path = str(normalized_output)

        patches: list[np.ndarray] = []
        patch_records: list[dict] = []
        rejected = 0
        valid_mask = analysis_mask > 0
        bbox_x, bbox_y, _, _ = normalization_bbox
        for y in sliding_positions(analysis_height, args.patch_size, args.stride):
            for x in sliding_positions(analysis_width, args.patch_size, args.stride):
                fraction = float(
                    valid_mask[y : y + args.patch_size, x : x + args.patch_size].mean()
                )
                if fraction < args.min_valid_fraction:
                    rejected += 1
                    continue
                patch = analysis_image[y : y + args.patch_size, x : x + args.patch_size]
                patch_id = patch_id_for(stem, x, y)
                x_original = bbox_x + x / normalization_scale
                y_original = bbox_y + y / normalization_scale
                size_original = args.patch_size / normalization_scale
                patches.append(patch)
                patch_records.append(
                    {
                        "patch_id": patch_id,
                        "image_path": str(image_path),
                        "source_relpath": str(stem.with_suffix(image_path.suffix)),
                        "x": x,
                        "y": y,
                        "width": args.patch_size,
                        "height": args.patch_size,
                        "x_original": round(float(x_original), 3),
                        "y_original": round(float(y_original), 3),
                        "width_original": round(float(size_original), 3),
                        "height_original": round(float(size_original), 3),
                        "normalized": normalization_enabled,
                        "normalized_width": analysis_width,
                        "normalized_height": analysis_height,
                        "normalized_input_path": normalized_input_path,
                        "valid_fraction": round(fraction, 6),
                        "mask_path": str(mask_path),
                        "yolo_mask_path": yolo_mask_path,
                    }
                )

        patch_rows: list[dict] = []
        if patches:
            predictions, probabilities = predict_patch_batch(bundle, patches)
            for patch, record, prediction, probability_row in zip(
                patches,
                patch_records,
                predictions,
                probabilities,
                strict=True,
            ):
                label = str(prediction)
                confidence = float(np.max(probability_row))
                saved_patch_path = ""
                if not args.no_save_patches:
                    patch_output = patches_dir / label / f"{record['patch_id']}.jpg"
                    patch_output.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(patch_output), patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_patch_path = str(patch_output)

                row = {
                    **record,
                    "patch_path": saved_patch_path,
                    "prediction": label,
                    "confidence": round(confidence, 6),
                }
                for class_index, class_label in enumerate(model_classes):
                    row[f"prob_{class_label}"] = round(float(probability_row[class_index]), 6)
                patch_rows.append(row)

        cv2.imwrite(
            str(granulometry_overlay_path),
            draw_patch_overlay(analysis_image, analysis_mask, patch_rows, args.overlay_alpha),
        )

        detections, max_yolo_conf, mean_yolo_conf = confidence_summary(result)
        summary = summary_for_image(
            image_path=image_path,
            width=analysis_width,
            height=analysis_height,
            detections=detections,
            max_yolo_conf=max_yolo_conf,
            mean_yolo_conf=mean_yolo_conf,
            mask_area_px=mask_area_px,
            candidate_count=len(candidate_masks),
            rejected_patches=rejected,
            patch_rows=patch_rows,
            model_classes=model_classes,
        )
        summary["original_width"] = width
        summary["original_height"] = height
        summary["normalized"] = normalization_enabled
        summary["normalized_width"] = analysis_width if normalization_enabled else ""
        summary["normalized_height"] = analysis_height if normalization_enabled else ""
        summary["normalize_width"] = args.normalize_width
        summary["normalization_scale"] = round(float(normalization_scale), 6)
        summary["normalized_background"] = args.normalized_background
        summary["normalized_input_path"] = normalized_input_path
        summary["yolo_mask_path"] = yolo_mask_path
        summary["mask_path"] = str(mask_path)
        summary["yolo_overlay_path"] = str(yolo_overlay_path)
        summary["granulometry_overlay_path"] = str(granulometry_overlay_path)
        summary_rows.append(summary)
        all_patch_rows.extend(patch_rows)

        print(
            f"[{index}/{len(image_paths)}] {image_path.name} "
            f"patches={len(patch_rows)} pred={summary['dominant_label']} "
            f"IG={summary['indice_granulometrico']} "
            f"dispersion={summary['dispersion_granulometrica']}"
        )

    detail_fields = [
        "patch_id",
        "image_path",
        "source_relpath",
        "x",
        "y",
        "width",
        "height",
        "x_original",
        "y_original",
        "width_original",
        "height_original",
        "normalized",
        "normalized_width",
        "normalized_height",
        "normalized_input_path",
        "valid_fraction",
        "mask_path",
        "yolo_mask_path",
        "patch_path",
        "prediction",
        "confidence",
        *[f"prob_{label}" for label in model_classes],
    ]
    write_csv(run_dir / "details_por_patch.csv", all_patch_rows, fieldnames=detail_fields)
    write_csv(run_dir / "summary.csv", summary_rows)

    print(f"Resumen: {run_dir / 'summary.csv'}")
    print(f"Detalle patches: {run_dir / 'details_por_patch.csv'}")
    print(f"Overlays: {granulometry_overlays_dir}")


if __name__ == "__main__":
    main()
