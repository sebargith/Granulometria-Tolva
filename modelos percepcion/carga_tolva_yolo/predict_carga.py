from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el modelo YOLO-seg de percepcion de carga visible en tolva "
            "y guarda mascaras binarias, overlays y un CSV de control."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=SCRIPT_DIR / "best.pt",
        help="Ruta al modelo .pt de Ultralytics.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "input",
        help="Imagen o carpeta de imagenes a procesar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/yolo_carga",
        help="Carpeta base para guardar resultados.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Nombre del experimento. Si se omite, usa el nombre de la carpeta/imagen "
            "de entrada. Si ya existe, agrega fecha y hora."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de entrada YOLO.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianza minima.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU para NMS.")
    parser.add_argument(
        "--device",
        default=None,
        help="Dispositivo para inferencia: cpu, 0, 0,1, etc. Si se omite, Ultralytics decide.",
    )
    parser.add_argument(
        "--class-ids",
        default="0",
        help="IDs de clases separados por coma. Este modelo usa 0 para Carga.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("union", "largest"),
        default="union",
        help="Como combinar multiples mascaras detectadas por frame.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Buscar imagenes recursivamente si source es una carpeta.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limita la cantidad de imagenes para pruebas rapidas.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0,
        help="Descarta la mascara final si cubre menos que esta fraccion de la imagen.",
    )
    parser.add_argument(
        "--boxes-as-masks",
        action="store_true",
        help="Si el modelo no entrega segmentacion, usa cajas como mascaras rectangulares.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Transparencia de la mascara en el overlay.",
    )
    parser.add_argument(
        "--save-model-preview",
        action="store_true",
        help="Guarda tambien la visualizacion nativa de Ultralytics.",
    )
    return parser.parse_args()


def parse_class_ids(raw_value: str | None) -> list[int] | None:
    if raw_value is None or raw_value.strip() == "":
        return None
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def list_images(source: Path, recursive: bool, max_images: int | None) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"El archivo no parece ser una imagen soportada: {source}")
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(f"No existe la fuente indicada: {source}")

    iterator: Iterable[Path] = source.rglob("*") if recursive else source.glob("*")
    images = sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        images = images[:max_images]
    return images


def make_run_dir(output_dir: Path, run_name: str | None) -> Path:
    run_name = unique_run_name(output_dir, run_name)
    run_dir = output_dir / run_name
    (run_dir / "masks").mkdir(parents=True, exist_ok=False)
    (run_dir / "overlays").mkdir(parents=True, exist_ok=False)
    (run_dir / "previews").mkdir(parents=True, exist_ok=False)
    return run_dir


def sanitize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "source"


def default_run_name(source: Path) -> str:
    if source.is_file():
        return sanitize_name(source.stem)
    return sanitize_name(source.name)


def unique_run_name(output_dir: Path, run_name: str | None) -> str:
    if run_name is None:
        raise ValueError("run_name debe resolverse antes de llamar unique_run_name.")

    candidate = sanitize_name(run_name)
    if not (output_dir / candidate).exists():
        return candidate

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_with_time = f"{candidate}_{timestamp}"
    if not (output_dir / candidate_with_time).exists():
        return candidate_with_time

    suffix = 2
    while (output_dir / f"{candidate_with_time}_{suffix}").exists():
        suffix += 1
    return f"{candidate_with_time}_{suffix}"


def output_stem(image_path: Path, source: Path) -> Path:
    image_path = image_path.resolve()
    source = source.resolve()
    if source.is_dir():
        rel_path = image_path.relative_to(source)
    else:
        rel_path = Path(image_path.name)
    return rel_path.with_suffix("")


def polygons_to_masks(result, height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if result.masks is None:
        return masks

    for polygon in result.masks.xy:
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(polygon) >= 3:
            points = np.round(polygon).astype(np.int32)
            cv2.fillPoly(mask, [points], 255)
        masks.append(mask)
    return masks


def boxes_to_masks(result, height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if result.boxes is None or result.boxes.xyxy is None:
        return masks

    boxes = result.boxes.xyxy.cpu().numpy()
    for x1, y1, x2, y2 in boxes:
        mask = np.zeros((height, width), dtype=np.uint8)
        left = int(np.clip(round(x1), 0, width - 1))
        top = int(np.clip(round(y1), 0, height - 1))
        right = int(np.clip(round(x2), 0, width - 1))
        bottom = int(np.clip(round(y2), 0, height - 1))
        if right > left and bottom > top:
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


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    overlay = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[:, :] = (0, 180, 255)  # BGR: amarillo/naranjo para destacar carga_tolva.
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(image, 1.0 - alpha, color_layer, alpha, 0)[mask_bool]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def confidence_summary(result) -> tuple[int, float | None, float | None, str, str]:
    if result.boxes is None or result.boxes.conf is None:
        return 0, None, None, "", ""

    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
    names = getattr(result, "names", {}) or {}
    class_names = [str(names.get(int(cls), int(cls))) for cls in classes]
    return (
        int(len(confs)),
        float(np.max(confs)) if len(confs) else None,
        float(np.mean(confs)) if len(confs) else None,
        "|".join(str(int(cls)) for cls in classes),
        "|".join(class_names),
    )


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    args.model = args.model.expanduser().resolve()
    args.source = args.source.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.run_name is None:
        args.run_name = default_run_name(args.source)
    class_ids = parse_class_ids(args.class_ids)

    try:
        global cv2, np, YOLO
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Faltan dependencias. Instala primero con: "
            "python -m pip install -r requirements.txt"
        ) from exc

    if not args.model.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {args.model}. Copia best.pt en {SCRIPT_DIR} o usa --model."
        )

    image_paths = list_images(args.source, args.recursive, args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imagenes en: {args.source}")

    run_dir = make_run_dir(args.output_dir, args.run_name)
    masks_dir = run_dir / "masks"
    overlays_dir = run_dir / "overlays"
    previews_dir = run_dir / "previews"
    csv_path = run_dir / "predictions.csv"

    save_json(
        run_dir / "params.json",
        {
            "model": str(args.model),
            "source": str(args.source),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "class_ids": class_ids,
            "mask_mode": args.mask_mode,
            "recursive": args.recursive,
            "max_images": args.max_images,
            "min_area_ratio": args.min_area_ratio,
            "boxes_as_masks": args.boxes_as_masks,
        },
    )

    print(f"Modelo: {args.model}")
    print(f"Imagenes encontradas: {len(image_paths)}")
    print(f"Salida: {run_dir}")

    model = YOLO(str(args.model))
    predict_kwargs = {
        "source": [str(path) for path in image_paths],
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "classes": class_ids,
        "device": args.device,
        "stream": True,
        "verbose": False,
    }

    rows: list[dict] = []
    results = model.predict(**predict_kwargs)

    for index, result in enumerate(results, start=1):
        result_path = Path(result.path)
        image_path = image_paths[index - 1]
        image = result.orig_img
        height, width = image.shape[:2]
        stem = output_stem(image_path, args.source)

        mask_path = masks_dir / stem.parent / f"{stem.name}_mask.png"
        overlay_path = overlays_dir / stem.parent / f"{stem.name}_overlay.jpg"
        preview_path = previews_dir / stem.parent / f"{stem.name}_preview.jpg"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.parent.mkdir(parents=True, exist_ok=True)

        candidate_masks = polygons_to_masks(result, height, width)
        used_boxes_as_masks = False
        if not candidate_masks and args.boxes_as_masks:
            candidate_masks = boxes_to_masks(result, height, width)
            used_boxes_as_masks = bool(candidate_masks)

        if candidate_masks:
            mask = combine_masks(candidate_masks, args.mask_mode)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        mask_area_px = int(np.count_nonzero(mask))
        mask_area_ratio = mask_area_px / float(height * width)
        if mask_area_ratio < args.min_area_ratio:
            mask[:] = 0
            mask_area_px = 0
            mask_area_ratio = 0.0

        overlay = make_overlay(image, mask, args.alpha)
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(overlay_path), overlay)
        if args.save_model_preview:
            cv2.imwrite(str(preview_path), result.plot())

        detections, max_conf, mean_conf, class_ids_seen, class_names_seen = confidence_summary(result)
        rows.append(
            {
                "image_path": str(image_path),
                "result_path": str(result_path),
                "image_name": image_path.name,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "preview_path": str(preview_path) if args.save_model_preview else "",
                "width": width,
                "height": height,
                "detections": detections,
                "mask_candidates": len(candidate_masks),
                "has_mask": mask_area_px > 0,
                "used_boxes_as_masks": used_boxes_as_masks,
                "mask_area_px": mask_area_px,
                "mask_area_ratio": round(mask_area_ratio, 6),
                "max_conf": "" if max_conf is None else round(max_conf, 6),
                "mean_conf": "" if mean_conf is None else round(mean_conf, 6),
                "class_ids": class_ids_seen,
                "class_names": class_names_seen,
            }
        )

        print(
            f"[{index}/{len(image_paths)}] {image_path.name} "
            f"det={detections} mask_area={mask_area_ratio:.3f}"
        )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    valid_masks = sum(1 for row in rows if row["has_mask"])
    print(f"Frames con mascara: {valid_masks}/{len(rows)}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
