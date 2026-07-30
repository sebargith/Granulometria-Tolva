from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_YOLO_MODEL = (
    PROJECT_ROOT
    / "modelos percepcion"
    / "carga_tolva_yolo"
    / "haul_truck_segment_m_736_low_data_aug_07012025.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "grillas_normalizadas"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera cargas normalizadas por ancho de mascara y grillas numeradas "
            "para etiquetado asistido por IA."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="Imagen o carpeta de frames.")
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=DEFAULT_YOLO_MODEL,
        help="Modelo YOLO-seg para segmentar carga.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta base de salida.",
    )
    parser.add_argument("--run-name", default=None, help="Nombre del experimento.")
    parser.add_argument(
        "--widths",
        default="896",
        help="Anchos normalizados separados por coma.",
    )
    parser.add_argument("--cell-size", type=int, default=96, help="Tamano de celda.")
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.8,
        help="Fraccion minima de celda cubierta por la mascara para numerarla.",
    )
    parser.add_argument("--recursive", action="store_true", help="Buscar imagenes recursivamente.")
    parser.add_argument("--max-images", type=int, default=None, help="Limite de imagenes.")
    parser.add_argument("--imgsz", type=int, default=736, help="Tamano de entrada YOLO.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianza minima YOLO.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU NMS YOLO.")
    parser.add_argument("--device", default=None, help="Dispositivo YOLO: cpu, 0, etc.")
    parser.add_argument(
        "--class-ids",
        default="1",
        help="IDs de clase YOLO separados por coma. El modelo de carga usa 1 para material.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("union", "largest"),
        default="union",
        help="Como combinar multiples mascaras.",
    )
    parser.add_argument(
        "--boxes-as-masks",
        action="store_true",
        help="Usar cajas como mascaras si YOLO no devuelve segmentacion.",
    )
    parser.add_argument(
        "--background",
        choices=("black", "dim", "original"),
        default="black",
        help="Fondo fuera de la mascara en el crop normalizado.",
    )
    return parser.parse_args()


def parse_int_list(raw_value: str) -> list[int]:
    values = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("Debe indicarse al menos un ancho normalizado.")
    return values


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


def valid_cells(mask: np.ndarray, cell_size: int, min_valid_fraction: float) -> list[dict]:
    height, width = mask.shape[:2]
    cells: list[dict] = []
    cell_number = 1
    for y in range(0, height - cell_size + 1, cell_size):
        for x in range(0, width - cell_size + 1, cell_size):
            fraction = float((mask[y : y + cell_size, x : x + cell_size] > 0).mean())
            if fraction < min_valid_fraction:
                continue
            cells.append(
                {
                    "cell_id": f"{cell_number:03d}",
                    "x": x,
                    "y": y,
                    "width": cell_size,
                    "height": cell_size,
                    "valid_fraction": round(fraction, 6),
                }
            )
            cell_number += 1
    return cells


def draw_grid(image: np.ndarray, mask: np.ndarray, cells: list[dict], cell_size: int) -> np.ndarray:
    output = image.copy()
    contour_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 255, 255), 2)

    for cell in cells:
        x = int(cell["x"])
        y = int(cell["y"])
        color = (0, 230, 255)
        cv2.rectangle(output, (x, y), (x + cell_size, y + cell_size), color, 2)
        label = str(cell["cell_id"])
        text_scale = 0.46 if cell_size <= 64 else 0.56
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            thickness,
        )
        text_x = x + (cell_size - text_width) // 2
        text_y = y + (cell_size + text_height) // 2
        cv2.putText(
            output,
            label,
            (text_x + 1, text_y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    args.source = args.source.expanduser().resolve()
    args.yolo_model = args.yolo_model.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    widths = parse_int_list(args.widths)
    class_ids = parse_class_ids(args.class_ids)
    if not args.yolo_model.is_file():
        raise FileNotFoundError(f"No existe yolo-model: {args.yolo_model}")
    if args.cell_size <= 0:
        raise ValueError("--cell-size debe ser mayor que 0")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise ValueError("--min-valid-fraction debe estar entre 0 y 1")

    images = list_images(args.source, args.recursive, args.max_images)
    if not images:
        raise FileNotFoundError(f"No se encontraron imagenes en: {args.source}")

    run_name = args.run_name or default_run_name(args.source)
    run_dir = unique_run_dir(args.output_dir, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(
        run_dir / "params.json",
        {
            "source": str(args.source),
            "yolo_model": str(args.yolo_model),
            "widths": widths,
            "cell_size": args.cell_size,
            "min_valid_fraction": args.min_valid_fraction,
            "background": args.background,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "class_ids": class_ids,
            "mask_mode": args.mask_mode,
            "boxes_as_masks": args.boxes_as_masks,
        },
    )

    from ultralytics import YOLO

    model = YOLO(str(args.yolo_model))
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=class_ids,
        device=args.device,
        stream=True,
        verbose=False,
    )

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    print(f"Imagenes: {len(images)}")
    print(f"Salida: {run_dir}")

    for index, result in enumerate(results, start=1):
        image_path = images[index - 1]
        image = result.orig_img
        height, width = image.shape[:2]
        masks = polygons_to_masks(result, height, width)
        if not masks and args.boxes_as_masks:
            masks = boxes_to_masks(result, height, width)
        mask = combine_masks(masks, args.mask_mode) if masks else np.zeros((height, width), dtype=np.uint8)
        bbox = mask_bbox(mask)
        stem = output_stem(image_path, args.source)

        if bbox is None:
            summary_rows.append(
                {
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "has_mask": False,
                    "mask_area_ratio": 0.0,
                    "bbox_width": 0,
                    "bbox_height": 0,
                }
            )
            print(f"[{index}/{len(images)}] {image_path.name} sin mascara")
            continue

        x, y, bbox_width, bbox_height = bbox
        mask_area_ratio = int(np.count_nonzero(mask)) / float(width * height)
        summary_base = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "has_mask": True,
            "mask_area_ratio": round(mask_area_ratio, 6),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
        }

        for target_width in widths:
            width_dir = run_dir / f"width_{target_width}"
            normalized_dir = width_dir / "normalized"
            grid_dir = width_dir / "grid"
            mask_dir = width_dir / "masks"
            for directory in (normalized_dir, grid_dir, mask_dir):
                directory.mkdir(parents=True, exist_ok=True)

            normalized, normalized_mask, scale = crop_and_normalize(
                image=image,
                mask=mask,
                bbox=bbox,
                target_width=target_width,
                background=args.background,
            )
            cells = valid_cells(
                normalized_mask,
                cell_size=args.cell_size,
                min_valid_fraction=args.min_valid_fraction,
            )
            grid = draw_grid(normalized, normalized_mask, cells, args.cell_size)

            normalized_path = normalized_dir / stem.parent / f"{stem.name}_normalized.jpg"
            grid_path = grid_dir / stem.parent / f"{stem.name}_grid.jpg"
            mask_path = mask_dir / stem.parent / f"{stem.name}_mask.png"
            for path in (normalized_path, grid_path, mask_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(normalized_path), normalized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(mask_path), normalized_mask)

            for cell in cells:
                all_rows.append(
                    {
                        "image_path": str(image_path),
                        "image_name": image_path.name,
                        "width_variant": target_width,
                        "cell_id": cell["cell_id"],
                        "x_norm": cell["x"],
                        "y_norm": cell["y"],
                        "width_norm": cell["width"],
                        "height_norm": cell["height"],
                        "valid_fraction": cell["valid_fraction"],
                        "x_original": round(x + cell["x"] / scale, 3),
                        "y_original": round(y + cell["y"] / scale, 3),
                        "width_original": round(cell["width"] / scale, 3),
                        "height_original": round(cell["height"] / scale, 3),
                        "normalized_path": str(normalized_path),
                        "grid_path": str(grid_path),
                        "mask_path": str(mask_path),
                    }
                )
            summary_rows.append(
                {
                    **summary_base,
                    "width_variant": target_width,
                    "normalized_width": normalized.shape[1],
                    "normalized_height": normalized.shape[0],
                    "cell_count": len(cells),
                    "scale": round(scale, 6),
                    "normalized_path": str(normalized_path),
                    "grid_path": str(grid_path),
                    "mask_path": str(mask_path),
                }
            )

        print(f"[{index}/{len(images)}] {image_path.name} bbox={bbox_width}x{bbox_height}")

    write_csv(run_dir / "cells.csv", all_rows)
    write_csv(run_dir / "summary.csv", summary_rows)
    print(f"Resumen: {run_dir / 'summary.csv'}")
    print(f"Celdas: {run_dir / 'cells.csv'}")


if __name__ == "__main__":
    main()
