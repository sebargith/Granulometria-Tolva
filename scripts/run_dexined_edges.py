from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from granulometria.dexined import (  # noqa: E402
    DexiNedConfig,
    apply_mask_to_edge,
    binarize_edge_map,
    infer_edge_map,
    load_dexined_model,
    make_edge_overlay,
    make_foreground_mask,
    parse_image_size,
    read_image_bgr,
    refine_binary_edges,
    write_image,
)
from granulometria.paths import LEGACY_DATA_DIR  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_DATASET = (
    LEGACY_DATA_DIR / "datasets_psinet" / "dataset_aug_rock_20241107"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera mapas de bordes con DexiNed para imagenes/recortes de carga. "
            "Pensado como primera etapa antes de extraer contornos y features."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "Imagen, carpeta de imagenes o raiz de dataset YOLO con images/train e images/val."
        ),
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Splits a usar si --source es raiz de dataset YOLO. Ejemplo: train,val o train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "dexined_edges",
        help="Carpeta base de salida.",
    )
    parser.add_argument("--run-name", default=None, help="Nombre de corrida.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Checkpoint local .pth de DexiNed. Si se omite, Kornia puede usar pesos preentrenados.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="No descargar/usar pesos preentrenados de Kornia.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--image-size",
        default="original",
        help="Tamano de entrada: 512, 736, 512x384, original o 0.",
    )
    parser.add_argument(
        "--preprocess",
        choices=("rgb", "gray", "gray_clahe"),
        default="gray_clahe",
        help="Preprocesamiento previo a DexiNed.",
    )
    parser.add_argument(
        "--normalization",
        choices=("bgr255_mean", "zero_one", "imagenet"),
        default="bgr255_mean",
        help=(
            "Normalizacion de entrada. bgr255_mean replica el preprocesamiento "
            "del DexiNed original: BGR 0-255 menos media."
        ),
    )
    parser.add_argument(
        "--output-transform",
        choices=("sigmoid", "none"),
        default="sigmoid",
        help="Transformacion aplicada a la salida del modelo antes de guardar.",
    )
    parser.add_argument(
        "--contrast-stretch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reescala el edge map por percentiles para que no quede visualmente negro.",
    )
    parser.add_argument(
        "--contrast-percentiles",
        default="1x99",
        help="Percentiles bajo/alto para --contrast-stretch. Ejemplo: 1x99 o 2x98.",
    )
    parser.add_argument(
        "--threshold",
        choices=("none", "fixed", "otsu", "percentile"),
        default="otsu",
        help="Binarizacion del mapa de bordes.",
    )
    parser.add_argument(
        "--threshold-value",
        type=int,
        default=90,
        help="Valor fijo 0-255 o percentil si --threshold percentile.",
    )
    parser.add_argument(
        "--mask-black-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enmascara fondo negro de recortes para evitar bordes artificiales.",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=int,
        default=8,
        help="Umbral de gris para separar primer plano del fondo negro.",
    )
    parser.add_argument(
        "--foreground-erode",
        type=int,
        default=8,
        help="Erosion de la mascara de primer plano para remover bordes del recorte.",
    )
    parser.add_argument(
        "--roi-rel",
        default=None,
        help=(
            "ROI rectangular relativa x1,y1,x2,y2 para ignorar bordes/timestamp. "
            "Ejemplo: 0.08,0.02,0.92,0.86."
        ),
    )
    parser.add_argument(
        "--binary-open-iterations",
        type=int,
        default=1,
        help="Limpieza morfologica de puntos/ruido en el borde binario.",
    )
    parser.add_argument(
        "--binary-close-iterations",
        type=int,
        default=0,
        help="Cierre morfologico para conectar bordes cercanos.",
    )
    parser.add_argument(
        "--thin-binary",
        action="store_true",
        help="Adelgaza bordes binarios para contornos/features mas finos.",
    )
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid-size", default="8x8")
    parser.add_argument(
        "--output-index",
        type=int,
        default=-1,
        help="Indice de salida DexiNed. -1 usa la fusion final multiescala.",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-binary", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_pair(value: str) -> tuple[int, int]:
    text = value.strip().lower()
    if "x" not in text:
        side = int(text)
        return side, side
    left, right = text.split("x", maxsplit=1)
    return int(left), int(right)


def parse_float_pair(value: str) -> tuple[float, float]:
    text = value.strip().lower()
    if "x" not in text:
        raise ValueError("Usa formato bajo x alto, por ejemplo: 1x99")
    left, right = text.split("x", maxsplit=1)
    return float(left), float(right)


def parse_roi_rel(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None or value.strip() == "":
        return None
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi-rel debe tener formato x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("--roi-rel debe estar en rango [0, 1] y cumplir x1<x2, y1<y2")
    return x1, y1, x2, y2


def make_roi_mask(shape: tuple[int, int], roi_rel: tuple[float, float, float, float] | None):
    if roi_rel is None:
        return None

    import numpy as np

    height, width = shape
    x1, y1, x2, y2 = roi_rel
    left = int(round(x1 * width))
    top = int(round(y1 * height))
    right = int(round(x2 * width))
    bottom = int(round(y2 * height))

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return mask


def combine_masks(mask_a, mask_b):
    if mask_a is None:
        return mask_b
    if mask_b is None:
        return mask_a

    import cv2

    return cv2.bitwise_and(mask_a, mask_b)


def make_run_dir(output_dir: Path, run_name: str | None) -> Path:
    if run_name is None:
        run_name = f"dexined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    (run_dir / "edge_maps").mkdir(parents=True, exist_ok=True)
    (run_dir / "binary_edges").mkdir(parents=True, exist_ok=True)
    (run_dir / "overlays").mkdir(parents=True, exist_ok=True)
    return run_dir


def list_images_in_dir(path: Path, recursive: bool) -> list[Path]:
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(
        item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def source_folder_name(source: Path) -> str:
    if source.is_file():
        return source.parent.name
    return source.name


def collect_image_jobs(
    source: Path,
    splits: list[str],
    recursive: bool,
    max_images: int | None,
) -> list[tuple[Path, Path]]:
    source = source.expanduser().resolve()
    source_name = source_folder_name(source)

    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"El archivo no parece imagen soportada: {source}")
        return [(source, Path(source_name) / source.name)]

    if not source.is_dir():
        raise FileNotFoundError(f"No existe la fuente indicada: {source}")

    dataset_images_dir = source / "images"
    jobs: list[tuple[Path, Path]] = []
    if dataset_images_dir.is_dir():
        for split in splits:
            split_dir = dataset_images_dir / split
            if not split_dir.is_dir():
                continue
            for image_path in list_images_in_dir(split_dir, recursive=recursive):
                jobs.append(
                    (
                        image_path,
                        Path(source_name) / split / image_path.relative_to(split_dir),
                    )
                )
    else:
        for image_path in list_images_in_dir(source, recursive=recursive):
            jobs.append((image_path, Path(source_name) / image_path.relative_to(source)))

    if max_images is not None:
        jobs = jobs[:max_images]
    return jobs


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def image_stats(image) -> dict[str, float | int]:
    import numpy as np

    return {
        "edge_min": int(image.min()),
        "edge_max": int(image.max()),
        "edge_mean": round(float(image.mean()), 6),
        "edge_nonzero": int(np.count_nonzero(image)),
    }


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve() if args.weights is not None else None
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    image_size = parse_image_size(args.image_size)
    clahe_tile_grid_size = parse_pair(args.clahe_tile_grid_size)
    contrast_percentiles = parse_float_pair(args.contrast_percentiles)
    roi_rel = parse_roi_rel(args.roi_rel)

    jobs = collect_image_jobs(
        source=source,
        splits=splits,
        recursive=args.recursive,
        max_images=args.max_images,
    )
    if not jobs:
        raise FileNotFoundError(f"No se encontraron imagenes en: {source}")

    run_dir = make_run_dir(output_dir, args.run_name)
    edge_dir = run_dir / "edge_maps"
    binary_dir = run_dir / "binary_edges"
    overlay_dir = run_dir / "overlays"
    csv_path = run_dir / "edge_outputs.csv"

    config = DexiNedConfig(
        weights=weights,
        pretrained=not args.no_pretrained,
        device=args.device,
        image_size=image_size,
        preprocess=args.preprocess,
        normalization=args.normalization,
        output_transform=args.output_transform,
        contrast_stretch=args.contrast_stretch,
        contrast_percentiles=contrast_percentiles,
        output_index=args.output_index,
        threshold_mode=args.threshold,
        threshold_value=args.threshold_value,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )

    save_json(
        run_dir / "params.json",
        {
            "source": str(source),
            "source_output_folder": source_folder_name(source),
            "splits": splits,
            "weights": None if weights is None else str(weights),
            "pretrained": config.pretrained,
            "device": args.device,
            "image_size": args.image_size,
            "preprocess": args.preprocess,
            "normalization": args.normalization,
            "output_transform": args.output_transform,
            "contrast_stretch": args.contrast_stretch,
            "contrast_percentiles": args.contrast_percentiles,
            "threshold": args.threshold,
            "threshold_value": args.threshold_value,
            "mask_black_background": args.mask_black_background,
            "foreground_threshold": args.foreground_threshold,
            "foreground_erode": args.foreground_erode,
            "roi_rel": args.roi_rel,
            "binary_open_iterations": args.binary_open_iterations,
            "binary_close_iterations": args.binary_close_iterations,
            "thin_binary": args.thin_binary,
            "clahe_clip_limit": args.clahe_clip_limit,
            "clahe_tile_grid_size": args.clahe_tile_grid_size,
            "output_index": args.output_index,
            "recursive": args.recursive,
            "max_images": args.max_images,
            "save_overlay": args.save_overlay,
            "save_binary": args.save_binary,
        },
    )

    print(f"Imagenes encontradas: {len(jobs)}")
    print(f"Salida: {run_dir}")
    print("Cargando DexiNed...")
    model, device = load_dexined_model(config)
    print(f"Dispositivo: {device}")

    rows: list[dict] = []
    for index, (image_path, relative_path) in enumerate(jobs, start=1):
        output_stem = relative_path.with_suffix("")
        edge_path = edge_dir / output_stem.parent / f"{output_stem.name}_edge.png"
        binary_path = binary_dir / output_stem.parent / f"{output_stem.name}_edge_bin.png"
        overlay_path = overlay_dir / output_stem.parent / f"{output_stem.name}_edge_overlay.jpg"

        image_bgr = read_image_bgr(image_path)
        foreground_mask = None
        if args.mask_black_background:
            foreground_mask = make_foreground_mask(
                image_bgr,
                threshold=args.foreground_threshold,
                erode_iterations=args.foreground_erode,
            )
        roi_mask = make_roi_mask(image_bgr.shape[:2], roi_rel)
        foreground_mask = combine_masks(foreground_mask, roi_mask)

        edge_map = infer_edge_map(image_bgr, model, device, config)
        if foreground_mask is not None:
            edge_map = apply_mask_to_edge(edge_map, foreground_mask)

        binary_edge = binarize_edge_map(
            edge_map,
            config.threshold_mode,
            config.threshold_value,
            valid_mask=foreground_mask,
        )
        if binary_edge is not None:
            binary_edge = refine_binary_edges(
                binary_edge,
                open_iterations=args.binary_open_iterations,
                close_iterations=args.binary_close_iterations,
                thin=args.thin_binary,
            )
            if foreground_mask is not None:
                binary_edge = apply_mask_to_edge(binary_edge, foreground_mask)

        stats = image_stats(edge_map)

        write_image(edge_path, edge_map)
        if args.save_binary and binary_edge is not None:
            write_image(binary_path, binary_edge)
        if args.save_overlay:
            overlay = make_edge_overlay(image_bgr, edge_map, binary_edge)
            write_image(overlay_path, overlay)

        rows.append(
            {
                "image_path": str(image_path),
                "image_name": image_path.name,
                "source_folder": relative_path.parts[0] if relative_path.parts else "",
                "edge_path": str(edge_path),
                "binary_path": str(binary_path) if args.save_binary and binary_edge is not None else "",
                "overlay_path": str(overlay_path) if args.save_overlay else "",
                "relative_path": str(relative_path),
                **stats,
            }
        )
        print(
            f"[{index}/{len(jobs)}] {relative_path} "
            f"edge_max={stats['edge_max']} edge_mean={stats['edge_mean']}"
        )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
