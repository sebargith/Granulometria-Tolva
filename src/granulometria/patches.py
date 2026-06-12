from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    patch_path: str
    source_image: str
    source_relpath: str
    sequence: str
    x: int
    y: int
    width: int
    height: int
    image_width: int
    image_height: int
    valid_fraction: float


def list_images(source: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path.resolve()
        for path in source.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length < patch_size:
        return []
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def find_mask(image_path: Path, source: Path, mask_dir: Path) -> Path | None:
    relative = image_path.relative_to(source.resolve())
    candidates = [
        mask_dir / relative,
        mask_dir / relative.with_name(f"{relative.stem}_mask{relative.suffix}"),
        mask_dir / relative.with_suffix(".png"),
        mask_dir / relative.with_name(f"{relative.stem}_mask.png"),
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def load_valid_mask(
    image: np.ndarray,
    image_path: Path,
    source: Path,
    mask_dir: Path | None,
    mask_black_background: bool,
    foreground_threshold: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    if mask_dir is not None:
        mask_path = find_mask(image_path, source, mask_dir.resolve())
        if mask_path is None:
            raise FileNotFoundError(f"No se encontro mascara para {image_path.name}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"No se pudo leer la mascara: {mask_path}")
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)

    if mask_black_background:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return (gray > foreground_threshold).astype(np.uint8)

    return np.ones((height, width), dtype=np.uint8)


def generate_patches(
    source: Path,
    output_dir: Path,
    patch_size: int,
    stride: int,
    min_valid_fraction: float,
    mask_dir: Path | None = None,
    mask_black_background: bool = False,
    foreground_threshold: int = 8,
    recursive: bool = True,
    limit: int | None = None,
) -> tuple[list[PatchRecord], int]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    patches_dir = output_dir / "images"
    patches_dir.mkdir(parents=True, exist_ok=True)

    records: list[PatchRecord] = []
    rejected = 0
    images = list_images(source, recursive=recursive)
    if limit is not None:
        images = images[:limit]

    for image_index, image_path in enumerate(images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rejected += 1
            continue

        height, width = image.shape[:2]
        valid_mask = load_valid_mask(
            image=image,
            image_path=image_path,
            source=source,
            mask_dir=mask_dir,
            mask_black_background=mask_black_background,
            foreground_threshold=foreground_threshold,
        )
        relative = image_path.relative_to(source)
        relative_key = "__".join(relative.with_suffix("").parts)
        sequence = relative.parts[0] if len(relative.parts) > 1 else source.name

        for y in sliding_positions(height, patch_size, stride):
            for x in sliding_positions(width, patch_size, stride):
                fraction = float(valid_mask[y : y + patch_size, x : x + patch_size].mean())
                if fraction < min_valid_fraction:
                    rejected += 1
                    continue

                patch_id = f"{relative_key}__x{x:05d}_y{y:05d}"
                patch_path = patches_dir / f"{patch_id}.jpg"
                patch = image[y : y + patch_size, x : x + patch_size]
                if not cv2.imwrite(str(patch_path), patch, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"No se pudo guardar el parche: {patch_path}")

                records.append(
                    PatchRecord(
                        patch_id=patch_id,
                        patch_path=str(patch_path.relative_to(output_dir)),
                        source_image=str(image_path),
                        source_relpath=str(relative),
                        sequence=sequence,
                        x=x,
                        y=y,
                        width=patch_size,
                        height=patch_size,
                        image_width=width,
                        image_height=height,
                        valid_fraction=round(fraction, 6),
                    )
                )

    write_manifest(output_dir / "patches.csv", records)
    return records, rejected


def write_manifest(path: Path, records: Iterable[PatchRecord]) -> None:
    rows = [asdict(record) for record in records]
    fieldnames = list(PatchRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
