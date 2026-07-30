from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DEFAULT_REFERENCES = (
    PROJECT_ROOT / "data" / "labeling" / "references_normalized_20260618_203838"
)
DEFAULT_PROMPT = PROJECT_ROOT / "docs" / "labeling" / "PROMPT_ETIQUETADO_IA_GRILLA.md"


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._+-]+", "_", value).strip("._-")
    return cleaned or "frame"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea zips para etiquetado IA usando cargas normalizadas con grilla."
    )
    parser.add_argument(
        "--grid-run",
        type=Path,
        required=True,
        help="Carpeta generada por generar_grillas_normalizadas.py.",
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--references-dir", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--frames-per-zip", type=int, default=5)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument(
        "--package-prefix",
        default="batch001",
        help="Prefijo para los zips generados, por ejemplo batch002.",
    )
    parser.add_argument(
        "--mix-by-source-parent",
        action="store_true",
        help=(
            "Intercala frames por nombre de carpeta padre de image_path. "
            "Util para mezclar fina/media/grueso dentro de cada zip."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def reset_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for item in path.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_references(target_dir: Path, references_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(
        references_dir,
        target_dir,
        ignore=shutil.ignore_patterns(".gitkeep", "__pycache__"),
    )


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def frame_folder_name(image_name: str) -> str:
    return safe_name(Path(image_name).stem)


def source_parent_name(frame_cells: pd.DataFrame) -> str:
    image_path = Path(str(frame_cells.iloc[0]["image_path"]))
    parent = image_path.parent.name
    return safe_name(parent) if parent else "sin_clase"


def mixed_frame_order(frame_entries: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for entry in frame_entries:
        groups.setdefault(str(entry["source_group"]), []).append(entry)
    for entries in groups.values():
        entries.sort(key=lambda item: item["frame_dir"].name)

    ordered_groups = sorted(groups)
    mixed: list[dict] = []
    while any(groups[group] for group in ordered_groups):
        for group in ordered_groups:
            if groups[group]:
                mixed.append(groups[group].pop(0))
    return mixed


def prepare_frame_folder(
    frame_dir: Path,
    image_name: str,
    frame_cells: pd.DataFrame,
    grid_run: Path,
    width: int,
) -> dict:
    frame_name = frame_folder_name(image_name)
    frame_dir.mkdir(parents=True, exist_ok=True)

    first = frame_cells.iloc[0]
    normalized_path = Path(str(first["normalized_path"]))
    grid_path = Path(str(first["grid_path"]))
    mask_path = Path(str(first["mask_path"]))
    if not normalized_path.is_file():
        normalized_path = grid_run / f"width_{width}" / "normalized" / f"{frame_name}_normalized.jpg"
    if not grid_path.is_file():
        grid_path = grid_run / f"width_{width}" / "grid" / f"{frame_name}_grid.jpg"
    if not mask_path.is_file():
        mask_path = grid_run / f"width_{width}" / "masks" / f"{frame_name}_mask.png"

    copy_file(normalized_path, frame_dir / "01_carga_normalizada.jpg")
    copy_file(grid_path, frame_dir / "02_grilla_numerada.jpg")
    copy_file(mask_path, frame_dir / "03_mascara.png")

    cells_rows = []
    template_rows = []
    for row in frame_cells.sort_values("cell_id").itertuples(index=False):
        cell_id = str(row.cell_id).zfill(3)
        cells_rows.append(
            {
                "cell_id": cell_id,
                "x_norm": int(row.x_norm),
                "y_norm": int(row.y_norm),
                "width_norm": int(row.width_norm),
                "height_norm": int(row.height_norm),
                "valid_fraction": float(row.valid_fraction),
                "image_name": image_name,
            }
        )
        template_rows.append(
            {
                "cell_id": cell_id,
                "label": "",
                "confidence": "",
                "notes": "",
            }
        )

    write_csv(frame_dir / "cells.csv", cells_rows)
    write_csv(
        frame_dir / "cells_template.csv",
        template_rows,
        fieldnames=["cell_id", "label", "confidence", "notes"],
    )
    with (frame_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "image_name": image_name,
                "frame_folder": frame_name,
                "width": width,
                "cell_count": len(cells_rows),
                "normalized_image": "01_carga_normalizada.jpg",
                "grid_image": "02_grilla_numerada.jpg",
                "mask_image": "03_mascara.png",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "frame_folder": frame_name,
        "image_name": image_name,
        "cells": len(cells_rows),
    }


def main() -> None:
    args = parse_args()
    grid_run = args.grid_run.resolve()
    batch_dir = args.batch_dir.resolve()
    references_dir = args.references_dir.resolve()
    prompt = args.prompt.resolve()

    cells_path = grid_run / "cells.csv"
    if not cells_path.is_file():
        raise FileNotFoundError(f"No existe cells.csv: {cells_path}")
    if not references_dir.is_dir():
        raise NotADirectoryError(f"No existe references-dir: {references_dir}")
    if not prompt.is_file():
        raise FileNotFoundError(f"No existe prompt: {prompt}")

    frames_output = batch_dir / "frames_a_etiquetar"
    zip_output = batch_dir / "zip_para_ia"
    responses_dir = batch_dir / "respuestas_ia"
    reviewed_dir = batch_dir / "labels_revisadas"
    for folder in (frames_output, zip_output, responses_dir, reviewed_dir):
        reset_dir(folder, args.overwrite)
        (folder / ".gitkeep").touch(exist_ok=True)

    cells = pd.read_csv(cells_path, encoding="utf-8-sig", dtype={"cell_id": str})
    cells = cells[cells["width_variant"].astype(int) == int(args.width)].copy()
    if cells.empty:
        raise ValueError(f"No hay celdas para width_variant={args.width}")

    frame_summaries: list[dict] = []
    frame_entries: list[dict] = []
    for image_name, frame_cells in cells.groupby("image_name", sort=True):
        frame_name = frame_folder_name(str(image_name))
        frame_dir = frames_output / frame_name
        summary = prepare_frame_folder(
            frame_dir=frame_dir,
            image_name=str(image_name),
            frame_cells=frame_cells,
            grid_run=grid_run,
            width=args.width,
        )
        source_group = source_parent_name(frame_cells)
        summary["source_group"] = source_group
        frame_summaries.append(summary)
        frame_entries.append({"frame_dir": frame_dir, "source_group": source_group})

    if args.mix_by_source_parent:
        frame_entries = mixed_frame_order(frame_entries)
    else:
        frame_entries.sort(key=lambda item: item["frame_dir"].name)

    batches_root = zip_output / "_batches"
    if batches_root.exists():
        shutil.rmtree(batches_root)
    batches_root.mkdir(parents=True)

    batch_rows = []
    for batch_index, start in enumerate(range(0, len(frame_entries), args.frames_per_zip), start=1):
        batch_entries = frame_entries[start : start + args.frames_per_zip]
        batch_frame_dirs = [entry["frame_dir"] for entry in batch_entries]
        package_name = f"{safe_name(args.package_prefix)}_part{batch_index:03d}"
        package_dir = batches_root / package_name
        package_dir.mkdir(parents=True)
        copy_references(package_dir / "referencias", references_dir)
        copy_file(prompt, package_dir / "PROMPT_ETIQUETADO_IA_GRILLA.md")
        target_frames = package_dir / "frames_a_etiquetar"
        target_frames.mkdir()

        cell_count = 0
        for frame_dir in batch_frame_dirs:
            shutil.copytree(frame_dir, target_frames / frame_dir.name)
            frame_cells = pd.read_csv(frame_dir / "cells.csv", encoding="utf-8-sig")
            cell_count += len(frame_cells)

        zip_path = zip_output / f"{package_name}.zip"
        zip_directory(package_dir, zip_path)
        batch_rows.append(
            {
                "package": package_name,
                "zip_path": str(zip_path),
                "frames": len(batch_frame_dirs),
                "cells": cell_count,
                "source_groups": ";".join(str(entry["source_group"]) for entry in batch_entries),
                "first_frame": batch_frame_dirs[0].name,
                "last_frame": batch_frame_dirs[-1].name,
            }
        )
        print(f"{package_name}: {len(batch_frame_dirs)} frames, {cell_count} celdas")

    write_csv(batch_dir / "frames_manifest.csv", frame_summaries)
    write_csv(batch_dir / "zip_para_ia" / "batches_manifest.csv", batch_rows)
    print(f"Frames preparados: {frames_output}")
    print(f"ZIPs: {zip_output}")
    print(f"Total frames: {len(frame_entries)}")
    print(f"Total celdas: {len(cells)}")


if __name__ == "__main__":
    main()
