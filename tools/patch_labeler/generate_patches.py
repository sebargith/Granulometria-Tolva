from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

from patches import generate_patches


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return normalized.lower() or "patches"


def unique_output_dir(base: Path, run_name: str, overwrite: bool) -> Path:
    output = base / safe_name(run_name)
    if overwrite or not output.exists():
        return output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"{safe_name(run_name)}_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera parches trazables desde imagenes de carga.")
    parser.add_argument("--source", type=Path, required=True, help="Carpeta con imagenes o recortes.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "patches")
    parser.add_argument("--run-name", help="Nombre del experimento. Por defecto usa el input.")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--mask-dir", type=Path, help="Carpeta opcional con mascaras binarias.")
    parser.add_argument("--mask-black-background", action="store_true")
    parser.add_argument("--foreground-threshold", type=int, default=8)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--limit", type=int, help="Limita imagenes para pruebas.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"No existe la carpeta source: {source}")
    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("patch-size y stride deben ser mayores que cero")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise ValueError("min-valid-fraction debe estar entre 0 y 1")

    run_name = args.run_name or source.name
    output_dir = unique_output_dir(args.output_root.resolve(), run_name, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, rejected = generate_patches(
        source=source,
        output_dir=output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        min_valid_fraction=args.min_valid_fraction,
        mask_dir=args.mask_dir,
        mask_black_background=args.mask_black_background,
        foreground_threshold=args.foreground_threshold,
        recursive=not args.no_recursive,
        limit=args.limit,
    )

    params = {
        "source": str(source),
        "output_dir": str(output_dir),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "min_valid_fraction": args.min_valid_fraction,
        "mask_dir": str(args.mask_dir.resolve()) if args.mask_dir else None,
        "mask_black_background": args.mask_black_background,
        "foreground_threshold": args.foreground_threshold,
        "recursive": not args.no_recursive,
        "images_limit": args.limit,
        "patches_generated": len(records),
        "windows_rejected": rejected,
    }
    (output_dir / "params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Source: {source}")
    print(f"Parches generados: {len(records)}")
    print(f"Ventanas rechazadas: {rejected}")
    print(f"Manifest: {output_dir / 'patches.csv'}")
    print(f"Salida: {output_dir}")


if __name__ == "__main__":
    main()
