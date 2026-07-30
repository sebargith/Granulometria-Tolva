from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PREDICTOR = SCRIPT_DIR / "predict_carga.py"
PATCH_GENERATOR = PROJECT_ROOT / "tools" / "patch_labeler" / "generate_patches.py"
LABELER_LAUNCHER = PROJECT_ROOT / "tools" / "patch_labeler" / "launcher.py"
DEFAULT_MODEL = SCRIPT_DIR / "haul_truck_segment_m_736_low_data_aug_07012025.pt"


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return normalized.lower() or "patches_carga"


def available_run_name(base_name: str, roots: list[Path]) -> str:
    candidate = safe_name(base_name)
    if all(not (root / candidate).exists() for root in roots):
        return candidate
    return f"{candidate}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta carga con YOLO y genera patches listos para etiquetar."
    )
    parser.add_argument("--source", type=Path, required=True, help="Carpeta con frames originales.")
    parser.add_argument("--run-name", help="Nombre compartido por mascaras y patches.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=736)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--open-labeler", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    model = args.model.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"No existe la carpeta source: {source}")
    if not model.is_file():
        raise FileNotFoundError(f"No existe el modelo: {model}")

    yolo_root = PROJECT_ROOT / "outputs" / "yolo_carga"
    patches_root = PROJECT_ROOT / "outputs" / "patches"
    run_name = available_run_name(args.run_name or source.name, [yolo_root, patches_root])
    mask_dir = yolo_root / run_name / "masks"
    patches_dir = patches_root / run_name

    predict_command = [
        sys.executable,
        str(PREDICTOR),
        "--model",
        str(model),
        "--source",
        str(source),
        "--output-dir",
        str(yolo_root),
        "--run-name",
        run_name,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--class-ids",
        "1",
    ]
    if args.recursive:
        predict_command.append("--recursive")

    patch_command = [
        sys.executable,
        str(PATCH_GENERATOR),
        "--source",
        str(source),
        "--mask-dir",
        str(mask_dir),
        "--output-root",
        str(patches_root),
        "--run-name",
        run_name,
        "--patch-size",
        str(args.patch_size),
        "--stride",
        str(args.stride),
        "--min-valid-fraction",
        str(args.min_valid_fraction),
    ]
    if not args.recursive:
        patch_command.append("--no-recursive")

    print("\n1/2 Segmentando carga con YOLO...\n")
    run(predict_command)
    print("\n2/2 Generando patches dentro de la mascara...\n")
    run(patch_command)

    print("\nDataset listo para etiquetar:")
    print(patches_dir)
    print("\nPara abrir Streamlit:")
    print(f'python "{LABELER_LAUNCHER}" "{patches_dir}"')

    if args.open_labeler:
        run([sys.executable, str(LABELER_LAUNCHER), str(patches_dir)])


if __name__ == "__main__":
    main()
