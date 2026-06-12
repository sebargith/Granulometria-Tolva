from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_DIR.parents[1]
APP_PATH = TOOL_DIR / "app.py"
FALLBACK_DATASET = PROJECT_ROOT / "Data parches test"


def latest_generated_dataset() -> Path:
    patches_root = PROJECT_ROOT / "outputs" / "patches"
    candidates = [
        path
        for path in patches_root.glob("*")
        if path.is_dir() and (path / "patches.csv").is_file()
    ]
    if not candidates:
        return FALLBACK_DATASET
    return max(candidates, key=lambda path: (path / "patches.csv").stat().st_mtime)


def available_port(start: int = 8501, attempts: int = 20) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No se encontro un puerto disponible para el etiquetador")


def open_when_ready(url: str) -> None:
    health_url = f"{url}/_stcore/health"
    for _ in range(60):
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(0.5)


def main() -> int:
    dataset = (
        Path(sys.argv[1]).expanduser().resolve()
        if len(sys.argv) > 1
        else latest_generated_dataset()
    )
    if not dataset.exists():
        print(f"No existe el dataset inicial: {dataset}")
        print("Puedes arrastrar una carpeta sobre abrir_etiquetador.cmd para usar otra ubicacion.")
        input("Presiona Enter para cerrar...")
        return 1

    port = available_port()
    url = f"http://127.0.0.1:{port}"
    threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "dark",
        "--theme.primaryColor",
        "#21b8aa",
        "--theme.backgroundColor",
        "#11171a",
        "--theme.secondaryBackgroundColor",
        "#182126",
        "--theme.textColor",
        "#edf2f3",
        "--",
        "--dataset",
        str(dataset),
    ]

    print("Etiquetador de granulometria")
    print(f"Dataset: {dataset}")
    print(f"Direccion: {url}")
    print("Manten esta ventana abierta mientras uses el etiquetador.")
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
