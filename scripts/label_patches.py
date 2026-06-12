from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABELS = ["fino", "medio", "grueso", "mixto", "invalido", "ambiguo"]
LABEL_TITLES = {
    "fino": "Fino",
    "medio": "Medio",
    "grueso": "Grueso",
    "mixto": "Mixto",
    "invalido": "Invalido",
    "ambiguo": "Ambiguo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", default="Data parches test")
    parser.add_argument("--labels-file")
    args, _ = parser.parse_known_args()
    return args


def resolve_dataset(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def load_dataset(dataset: Path) -> tuple[pd.DataFrame, Path]:
    if dataset.is_file() and dataset.suffix.lower() == ".csv":
        manifest = dataset
        root = manifest.parent
    elif dataset.is_dir() and (dataset / "patches.csv").is_file():
        manifest = dataset / "patches.csv"
        root = dataset
    elif dataset.is_dir():
        images = sorted(
            path.resolve()
            for path in dataset.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        rows = [
            {
                "patch_id": str(path.relative_to(dataset).with_suffix("")).replace(os.sep, "__"),
                "patch_path": str(path),
                "source_image": "",
                "source_relpath": path.name,
                "sequence": dataset.name,
                "x": "",
                "y": "",
                "width": "",
                "height": "",
                "valid_fraction": 1.0,
            }
            for path in images
        ]
        return pd.DataFrame(rows), dataset
    else:
        raise FileNotFoundError(f"No existe el dataset: {dataset}")

    frame = pd.read_csv(manifest, dtype={"patch_id": str})
    required = {"patch_id", "patch_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {manifest.name}: {', '.join(sorted(missing))}")

    def absolute_patch(value: object) -> str:
        path = Path(str(value))
        return str(path if path.is_absolute() else (root / path).resolve())

    frame["patch_path"] = frame["patch_path"].map(absolute_patch)
    return frame, root


def load_labels(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(
            columns=[
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
            ]
        )
    labels = pd.read_csv(path, dtype={"patch_id": str}, keep_default_na=False)
    return labels.drop_duplicates(subset="patch_id", keep="last")


def save_label(path: Path, labels: pd.DataFrame, record: pd.Series, label: str) -> pd.DataFrame:
    values = {
        "patch_id": str(record["patch_id"]),
        "annotator": st.session_state.get("annotator", "").strip(),
        "label": label,
        "confidence": st.session_state.confidence,
        "notes": st.session_state.notes.strip(),
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "patch_path": str(record["patch_path"]),
        "source_image": str(record.get("source_image", "")),
        "x": record.get("x", ""),
        "y": record.get("y", ""),
        "width": record.get("width", ""),
        "height": record.get("height", ""),
    }
    labels = labels[labels["patch_id"].astype(str) != values["patch_id"]]
    labels = pd.concat([labels, pd.DataFrame([values])], ignore_index=True)
    labels = labels.sort_values("patch_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    labels.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    return labels


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def context_image(record: pd.Series) -> Image.Image | None:
    source_value = str(record.get("source_image", "")).strip()
    if not source_value or not Path(source_value).is_file():
        return None
    coordinate_names = ["x", "y", "width", "height"]
    if any(pd.isna(record.get(name)) or str(record.get(name)).strip() == "" for name in coordinate_names):
        return None
    image = load_rgb(Path(source_value))
    draw = ImageDraw.Draw(image)
    x, y, width, height = (int(float(record[name])) for name in coordinate_names)
    line_width = max(3, round(min(image.size) / 250))
    draw.rectangle((x, y, x + width, y + height), outline=(0, 220, 180), width=line_width)
    return image


def canny_image(image: Image.Image) -> Image.Image:
    rgb = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 7, 40, 40)
    edges = cv2.Canny(blurred, 60, 140)
    return Image.fromarray(edges)


def label_counts(labels: pd.DataFrame) -> dict[str, int]:
    counts = labels["label"].value_counts().to_dict() if not labels.empty else {}
    return {label: int(counts.get(label, 0)) for label in LABELS}


args = parse_args()
dataset_path = resolve_dataset(args.dataset)

st.set_page_config(page_title="Etiquetado granulometrico", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f5f7f8; color: #182126; }
    [data-testid="stHeader"] { background: rgba(245, 247, 248, 0.94); }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dce3e6; padding: 12px; border-radius: 6px; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #182126; }
    .stButton button { min-height: 44px; border-radius: 6px; font-weight: 650; }
    [data-testid="stImage"] img { border: 1px solid #cfd8dc; border-radius: 4px; max-height: 330px; object-fit: contain; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    dataset, dataset_root = load_dataset(dataset_path)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

if dataset.empty:
    st.warning("No se encontraron imagenes para etiquetar.")
    st.stop()

labels_path = resolve_dataset(args.labels_file) if args.labels_file else dataset_root / "labels.csv"
labels = load_labels(labels_path)
label_by_id = dict(zip(labels["patch_id"].astype(str), labels["label"], strict=False))
dataset["current_label"] = dataset["patch_id"].astype(str).map(label_by_id).fillna("")

dataset_key = f"{dataset_path}|{labels_path}"
if st.session_state.get("dataset_key") != dataset_key:
    st.session_state.dataset_key = dataset_key
    st.session_state.position = 0
    st.session_state.confidence = "alta"
    st.session_state.notes = ""

with st.sidebar:
    st.title("Granulometria")
    st.caption(dataset_root.name)
    view = st.selectbox(
        "Vista",
        ["Pendientes", "Todos", *[LABEL_TITLES[label] for label in LABELS]],
    )
    show_edges = st.toggle("Mostrar Canny", value=False)
    st.text_input("Anotador", key="annotator")
    st.divider()
    counts = label_counts(labels)
    for label in LABELS:
        st.write(f"{LABEL_TITLES[label]}: **{counts[label]}**")
    st.caption(str(labels_path))

if view == "Pendientes":
    visible = dataset[dataset["current_label"] == ""].copy()
elif view == "Todos":
    visible = dataset.copy()
else:
    selected_label = next(key for key, title in LABEL_TITLES.items() if title == view)
    visible = dataset[dataset["current_label"] == selected_label].copy()

visible = visible.reset_index(drop=True)
if visible.empty:
    st.success("No hay parches en esta vista.")
    st.stop()

st.session_state.position = min(st.session_state.position, len(visible) - 1)
record = visible.iloc[st.session_state.position]

header_left, header_progress = st.columns([3, 2])
with header_left:
    st.title("Etiquetado de parches")
    st.caption(str(record["source_relpath"]) if "source_relpath" in record else str(record["patch_id"]))
with header_progress:
    total_labeled = int((dataset["current_label"] != "").sum())
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Etiquetados", f"{total_labeled}/{len(dataset)}")
    metric_b.metric("Pendientes", len(dataset) - total_labeled)
    st.progress(total_labeled / len(dataset))

patch_path = Path(str(record["patch_path"]))
if not patch_path.is_file():
    st.error(f"No existe el parche: {patch_path}")
    st.stop()

patch_image = load_rgb(patch_path)
context = context_image(record)
panels: list[tuple[str, Image.Image]] = [("Parche", patch_image)]
if context is not None:
    panels.append(("Ubicacion en frame", context))
if show_edges:
    panels.append(("Bordes Canny", canny_image(patch_image)))
for column, (title, image) in zip(st.columns(len(panels)), panels, strict=True):
    with column:
        st.subheader(title)
        st.image(image, width="stretch")

current_label = str(record.get("current_label", ""))
if current_label:
    st.info(f"Etiqueta actual: {LABEL_TITLES.get(current_label, current_label)}")

st.radio("Confianza", ["alta", "media", "baja"], horizontal=True, key="confidence")
if st.session_state.pop("_clear_notes", False):
    st.session_state.notes = ""
st.text_input("Notas", key="notes", placeholder="Opcional")

button_columns = st.columns(len(LABELS))
clicked_label = None
for column, label in zip(button_columns, LABELS, strict=True):
    if column.button(LABEL_TITLES[label], width="stretch", key=f"label_{label}"):
        clicked_label = label

if clicked_label is not None:
    labels = save_label(labels_path, labels, record, clicked_label)
    st.session_state._clear_notes = True
    st.rerun()

previous_column, position_column, next_column, skip_column = st.columns([1, 2, 1, 1])
if previous_column.button("Anterior", width="stretch", disabled=st.session_state.position == 0):
    st.session_state.position -= 1
    st.session_state._clear_notes = True
    st.rerun()
position_column.caption(f"Elemento {st.session_state.position + 1} de {len(visible)} en la vista actual")
if next_column.button(
    "Siguiente", width="stretch", disabled=st.session_state.position >= len(visible) - 1
):
    st.session_state.position += 1
    st.session_state._clear_notes = True
    st.rerun()
if skip_column.button("Saltar", width="stretch"):
    st.session_state.position = (st.session_state.position + 1) % len(visible)
    st.session_state._clear_notes = True
    st.rerun()
