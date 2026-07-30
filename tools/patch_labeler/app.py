from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}
LABELS = ["fino", "medio", "grueso", "mixto", "invalido", "ambiguo"]
CONFIDENCES = ["alta", "media", "baja"]
LABEL_COLUMNS = [
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
LABEL_TITLES = {
    "fino": "Fino",
    "medio": "Medio",
    "grueso": "Grueso",
    "mixto": "Mixto",
    "invalido": "Invalido",
    "ambiguo": "Ambiguo",
}
CONFIDENCE_TITLES = {
    "alta": "Alta",
    "media": "Media",
    "baja": "Baja",
}
CONFIDENCE_COLORS = {
    "alta": "#27d17f",
    "media": "#f7c948",
    "baja": "#ff6b5f",
}
TOOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from granulometria.metrics import compute_granulometry_metrics
from granulometria.paths import (
    FINAL_CNN_MODEL,
    FINAL_RF_MODEL,
    INFERENCE_SCRIPT,
    PATCH_FRAME_DATASET_DIR,
    VIDEO_PIPELINE_OUTPUT_DIR,
    PERCEPTION_MODEL,
)


ICON_PATH = TOOL_DIR / "assets" / "patch_labeler.ico"
STATE_FILE_NAME = ".patch_labeler_state.json"
VIDEO_PIPELINE_ROOT = VIDEO_PIPELINE_OUTPUT_DIR
DEFAULT_YOLO_MODEL = PERCEPTION_MODEL
DEFAULT_CNN_MODEL = FINAL_CNN_MODEL
DEFAULT_RF_MODEL = FINAL_RF_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", default=str(PATCH_FRAME_DATASET_DIR))
    parser.add_argument("--labels-file")
    args, _ = parser.parse_known_args()
    return args


def resolve_dataset(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def first_existing(paths: list[Path]) -> str:
    for path in paths:
        if path.is_file():
            return str(path.resolve())
    return ""


def file_cache_token(path: Path) -> tuple[bool, int, int]:
    if not path.is_file():
        return False, 0, 0
    stat = path.stat()
    return True, int(stat.st_mtime_ns), int(stat.st_size)


def dataset_cache_token(dataset: Path) -> tuple:
    if dataset.is_file() and dataset.suffix.lower() == ".csv":
        manifest = dataset
        root = dataset.parent
    elif dataset.is_dir() and (dataset / "patches.csv").is_file():
        manifest = dataset / "patches.csv"
        root = dataset
    else:
        return ("raw_images", str(dataset), 0)

    return (
        str(manifest.resolve()),
        file_cache_token(manifest),
        file_cache_token(root / "params.json"),
    )


def read_dataset_params(root: Path) -> dict:
    params_path = root / "params.json"
    if not params_path.is_file():
        return {}
    try:
        return json.loads(params_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def frame_folder_from_values(patch_id: object, source_image: object = "") -> str:
    patch_text = str(patch_id)
    if "__cell_" in patch_text:
        return patch_text.split("__cell_", 1)[0]
    source_text = str(source_image).strip()
    if source_text:
        return Path(source_text).stem
    return ""


def enrich_context_paths(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    frame = frame.copy()
    params = read_dataset_params(root)
    batch_dir = Path(str(params.get("batch_dir", ""))) if params.get("batch_dir") else None
    frames_dir = batch_dir / "frames_a_etiquetar" if batch_dir else None

    if "frame_folder" not in frame.columns:
        frame["frame_folder"] = frame.apply(
            lambda row: frame_folder_from_values(
                row.get("patch_id", ""),
                row.get("source_image", ""),
            ),
            axis=1,
        )

    def context_paths(row: pd.Series) -> pd.Series:
        frame_folder = str(row.get("frame_folder", "")).strip()
        source_value = str(row.get("source_image", "")).strip()
        source_candidates: list[Path] = []
        grid_candidates: list[Path] = []

        if frames_dir is not None and frame_folder:
            frame_dir = frames_dir / frame_folder
            source_candidates.extend(
                [
                    frame_dir / "01_carga_normalizada.jpg",
                    frame_dir / "01_carga_normalizada.png",
                ]
            )
            grid_candidates.extend(
                [
                    frame_dir / "02_grilla_numerada.jpg",
                    frame_dir / "02_grilla_numerada.png",
                ]
            )

        if source_value:
            source_path = Path(source_value)
            source_candidates.append(source_path if source_path.is_absolute() else root / source_path)

        return pd.Series(
            {
                "source_frame_path": first_existing(source_candidates),
                "grid_frame_path": first_existing(grid_candidates),
            }
        )

    context_frame = frame.apply(context_paths, axis=1)
    for column in context_frame.columns:
        frame[column] = context_frame[column]
    return frame


def load_dataset_uncached(dataset: Path) -> tuple[pd.DataFrame, Path]:
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
    frame = enrich_context_paths(frame, root)
    return frame, root


@st.cache_data(show_spinner=False, max_entries=8)
def load_dataset_cached(dataset_text: str, token: tuple) -> tuple[pd.DataFrame, str]:
    dataset, root = load_dataset_uncached(Path(dataset_text))
    return dataset, str(root)


def load_labels_uncached(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    labels = pd.read_csv(path, dtype={"patch_id": str}, keep_default_na=False)
    for column in LABEL_COLUMNS:
        if column not in labels.columns:
            labels[column] = ""
    preferred_columns = LABEL_COLUMNS + [column for column in labels.columns if column not in LABEL_COLUMNS]
    return labels[preferred_columns].drop_duplicates(subset="patch_id", keep="last")


@st.cache_data(show_spinner=False, max_entries=16)
def load_labels_cached(path_text: str, token: tuple) -> pd.DataFrame:
    return load_labels_uncached(Path(path_text))


def label_values(
    record: pd.Series,
    label: str,
    confidence: str,
    notes: str,
    annotator: str,
) -> dict[str, object]:
    return {
        "patch_id": str(record["patch_id"]),
        "annotator": annotator.strip(),
        "label": label,
        "confidence": confidence,
        "notes": notes.strip(),
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "patch_path": str(record["patch_path"]),
        "source_image": str(record.get("source_image", "")),
        "x": record.get("x", ""),
        "y": record.get("y", ""),
        "width": record.get("width", ""),
        "height": record.get("height", ""),
    }


def write_labels(path: Path, labels: pd.DataFrame) -> None:
    labels = labels.sort_values("patch_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    labels.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def upsert_label(
    path: Path,
    labels: pd.DataFrame,
    record: pd.Series,
    label: str,
    confidence: str,
    notes: str,
    annotator: str,
) -> pd.DataFrame:
    values = label_values(record, label, confidence, notes, annotator)
    same_patch = labels["patch_id"].astype(str) == values["patch_id"]
    if same_patch.any():
        previous = labels.loc[same_patch].iloc[-1].to_dict()
        previous.update(values)
        values = previous
    labels = labels[labels["patch_id"].astype(str) != values["patch_id"]]
    labels = pd.concat([labels, pd.DataFrame([values])], ignore_index=True)
    write_labels(path, labels)
    return labels


def save_label(path: Path, labels: pd.DataFrame, record: pd.Series, label: str) -> pd.DataFrame:
    return upsert_label(
        path=path,
        labels=labels,
        record=record,
        label=label,
        confidence=st.session_state.confidence,
        notes=st.session_state.notes,
        annotator=st.session_state.get("annotator", ""),
    )


@st.cache_data(show_spinner=False, max_entries=512)
def load_rgb_cached(path_text: str, token: tuple) -> Image.Image:
    with Image.open(path_text) as image:
        return image.convert("RGB")


def load_rgb(path: Path) -> Image.Image:
    path = path.resolve()
    return load_rgb_cached(str(path), file_cache_token(path)).copy()


def record_image_path(record: pd.Series, use_grid: bool = False) -> Path | None:
    preferred_columns = ["grid_frame_path", "source_frame_path"] if use_grid else ["source_frame_path", "grid_frame_path"]
    for column in preferred_columns:
        value = str(record.get(column, "")).strip()
        if value and Path(value).is_file():
            return Path(value)

    source_value = str(record.get("source_image", "")).strip()
    if source_value and Path(source_value).is_file():
        return Path(source_value)
    return None


def context_image(record: pd.Series, use_grid: bool = False) -> Image.Image | None:
    image_path = record_image_path(record, use_grid=use_grid)
    if image_path is None:
        return None
    coordinate_names = ["x", "y", "width", "height"]
    if any(pd.isna(record.get(name)) or str(record.get(name)).strip() == "" for name in coordinate_names):
        return None
    image = load_rgb(image_path)
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


def option_title(value: str) -> str:
    if value == "todas":
        return "Todas"
    return LABEL_TITLES.get(value, value.title())


def confidence_title(value: str) -> str:
    if value == "todas":
        return "Todas"
    return CONFIDENCE_TITLES.get(value, value.title())


def safe_key(value: object) -> str:
    text = str(value)
    return "".join(character if character.isalnum() else "_" for character in text)[:90]


def state_path(dataset_root: Path) -> Path:
    return dataset_root / STATE_FILE_NAME


def load_app_state(dataset_root: Path) -> dict:
    path = state_path(dataset_root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_app_state(dataset_root: Path, **updates: object) -> None:
    path = state_path(dataset_root)
    payload = load_app_state(dataset_root)
    payload.update(updates)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def index_for_patch_id(frame: pd.DataFrame, patch_id: str) -> int | None:
    if not patch_id:
        return None
    matches = frame.index[frame["patch_id"].astype(str) == str(patch_id)].tolist()
    return int(matches[0]) if matches else None


def confidence_badge(value: str) -> str:
    confidence = str(value).strip().lower()
    color = CONFIDENCE_COLORS.get(confidence, "#8a98a0")
    title = CONFIDENCE_TITLES.get(confidence, confidence.title() or "Sin confianza")
    return (
        f"<span class='confidence-badge' style='"
        f"background:{color}; color:#0f171a;'>{title}</span>"
    )


def install_review_shortcuts() -> None:
    components.html(
        """
        <script>
        const parentDocument = window.parent.document;
        if (!window.parent.__patchLabelerReviewShortcutsInstalled) {
            window.parent.__patchLabelerReviewShortcutsInstalled = true;
            parentDocument.addEventListener("keydown", (event) => {
                const key = (event.key || "").toLowerCase();
                if (key !== "a" && key !== "d") return;
                if (event.ctrlKey || event.metaKey || event.altKey) return;

                const target = event.target;
                const tagName = (target && target.tagName ? target.tagName : "").toLowerCase();
                if (["input", "textarea", "select"].includes(tagName)) return;
                if (target && target.isContentEditable) return;

                const label = key === "a" ? "Anterior (A)" : "Siguiente (D)";
                const buttons = Array.from(parentDocument.querySelectorAll("button"));
                const button = buttons.find((item) => item.innerText.trim() === label);
                if (button && !button.disabled) {
                    event.preventDefault();
                    button.click();
                }
            }, true);
        }
        </script>
        """,
        height=0,
        width=0,
    )


def current_value(record: pd.Series, name: str, default: str = "") -> str:
    value = record.get(name, default)
    if pd.isna(value):
        return default
    return str(value)


def selected_labels_filter(label_filter: list[str]) -> list[str]:
    return [label for label in label_filter if label in LABELS]


def render_review_grid(
    dataset: pd.DataFrame,
    labels: pd.DataFrame,
    labels_path: Path,
    show_edges: bool,
    label_filter: list[str],
    confidence_filter: str,
    only_labeled: bool,
    search_text: str,
    page_size: int,
    show_context: bool,
    show_grid_context: bool,
) -> None:
    review = dataset.copy()
    if only_labeled:
        review = review[review["current_label"] != ""].copy()
    selected_labels = selected_labels_filter(label_filter)
    review = review[review["current_label"].isin(selected_labels)].copy()
    if confidence_filter != "todas":
        review = review[review["current_confidence"] == confidence_filter].copy()

    search = search_text.strip().lower()
    if search:
        search_columns = [
            "patch_id",
            "source_relpath",
            "source_image",
            "current_notes",
            "current_annotator",
        ]
        search_mask = pd.Series(False, index=review.index)
        for column in search_columns:
            if column in review.columns:
                search_mask = search_mask | review[column].astype(str).str.lower().str.contains(
                    search, regex=False, na=False
                )
        review = review[search_mask].copy()

    filter_key = "|".join(
        [
            ",".join(selected_labels),
            confidence_filter,
            str(only_labeled),
            search,
            str(page_size),
        ]
    )
    if st.session_state.get("review_filter_key") != filter_key:
        st.session_state.review_filter_key = filter_key
        st.session_state.review_page = 0

    review = review.reset_index(drop=True)
    total_labeled = int((dataset["current_label"] != "").sum())

    header_left, header_metrics = st.columns([3, 2])
    with header_left:
        st.title("Revision de patches")
        st.caption(str(labels_path))
    with header_metrics:
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Parches", len(dataset))
        metric_b.metric("Etiquetados", total_labeled)
        metric_c.metric("Filtrados", len(review))

    if review.empty:
        st.info("No hay patches que coincidan con los filtros de revision.")
        return

    total_pages = max(1, int(np.ceil(len(review) / page_size)))
    st.session_state.review_page = min(st.session_state.get("review_page", 0), total_pages - 1)
    start = st.session_state.review_page * page_size
    end = start + page_size
    page = review.iloc[start:end]

    nav_prev, nav_status, nav_next = st.columns([1, 3, 1])
    if nav_prev.button("Pagina anterior", width="stretch", disabled=st.session_state.review_page == 0):
        st.session_state.review_page -= 1
        st.rerun()
    nav_status.caption(
        f"Pagina {st.session_state.review_page + 1} de {total_pages} - "
        f"mostrando {start + 1}-{min(end, len(review))} de {len(review)}"
    )
    if nav_next.button(
        "Pagina siguiente",
        width="stretch",
        disabled=st.session_state.review_page >= total_pages - 1,
    ):
        st.session_state.review_page += 1
        st.rerun()

    st.divider()
    columns = st.columns(4)
    for index, (_, record) in enumerate(page.iterrows()):
        patch_id = str(record["patch_id"])
        key = safe_key(patch_id)
        with columns[index % 4]:
            with st.container(border=True):
                patch_path = Path(str(record["patch_path"]))
                patch_image = None
                if patch_path.is_file():
                    patch_image = load_rgb(patch_path)
                    st.image(patch_image, width="stretch")
                else:
                    st.warning(f"No existe: {patch_path.name}")

                current_label = current_value(record, "current_label")
                current_confidence = current_value(record, "current_confidence", "alta")
                current_notes = current_value(record, "current_notes")
                current_annotator = current_value(record, "current_annotator")
                source_name = current_value(record, "source_relpath", patch_path.name)

                st.markdown(f"**{LABEL_TITLES.get(current_label, 'Sin etiqueta')}**")
                st.caption(source_name)
                if current_annotator:
                    st.caption(f"Anotador: {current_annotator}")

                label_index = LABELS.index(current_label) if current_label in LABELS else 0
                confidence_index = (
                    CONFIDENCES.index(current_confidence)
                    if current_confidence in CONFIDENCES
                    else 0
                )
                new_label = st.selectbox(
                    "Etiqueta",
                    LABELS,
                    index=label_index,
                    format_func=lambda value: LABEL_TITLES[value],
                    key=f"review_label_{key}",
                )
                new_confidence = st.selectbox(
                    "Confianza",
                    CONFIDENCES,
                    index=confidence_index,
                    format_func=str.title,
                    key=f"review_confidence_{key}",
                )
                new_notes = st.text_input(
                    "Notas",
                    value=current_notes,
                    key=f"review_notes_{key}",
                )

                changed = (
                    new_label != current_label
                    or new_confidence != current_confidence
                    or new_notes.strip() != current_notes.strip()
                )
                if st.button("Guardar revision", width="stretch", key=f"review_save_{key}", disabled=not changed):
                    annotator = st.session_state.get("annotator", "").strip() or "revision"
                    upsert_label(
                        path=labels_path,
                        labels=labels,
                        record=record,
                        label=new_label,
                        confidence=new_confidence,
                        notes=new_notes,
                        annotator=annotator,
                    )
                    st.rerun()

                if show_context:
                    context = context_image(record, use_grid=show_grid_context)
                    if context is not None:
                        with st.expander("Contexto"):
                            st.image(context, width="stretch")
                if show_edges and patch_image is not None:
                    with st.expander("Canny"):
                        st.image(canny_image(patch_image), width="stretch")


def filtered_review_dataset(
    dataset: pd.DataFrame,
    label_filter: list[str],
    confidence_filter: str,
    only_labeled: bool,
    search_text: str,
) -> pd.DataFrame:
    review = dataset.copy()
    if only_labeled:
        review = review[review["current_label"] != ""].copy()
    selected_labels = selected_labels_filter(label_filter)
    review = review[review["current_label"].isin(selected_labels)].copy()
    if confidence_filter != "todas":
        review = review[review["current_confidence"] == confidence_filter].copy()

    search = search_text.strip().lower()
    if search:
        search_columns = [
            "patch_id",
            "source_relpath",
            "source_image",
            "frame_folder",
            "current_notes",
            "current_annotator",
        ]
        search_mask = pd.Series(False, index=review.index)
        for column in search_columns:
            if column in review.columns:
                search_mask = search_mask | review[column].astype(str).str.lower().str.contains(
                    search, regex=False, na=False
                )
        review = review[search_mask].copy()
    return review.reset_index(drop=True)


def save_quick_review(
    dataset_root: Path,
    review: pd.DataFrame,
    labels_path: Path,
    labels: pd.DataFrame,
    record: pd.Series,
    label: str,
    confidence: str,
    notes: str,
    auto_advance: bool,
    total_visible: int,
) -> None:
    annotator = st.session_state.get("annotator", "").strip() or "revision"
    upsert_label(
        path=labels_path,
        labels=labels,
        record=record,
        label=label,
        confidence=confidence,
        notes=notes,
        annotator=annotator,
    )
    if auto_advance and total_visible > 1:
        next_index = min(
            st.session_state.get("review_index", 0) + 1,
            total_visible - 1,
        )
        st.session_state.review_index = next_index
        save_app_state(dataset_root, review_patch_id=str(review.iloc[next_index]["patch_id"]))
    else:
        save_app_state(dataset_root, review_patch_id=str(record["patch_id"]))
    st.rerun()


def render_review_quick(
    dataset_root: Path,
    dataset: pd.DataFrame,
    labels: pd.DataFrame,
    labels_path: Path,
    show_edges: bool,
    label_filter: list[str],
    confidence_filter: str,
    only_labeled: bool,
    search_text: str,
    show_context: bool,
    show_grid_context: bool,
    auto_advance: bool,
) -> None:
    review = filtered_review_dataset(
        dataset=dataset,
        label_filter=label_filter,
        confidence_filter=confidence_filter,
        only_labeled=only_labeled,
        search_text=search_text,
    )
    filter_key = "|".join(
        [
            "rapida",
            ",".join(selected_labels_filter(label_filter)),
            confidence_filter,
            str(only_labeled),
            search_text.strip().lower(),
        ]
    )
    if st.session_state.get("review_filter_key") != filter_key:
        st.session_state.review_filter_key = filter_key
        restored_index = index_for_patch_id(
            review,
            str(load_app_state(dataset_root).get("review_patch_id", "")),
        )
        st.session_state.review_index = restored_index if restored_index is not None else 0

    total_labeled = int((dataset["current_label"] != "").sum())
    header_left, header_metrics = st.columns([3, 2])
    with header_left:
        st.title("Revision rapida de patches")
        st.caption(str(labels_path))
    with header_metrics:
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Parches", len(dataset))
        metric_b.metric("Etiquetados", total_labeled)
        metric_c.metric("Filtrados", len(review))

    if review.empty:
        st.info("No hay patches que coincidan con los filtros de revision.")
        return

    st.session_state.review_index = min(st.session_state.get("review_index", 0), len(review) - 1)
    record = review.iloc[st.session_state.review_index]
    save_app_state(dataset_root, review_patch_id=str(record["patch_id"]))
    patch_id = str(record["patch_id"])
    key = safe_key(patch_id)
    patch_path = Path(str(record["patch_path"]))
    current_label = current_value(record, "current_label")
    current_confidence = current_value(record, "current_confidence", "alta") or "alta"
    current_notes = current_value(record, "current_notes")
    current_annotator = current_value(record, "current_annotator")

    st.caption(f"Patch {st.session_state.review_index + 1} de {len(review)} en la vista filtrada")
    install_review_shortcuts()

    patch_column, source_column = st.columns([1, 2])
    with patch_column:
        st.subheader("Patch")
        patch_image = None
        if patch_path.is_file():
            patch_image = load_rgb(patch_path)
            st.image(patch_image, width="stretch")
        else:
            st.warning(f"No existe: {patch_path}")

        st.markdown(
            f"**Actual:** {LABEL_TITLES.get(current_label, 'Sin etiqueta')}"
            f"&nbsp;&nbsp;{confidence_badge(current_confidence)}",
            unsafe_allow_html=True,
        )
        if current_confidence:
            st.caption(f"Confianza: {confidence_title(current_confidence)}")
        if current_annotator:
            st.caption(f"Anotador: {current_annotator}")
        st.caption(str(record.get("source_relpath", patch_path.name)))

    with source_column:
        st.subheader("Imagen original")
        if show_context:
            context = context_image(record, use_grid=show_grid_context)
            if context is not None:
                st.image(context, width="stretch")
            else:
                st.info("No hay imagen original resuelta para este patch.")
        else:
            st.info("Activa 'Mostrar imagen original' en la barra lateral.")

    if show_edges and patch_image is not None:
        with st.expander("Bordes Canny del patch"):
            st.image(canny_image(patch_image), width="stretch")

    confidence_index = CONFIDENCES.index(current_confidence) if current_confidence in CONFIDENCES else 0
    selected_confidence = st.radio(
        "Confianza",
        CONFIDENCES,
        index=confidence_index,
        horizontal=True,
        format_func=confidence_title,
        key=f"quick_confidence_{key}",
    )
    st.markdown(
        f"Seleccionada: {confidence_badge(selected_confidence)}",
        unsafe_allow_html=True,
    )
    selected_notes = st.text_input(
        "Notas",
        value=current_notes,
        key=f"quick_notes_{key}",
        placeholder="Opcional",
    )

    primary_columns = st.columns(3)
    for column, label in zip(primary_columns, ["fino", "medio", "grueso"], strict=True):
        if column.button(
            LABEL_TITLES[label],
            width="stretch",
            key=f"quick_primary_{label}_{key}",
            type="primary" if label == current_label else "secondary",
        ):
            save_quick_review(
                dataset_root=dataset_root,
                review=review,
                labels_path=labels_path,
                labels=labels,
                record=record,
                label=label,
                confidence=selected_confidence,
                notes=selected_notes,
                auto_advance=auto_advance,
                total_visible=len(review),
            )

    secondary_columns = st.columns(3)
    for column, label in zip(secondary_columns, ["mixto", "ambiguo", "invalido"], strict=True):
        if column.button(
            LABEL_TITLES[label],
            width="stretch",
            key=f"quick_secondary_{label}_{key}",
            type="primary" if label == current_label else "secondary",
        ):
            save_quick_review(
                dataset_root=dataset_root,
                review=review,
                labels_path=labels_path,
                labels=labels,
                record=record,
                label=label,
                confidence=selected_confidence,
                notes=selected_notes,
                auto_advance=auto_advance,
                total_visible=len(review),
            )

    if current_label in LABELS:
        if st.button("Guardar confianza/notas y seguir", width="stretch", key=f"quick_save_current_{key}"):
            save_quick_review(
                dataset_root=dataset_root,
                review=review,
                labels_path=labels_path,
                labels=labels,
                record=record,
                label=current_label,
                confidence=selected_confidence,
                notes=selected_notes,
                auto_advance=auto_advance,
                total_visible=len(review),
            )

    nav_prev, nav_status, nav_next = st.columns([1.2, 2, 1.2])
    if nav_prev.button(
        "Anterior (A)",
        width="stretch",
        disabled=st.session_state.review_index == 0,
        key=f"quick_prev_{key}",
    ):
        st.session_state.review_index -= 1
        save_app_state(
            dataset_root,
            review_patch_id=str(review.iloc[st.session_state.review_index]["patch_id"]),
        )
        st.rerun()
    nav_status.caption(
        f"Patch {st.session_state.review_index + 1} de {len(review)} en la vista filtrada"
    )
    if nav_next.button(
        "Siguiente (D)",
        width="stretch",
        disabled=st.session_state.review_index >= len(review) - 1,
        key=f"quick_next_{key}",
    ):
        st.session_state.review_index += 1
        save_app_state(
            dataset_root,
            review_patch_id=str(review.iloc[st.session_state.review_index]["patch_id"]),
        )
        st.rerun()


def render_review(
    dataset_root: Path,
    dataset: pd.DataFrame,
    labels: pd.DataFrame,
    labels_path: Path,
    show_edges: bool,
    label_filter: list[str],
    confidence_filter: str,
    only_labeled: bool,
    search_text: str,
    page_size: int,
    show_context: bool,
    show_grid_context: bool,
    review_layout: str,
    auto_advance: bool,
) -> None:
    if review_layout == "Rapida":
        render_review_quick(
            dataset_root=dataset_root,
            dataset=dataset,
            labels=labels,
            labels_path=labels_path,
            show_edges=show_edges,
            label_filter=label_filter,
            confidence_filter=confidence_filter,
            only_labeled=only_labeled,
            search_text=search_text,
            show_context=show_context,
            show_grid_context=show_grid_context,
            auto_advance=auto_advance,
        )
        return

    render_review_grid(
        dataset=dataset,
        labels=labels,
        labels_path=labels_path,
        show_edges=show_edges,
        label_filter=label_filter,
        confidence_filter=confidence_filter,
        only_labeled=only_labeled,
        search_text=search_text,
        page_size=page_size,
        show_context=show_context,
        show_grid_context=show_grid_context,
    )


def slugify(value: str, fallback: str = "corrida") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("._-")
    return normalized or fallback


def unique_child_dir(parent: Path, base_name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    base = slugify(base_name)
    candidate = parent / base
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = parent / f"{base}_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No se pudo crear un directorio unico para {base_name}")


def resolve_existing_video(value: str) -> Path | None:
    value = value.strip().strip('"')
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No existe el video: {path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Extension de video no soportada: {path.suffix}")
    return path


def copy_video_to_run(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def save_uploaded_video(uploaded_file, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Extension de video no soportada: {suffix}")
    destination = destination_dir / slugify(Path(uploaded_file.name).stem, "video")
    destination = destination.with_suffix(suffix)
    with destination.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return destination


def extract_video_frames(
    video_path: Path,
    frames_dir: Path,
    interval_seconds: float,
    max_frames: int,
    jpeg_quality: int = 95,
) -> dict:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    interval_seconds = max(float(interval_seconds), 0.01)
    frame_step = max(1, int(round(interval_seconds * fps))) if fps > 0 else 1

    if frame_count > 0:
        frame_indices = list(range(0, frame_count, frame_step))
    else:
        frame_indices = []

    if max_frames > 0:
        frame_indices = frame_indices[:max_frames]

    saved = 0
    failed = 0
    if frame_indices:
        for source_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                failed += 1
                continue
            timestamp_seconds = source_index / fps if fps > 0 else float(saved)
            output_name = (
                f"{video_path.stem}_frame{saved:05d}_src{source_index:08d}_"
                f"t{timestamp_seconds:08.2f}s.jpg"
            )
            output_path = frames_dir / output_name
            if cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                saved += 1
            else:
                failed += 1
    else:
        source_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if source_index % frame_step == 0:
                output_name = f"{video_path.stem}_frame{saved:05d}_src{source_index:08d}.jpg"
                output_path = frames_dir / output_name
                if cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    saved += 1
                    if max_frames > 0 and saved >= max_frames:
                        break
                else:
                    failed += 1
            source_index += 1

    capture.release()
    if saved == 0:
        raise RuntimeError("No se pudo extraer ningun frame del video.")

    return {
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "fps": fps,
        "source_frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "interval_seconds": interval_seconds,
        "frame_step": frame_step,
        "frames_extracted": saved,
        "failed_reads": failed,
    }


def parse_inference_output_dir(stdout: str, output_dir: Path, run_name: str) -> Path:
    for line in stdout.splitlines():
        if line.strip().startswith("Salida:"):
            return Path(line.split("Salida:", 1)[1].strip()).resolve()
    candidates = sorted(
        [path for path in output_dir.glob(f"{slugify(run_name)}*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    raise RuntimeError("La inferencia termino, pero no se pudo resolver la carpeta de salida.")


def run_granulometry_inference(
    frames_dir: Path,
    run_name: str,
    output_dir: Path,
    patch_model: Path,
    patch_model_type: str,
    yolo_model: Path,
    interval_config: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--source",
        str(frames_dir),
        "--patch-model",
        str(patch_model),
        "--patch-model-type",
        patch_model_type,
        "--yolo-model",
        str(yolo_model),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--imgsz",
        str(interval_config["imgsz"]),
        "--conf",
        str(interval_config["conf"]),
        "--iou",
        str(interval_config["iou"]),
        "--class-ids",
        str(interval_config["class_ids"]),
        "--mask-mode",
        str(interval_config["mask_mode"]),
        "--patch-size",
        str(interval_config["patch_size"]),
        "--stride",
        str(interval_config["stride"]),
        "--min-valid-fraction",
        str(interval_config["min_valid_fraction"]),
        "--min-mask-area-ratio",
        str(interval_config["min_mask_area_ratio"]),
        "--overlay-alpha",
        str(interval_config["overlay_alpha"]),
        "--normalize-width",
        str(interval_config["normalize_width"]),
        "--normalized-background",
        str(interval_config["normalized_background"]),
    ]
    if interval_config.get("patch_device"):
        command.extend(["--patch-device", str(interval_config["patch_device"])])
    if interval_config.get("device"):
        command.extend(["--device", str(interval_config["device"])])
    if interval_config.get("boxes_as_masks"):
        command.append("--boxes-as-masks")
    if interval_config.get("no_save_patches"):
        command.append("--no-save-patches")

    started = time.time()
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - started
    if process.returncode != 0:
        raise RuntimeError(
            "Fallo la inferencia de granulometria.\n\n"
            f"Comando: {' '.join(command)}\n\n"
            f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}"
        )
    inference_dir = parse_inference_output_dir(process.stdout, output_dir, run_name)
    return {
        "command": command,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "inference_dir": str(inference_dir),
    }


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def aggregate_granulometry(summary_csv: Path) -> dict:
    frame = pd.read_csv(summary_csv)
    patch_counts = numeric_series(frame, "patch_count")
    valid = frame[patch_counts > 0].copy()
    valid_patch_counts = numeric_series(valid, "patch_count")
    total_patches = int(valid_patch_counts.sum())

    if total_patches <= 0 or valid.empty:
        return {
            "frames_total": int(len(frame)),
            "frames_valid": 0,
            "valid_ratio": 0.0,
            "patches_total": 0,
            "share_fino": 0.0,
            "share_medio": 0.0,
            "share_grueso": 0.0,
            "dominant_label": "",
            "mean_patch_confidence": 0.0,
            "dominant_counts": {},
            "indice_granulometrico": None,
            "dispersion_granulometrica": None,
            "aggregation_method": "patch_weighted_valid_frames",
        }

    weighted_shares: dict[str, float] = {}
    for label in ("fino", "medio", "grueso"):
        values = numeric_series(valid, f"share_{label}")
        weighted_shares[f"share_{label}"] = float((values * valid_patch_counts).sum() / total_patches)

    confidence = numeric_series(valid, "mean_patch_confidence")
    mean_confidence = float((confidence * valid_patch_counts).sum() / total_patches)
    dominant_label = max(
        ("fino", "medio", "grueso"),
        key=lambda label: weighted_shares.get(f"share_{label}", 0.0),
    )
    dominant_counts = Counter(str(value) for value in valid.get("dominant_label", []) if str(value))
    metrics = compute_granulometry_metrics(
        {
            "fino": weighted_shares["share_fino"],
            "medio": weighted_shares["share_medio"],
            "grueso": weighted_shares["share_grueso"],
        }
    )

    return {
        "frames_total": int(len(frame)),
        "frames_valid": int(len(valid)),
        "valid_ratio": round(len(valid) / len(frame), 6) if len(frame) else 0.0,
        "patches_total": total_patches,
        "share_fino": round(weighted_shares["share_fino"], 6),
        "share_medio": round(weighted_shares["share_medio"], 6),
        "share_grueso": round(weighted_shares["share_grueso"], 6),
        "dominant_label": dominant_label,
        "mean_patch_confidence": round(mean_confidence, 6),
        "dominant_counts": dict(dominant_counts),
        "indice_granulometrico": round(
            float(metrics["indice_granulometrico"]),
            6,
        ),
        "dispersion_granulometrica": round(
            float(metrics["dispersion_granulometrica"]),
            6,
        ),
        "aggregation_method": "patch_weighted_valid_frames",
    }


def save_pipeline_summary(run_dir: Path, payload: dict) -> Path:
    summary_path = run_dir / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return summary_path


def save_video_summary_csv(run_dir: Path, payload: dict) -> Path:
    aggregate = dict(payload.get("aggregate", {}))
    aggregate.pop("dominant_counts", None)
    summary_path = run_dir / "video_summary.csv"
    pd.DataFrame(
        [
            {
                "run_name": payload.get("run_name", run_dir.name),
                "video_name": payload.get("video_name", ""),
                "video_path": payload.get("video_path", ""),
                "inference_dir": payload.get("inference_dir", ""),
                **aggregate,
            }
        ]
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path


def load_pipeline_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "pipeline_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def existing_pipeline_runs() -> list[Path]:
    if not VIDEO_PIPELINE_ROOT.is_dir():
        return []
    return sorted(
        [
            path
            for path in VIDEO_PIPELINE_ROOT.iterdir()
            if path.is_dir() and (path / "pipeline_summary.json").is_file()
        ],
        key=lambda path: (path / "pipeline_summary.json").stat().st_mtime,
        reverse=True,
    )


def percentage(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def render_metric_card(label: str, value: object, description: str) -> None:
    st.markdown(
        (
            '<div class="summary-metric-card">'
            f'<div class="summary-metric-label">{html.escape(label)}</div>'
            f'<div class="summary-metric-value">{html.escape(str(value))}</div>'
            f'<div class="summary-metric-description">{html.escape(description)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_download_button(label: str, path: Path) -> None:
    if not path.is_file():
        return
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv" if path.suffix.lower() == ".csv" else "application/octet-stream",
        width="stretch",
    )


def render_pipeline_results(run_dir: Path) -> None:
    payload = load_pipeline_summary(run_dir)
    if not payload:
        st.info("Selecciona o ejecuta una corrida para ver resultados.")
        return

    inference_dir = Path(str(payload.get("inference_dir", "")))
    summary_csv = inference_dir / "summary.csv"
    details_csv = inference_dir / "details_por_patch.csv"
    aggregate = payload.get("aggregate", {})
    aggregate_metrics = compute_granulometry_metrics(
        {
            "fino": aggregate.get("share_fino", 0.0),
            "medio": aggregate.get("share_medio", 0.0),
            "grueso": aggregate.get("share_grueso", 0.0),
        }
    )
    granulometry_index = aggregate.get(
        "indice_granulometrico",
        aggregate_metrics["indice_granulometrico"],
    )
    granulometry_dispersion = aggregate.get(
        "dispersion_granulometrica",
        aggregate_metrics["dispersion_granulometrica"],
    )

    st.title("Granulometria por video")
    st.subheader(str(payload.get("video_name", run_dir.name)))
    st.caption(str(run_dir))
    st.caption(
        "Tiempo de inferencia (YOLO + clasificador): "
        f"{float(payload.get('inference_elapsed_seconds', 0.0)):.1f} s"
    )

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    with metric_a:
        render_metric_card(
            "Frames extraidos",
            payload.get("frames_extracted", 0),
            "Imagenes tomadas del video segun el intervalo configurado.",
        )
    with metric_b:
        render_metric_card(
            "Frames validos",
            aggregate.get("frames_valid", 0),
            "Frames con carga detectada y patches suficientes para evaluar.",
        )
    with metric_c:
        render_metric_card(
            "Patches",
            aggregate.get("patches_total", 0),
            "Regiones de la carga clasificadas en fino, medio o grueso.",
        )
    with metric_d:
        render_metric_card(
            "Confianza media",
            percentage(aggregate.get("mean_patch_confidence", 0.0)),
            "Rango: 0% a 100%. Un valor mayor indica mayor seguridad del modelo.",
        )

    share_a, share_b, share_c, share_d = st.columns(4)
    with share_a:
        render_metric_card(
            "Fino",
            percentage(aggregate.get("share_fino", 0.0)),
            "Proporcion de patches clasificados como material fino.",
        )
    with share_b:
        render_metric_card(
            "Medio",
            percentage(aggregate.get("share_medio", 0.0)),
            "Proporcion de patches clasificados como material medio.",
        )
    with share_c:
        render_metric_card(
            "Grueso",
            percentage(aggregate.get("share_grueso", 0.0)),
            "Proporcion de patches clasificados como material grueso.",
        )
    with share_d:
        render_metric_card(
            "Dominante",
            str(aggregate.get("dominant_label", "")).upper() or "N/D",
            "Clase con el porcentaje mas alto entre fino, medio y grueso.",
        )

    indicator_a, indicator_b, indicator_c = st.columns(3)
    with indicator_a:
        render_metric_card(
            "Indice granulometrico",
            f"{float(granulometry_index):.3f}" if granulometry_index is not None else "N/D",
            "Rango: 1 a 3. 1 es fino, 2 es medio y 3 es grueso.",
        )
    with indicator_b:
        render_metric_card(
            "Dispersion granulometrica",
            (
                f"{float(granulometry_dispersion):.3f}"
                if granulometry_dispersion is not None
                else "N/D"
            ),
            "Rango: 0 a 1. 0 es homogenea y 1 es la maxima separacion de la mezcla.",
        )
    with indicator_c:
        render_metric_card(
            "Frames evaluables",
            percentage(aggregate.get("valid_ratio", 0.0)),
            "Porcentaje de frames extraidos que entregaron un resultado valido.",
        )

    if not summary_csv.is_file():
        st.warning(f"No existe summary.csv en {inference_dir}")
        return

    summary = pd.read_csv(summary_csv)
    overlays = [
        Path(str(path))
        for path in summary.get("granulometry_overlay_path", pd.Series(dtype=str)).tolist()
        if str(path).strip() and Path(str(path)).is_file()
    ]

    st.subheader("Overlays")
    if overlays:
        max_items = st.slider("Cantidad", min_value=1, max_value=len(overlays), value=min(6, len(overlays)))
        for row_start in range(0, max_items, 3):
            columns = st.columns(3)
            for column, overlay_path in zip(columns, overlays[row_start : row_start + 3], strict=False):
                with column:
                    st.image(str(overlay_path), width="stretch")
                    st.caption(overlay_path.name)
    else:
        st.info("No hay overlays disponibles.")

    st.subheader("Resultado por frame")
    display_columns = [
        "image_name",
        "patch_count",
        "share_fino",
        "share_medio",
        "share_grueso",
        "dominant_label",
        "indice_granulometrico",
        "dispersion_granulometrica",
        "mean_patch_confidence",
        "max_yolo_conf",
        "mask_area_ratio",
    ]
    available_columns = [column for column in display_columns if column in summary.columns]
    st.dataframe(summary[available_columns], width="stretch", hide_index=True)

    st.subheader("Descargas")
    download_columns = st.columns(4)
    with download_columns[0]:
        render_download_button("summary.csv", summary_csv)
    with download_columns[1]:
        render_download_button("details_por_patch.csv", details_csv)
    with download_columns[2]:
        render_download_button("video_summary.csv", run_dir / "video_summary.csv")
    with download_columns[3]:
        render_download_button("pipeline_summary.json", run_dir / "pipeline_summary.json")

    with st.expander("Estado tecnico de la corrida"):
        st.json(payload)


def render_video_pipeline() -> None:
    VIDEO_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    existing_runs = existing_pipeline_runs()

    with st.sidebar:
        st.title("Granulometria")
        st.caption("Pipeline video")
        st.subheader("Nueva corrida")
        default_run = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_name = st.text_input("run_name", value=default_run)
        uploaded_video = st.file_uploader(
            "Video",
            type=[extension.lstrip(".") for extension in sorted(VIDEO_EXTENSIONS)],
        )
        existing_video_text = st.text_input("Ruta de video existente", value="")
        interval_seconds = st.number_input("interval_seconds", min_value=0.1, value=1.0, step=0.1)
        max_frames = st.number_input("max_frames_video (0 = sin limite)", min_value=0, value=50, step=1)
        yolo_conf = st.slider("YOLO conf", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        yolo_device = st.text_input("device YOLO", value="0")
        patch_device = st.selectbox("device CNN", ["cuda", "cpu", ""], index=0)

        with st.expander("Parametros avanzados"):
            model_kind = st.selectbox("Modelo patches", ["CNN ResNet18", "Random Forest"], index=0)
            patch_model_default = DEFAULT_CNN_MODEL if model_kind == "CNN ResNet18" else DEFAULT_RF_MODEL
            patch_model_text = st.text_input("patch_model", value=str(patch_model_default))
            yolo_model_text = st.text_input("yolo_model", value=str(DEFAULT_YOLO_MODEL))
            imgsz = st.number_input("imgsz", min_value=128, value=736, step=32)
            iou = st.slider("YOLO IoU", min_value=0.1, max_value=0.95, value=0.7, step=0.05)
            patch_size = st.number_input("patch_size", min_value=32, value=96, step=16)
            stride = st.number_input("stride", min_value=32, value=96, step=16)
            normalize_width = st.number_input("normalize_width", min_value=0, value=896, step=32)
            min_valid_fraction = st.slider("min_valid_fraction", 0.0, 1.0, 0.5, 0.05)
            min_mask_area_ratio = st.slider("min_mask_area_ratio", 0.0, 0.5, 0.01, 0.01)
            overlay_alpha = st.slider("overlay_alpha", 0.05, 0.95, 0.45, 0.05)
            normalized_background = st.selectbox("normalized_background", ["black", "dim", "original"], index=0)
            mask_mode = st.selectbox("mask_mode", ["union", "largest"], index=0)
            class_ids = st.text_input("class_ids YOLO", value="1")
            no_save_patches = st.toggle("No guardar patches clasificados", value=False)
            boxes_as_masks = st.toggle("Usar cajas si no hay mascara", value=False)

        execute = st.button("Ejecutar pipeline", type="primary", width="stretch")
        st.divider()
        if existing_runs:
            selected_run_name = st.selectbox(
                "Corrida",
                [run.name for run in existing_runs],
                index=0,
            )
        else:
            selected_run_name = ""
            st.caption("Sin corridas previas")

    if execute:
        try:
            if uploaded_video is None and not existing_video_text.strip():
                raise ValueError("Sube un video o indica una ruta de video existente.")

            run_dir = unique_child_dir(VIDEO_PIPELINE_ROOT, run_name)
            video_input_dir = run_dir / "input"
            if uploaded_video is not None:
                video_path = save_uploaded_video(uploaded_video, video_input_dir)
            else:
                source_video = resolve_existing_video(existing_video_text)
                if source_video is None:
                    raise ValueError("La ruta del video esta vacia.")
                video_path = copy_video_to_run(source_video, video_input_dir)

            frames_dir = run_dir / "frames"
            inference_output_dir = run_dir / "inferencia"
            patch_model = Path(patch_model_text).expanduser().resolve()
            yolo_model = Path(yolo_model_text).expanduser().resolve()
            patch_model_type = "cnn" if model_kind == "CNN ResNet18" else "rf"

            with st.status("Procesando video", expanded=True) as status:
                status.write("Extrayendo frames del video.")
                extraction = extract_video_frames(
                    video_path=video_path,
                    frames_dir=frames_dir,
                    interval_seconds=float(interval_seconds),
                    max_frames=int(max_frames),
                )
                status.write(f"Frames extraidos: {extraction['frames_extracted']}")
                status.write("Ejecutando YOLO + clasificador de granulometria.")
                inference = run_granulometry_inference(
                    frames_dir=frames_dir,
                    run_name=slugify(run_name),
                    output_dir=inference_output_dir,
                    patch_model=patch_model,
                    patch_model_type=patch_model_type,
                    yolo_model=yolo_model,
                    interval_config={
                        "imgsz": int(imgsz),
                        "conf": float(yolo_conf),
                        "iou": float(iou),
                        "device": yolo_device.strip(),
                        "patch_device": patch_device.strip(),
                        "class_ids": class_ids.strip(),
                        "mask_mode": mask_mode,
                        "boxes_as_masks": bool(boxes_as_masks),
                        "patch_size": int(patch_size),
                        "stride": int(stride),
                        "min_valid_fraction": float(min_valid_fraction),
                        "min_mask_area_ratio": float(min_mask_area_ratio),
                        "overlay_alpha": float(overlay_alpha),
                        "normalize_width": int(normalize_width),
                        "normalized_background": normalized_background,
                        "no_save_patches": bool(no_save_patches),
                    },
                )
                summary_csv = Path(inference["inference_dir"]) / "summary.csv"
                aggregate = aggregate_granulometry(summary_csv)
                payload = {
                    "run_name": run_dir.name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "video_name": video_path.name,
                    "video_path": str(video_path),
                    "frames_dir": str(frames_dir),
                    "frames_extracted": extraction["frames_extracted"],
                    "extraction": extraction,
                    "inference_dir": inference["inference_dir"],
                    "inference_elapsed_seconds": round(float(inference["elapsed_seconds"]), 3),
                    "patch_model": str(patch_model),
                    "patch_model_type": patch_model_type,
                    "yolo_model": str(yolo_model),
                    "aggregate": aggregate,
                    "command": inference["command"],
                    "stdout": inference["stdout"],
                    "stderr": inference["stderr"],
                }
                save_pipeline_summary(run_dir, payload)
                save_video_summary_csv(run_dir, payload)
                status.update(label="Pipeline terminado", state="complete")
            st.success(f"Corrida lista: {run_dir}")
            render_pipeline_results(run_dir)
            return
        except Exception as error:
            st.error(str(error))
            return

    selected_run = VIDEO_PIPELINE_ROOT / selected_run_name if selected_run_name else None
    if selected_run is not None and selected_run.is_dir():
        render_pipeline_results(selected_run)
    else:
        st.title("Granulometria por video")
        st.info("Sube un video y ejecuta el pipeline para generar la primera corrida.")


args = parse_args()
dataset_path = resolve_dataset(args.dataset)

if ICON_PATH.is_file():
    with Image.open(ICON_PATH) as icon_image:
        page_icon = icon_image.copy()
else:
    page_icon = None
st.set_page_config(
    page_title="Etiquetado granulometrico",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #11171a; color: #edf2f3; }
    [data-testid="stHeader"] { background: rgba(17, 23, 26, 0.94); }
    [data-testid="stSidebar"] { background: #182126; border-right: 1px solid #344249; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] span,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stRadio"] p { color: #edf2f3 !important; }
    [data-testid="stMetric"] {
        background: #1c272c;
        border: 1px solid #344249;
        padding: 12px;
        border-radius: 6px;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #edf2f3; }
    .stButton > button {
        min-height: 44px;
        border-radius: 6px;
        border: 1px solid #506169;
        background: #243137;
        color: #ffffff !important;
        font-weight: 650;
    }
    .stButton > button p { color: #ffffff !important; }
    .stButton > button:hover {
        border-color: #21b8aa;
        background: #087f78;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"] {
        border-color: #21d6c6 !important;
        background: #13a79d !important;
        box-shadow: 0 0 0 2px rgba(33, 214, 198, 0.22);
    }
    .stButton > button[kind="primary"] p {
        color: #ffffff !important;
        font-weight: 800;
    }
    .stButton > button:disabled {
        border-color: #303c41;
        background: #1a2226;
        color: #6f7d83 !important;
    }
    .stButton > button:disabled p { color: #6f7d83 !important; }
    [data-testid="stTextInput"] input {
        border: 1px solid #506169;
        background: #1c272c !important;
        color: #edf2f3 !important;
    }
    [data-testid="stTextInput"] input::placeholder { color: #94a2a8 !important; opacity: 1; }
    [data-testid="stSelectbox"] > div > div {
        background: #1c272c;
        border-color: #506169;
        color: #edf2f3;
    }
    [data-testid="stImage"] img {
        border: 1px solid #425158;
        border-radius: 4px;
        max-height: 560px;
        object-fit: contain;
    }
    .confidence-badge {
        display: inline-block;
        min-width: 58px;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        text-align: center;
        vertical-align: middle;
    }
    .summary-metric-card {
        min-height: 158px;
        margin-bottom: 0.55rem;
        padding: 0.85rem 0.9rem;
        border: 1px solid #3a484f;
        border-radius: 6px;
        background: #1b262b;
        display: flex;
        flex-direction: column;
    }
    .summary-metric-label {
        color: #f4f7f8;
        font-size: 0.82rem;
        font-weight: 700;
        line-height: 1.25;
    }
    .summary-metric-value {
        margin-top: 0.45rem;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 500;
        line-height: 1.05;
        overflow-wrap: anywhere;
    }
    .summary-metric-description {
        margin-top: auto;
        padding-top: 0.7rem;
        color: #aebbc1;
        font-size: 0.76rem;
        line-height: 1.35;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    app_function = st.radio("Funcion", ["Patches", "Pipeline video"], index=0)

if app_function == "Pipeline video":
    render_video_pipeline()
    st.stop()

try:
    dataset, dataset_root_text = load_dataset_cached(
        str(dataset_path),
        dataset_cache_token(dataset_path),
    )
    dataset = dataset.copy()
    dataset_root = Path(dataset_root_text)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

if dataset.empty:
    st.warning("No se encontraron imagenes para etiquetar.")
    st.stop()

labels_path = resolve_dataset(args.labels_file) if args.labels_file else dataset_root / "labels.csv"
labels = load_labels_cached(str(labels_path), file_cache_token(labels_path)).copy()
label_by_id = dict(zip(labels["patch_id"].astype(str), labels["label"], strict=False))
dataset["current_label"] = dataset["patch_id"].astype(str).map(label_by_id).fillna("")
for label_column in ["confidence", "notes", "annotator", "labeled_at"]:
    label_meta = dict(zip(labels["patch_id"].astype(str), labels[label_column], strict=False))
    dataset[f"current_{label_column}"] = (
        dataset["patch_id"].astype(str).map(label_meta).fillna("")
    )

dataset_key = f"{dataset_path}|{labels_path}"
if st.session_state.get("dataset_key") != dataset_key:
    st.session_state.dataset_key = dataset_key
    st.session_state.position = 0
    st.session_state.review_page = 0
    st.session_state.review_index = 0
    st.session_state.confidence = "alta"
    st.session_state.notes = ""

with st.sidebar:
    st.title("Granulometria")
    st.caption(dataset_root.name)
    mode = st.radio("Modo", ["Etiquetado", "Revision"])
    if mode == "Etiquetado":
        view = st.selectbox(
            "Vista",
            ["Pendientes", "Todos", *[LABEL_TITLES[label] for label in LABELS]],
        )
    else:
        review_layout = st.radio("Vista de revision", ["Rapida", "Grilla"])
        review_label_filter = st.multiselect(
            "Clases a revisar",
            LABELS,
            default=LABELS,
            format_func=option_title,
        )
        review_confidence_filter = st.selectbox(
            "Confianza",
            ["todas", *CONFIDENCES],
            format_func=confidence_title,
        )
        review_only_labeled = st.toggle("Solo etiquetados", value=True)
        review_show_context = st.toggle("Mostrar imagen original", value=True)
        review_show_grid = st.toggle("Usar grilla numerada", value=False)
        review_auto_advance = (
            st.toggle("Avanzar al guardar", value=True)
            if review_layout == "Rapida"
            else False
        )
        review_search_text = st.text_input("Buscar patch/fuente/notas")
        review_page_size = (
            st.selectbox("Patches por pagina", [8, 12, 16, 24], index=1)
            if review_layout == "Grilla"
            else 12
        )
    show_edges = st.toggle("Mostrar Canny", value=False)
    st.text_input("Anotador", key="annotator")
    st.divider()
    counts = label_counts(labels)
    for label in LABELS:
        st.write(f"{LABEL_TITLES[label]}: **{counts[label]}**")
    st.caption(str(labels_path))

if mode == "Revision":
    render_review(
        dataset_root=dataset_root,
        dataset=dataset,
        labels=labels,
        labels_path=labels_path,
        show_edges=show_edges,
        label_filter=review_label_filter,
        confidence_filter=review_confidence_filter,
        only_labeled=review_only_labeled,
        search_text=review_search_text,
        page_size=int(review_page_size),
        show_context=review_show_context,
        show_grid_context=review_show_grid,
        review_layout=review_layout,
        auto_advance=review_auto_advance,
    )
    st.stop()

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
