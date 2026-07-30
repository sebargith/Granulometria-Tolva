from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "informe_final" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

TECH = (
    ROOT
    / "outputs"
    / "prueba_tecnica"
    / "prueba_tecnica_ampliada_cnn_batch003_revisado_20260717_084113"
)
RUN = (
    ROOT
    / "modelos granulometria"
    / "clasificador_patches"
    / "runs"
    / "resnet18_rgb_grid96_batch003_revisado_cuda_20260702"
)
EVAL_CSV = TECH / "evaluacion" / "evaluacion_por_imagen.csv"
SEG_STEM = "44775_2025-10-16_14-18-13.440363+00_7_0"
SEG_ORIGINAL = (
    ROOT / "data" / "evaluation" / "prueba_tecnica" / "Grueso" / f"{SEG_STEM}.jpg"
)
SEG_MASK = TECH / "masks_yolo_original" / "Grueso" / f"{SEG_STEM}_yolo_mask.png"
SEG_NORMALIZED = TECH / "normalized_inputs" / "Grueso" / f"{SEG_STEM}_normalized.jpg"
GRID_DIR = (
    ROOT
    / "data"
    / "labeling"
    / "batches"
    / "batch003_grilla96_data3_mixto_20260702"
    / "frames_a_etiquetar"
    / "44752_2025-10-16_12-53-21.812237+00_7_1"
)

BLUE = "#1f4e79"
DARK = "#1f2933"
MUTED = "#5f6b78"
LIGHT = "#f4f6f8"
BORDER = "#c9d1d9"
GREEN = "#45ad5a"
YELLOW = "#f0c419"
RED = "#e34a33"
CLASS_COLORS = {"fino": GREEN, "medio": YELLOW, "grueso": RED}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def save_rgb(image: Image.Image, path: Path, quality: int = 94) -> None:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, quality=quality)


def fit_image(path: Path, size: tuple[int, int], background="white") -> Image.Image:
    image = Image.open(path).convert("RGB")
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def labeled_panel(
    path: Path,
    label: str,
    size: tuple[int, int],
    label_color: str = DARK,
    background="white",
) -> Image.Image:
    panel = Image.new("RGB", (size[0], size[1] + 62), background)
    panel.paste(fit_image(path, size, background), (0, 62))
    draw = ImageDraw.Draw(panel)
    draw.text((18, 15), label, fill=label_color, font=font(28, True))
    return panel


def add_header(image: Image.Image, title: str, subtitle: str | None = None) -> Image.Image:
    header_h = 105 if subtitle else 78
    canvas = Image.new("RGB", (image.width, image.height + header_h), "white")
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((26, 18), title, fill=DARK, font=font(34, True))
    if subtitle:
        draw.text((28, 62), subtitle, fill=MUTED, font=font(22))
    return canvas


def read_evaluation() -> list[dict[str, str]]:
    with EVAL_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_eval(expected: str, predicted: str | None = None, correct: bool | None = None) -> dict[str, str]:
    for row in read_evaluation():
        if row["expected_class"].lower() != expected:
            continue
        if predicted is not None and row["predicted_class"].lower() != predicted:
            continue
        if correct is not None and (row["is_correct"].lower() == "true") != correct:
            continue
        return row
    raise RuntimeError(f"No evaluation row for expected={expected}, predicted={predicted}, correct={correct}")


def make_fig01() -> None:
    row = select_eval("medio", "medio", True)
    image = Image.open(row["comparison_path"]).convert("RGB")
    save_rgb(image, ASSETS / "fig01_salida_conceptual.jpg")


def rounded_box(ax, x, y, w, h, title, body, color, status):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.6,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.015, y + h - 0.095),
            0.12,
            0.055,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=0,
            facecolor=color,
        )
    )
    ax.text(x + 0.075, y + h - 0.067, status, ha="center", va="center", color="white", fontsize=9, weight="bold")
    ax.text(x + 0.02, y + h - 0.145, title, ha="left", va="top", color=DARK, fontsize=13, weight="bold")
    ax.text(x + 0.02, y + h - 0.215, body, ha="left", va="top", color=MUTED, fontsize=9.5, linespacing=1.35)


def make_fig02() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.94, "Alternativas evaluadas para granulometría visual", fontsize=20, weight="bold", color=DARK)
    ax.text(0.03, 0.885, "La solución actual prioriza clasificación supervisada sobre parches RGB normalizados.", fontsize=11, color=MUTED)
    boxes = [
        (0.03, "DexiNed", "Mapas de borde\ninternos y contornos.", "#6b7280", "Explorado"),
        (0.225, "OpenCV + RF", "Textura, bordes y\nclasificador clásico.", "#2f80ed", "Baseline"),
        (0.42, "CNN RGB", "ResNet18 aprende\npatrones visuales.", "#7b3fb2", "Actual"),
        (0.615, "Instancias", "Separación explícita\nde fragmentos.", "#d97706", "Alternativa"),
        (0.81, "Fusión 2D-3D", "Escala física y\ngeometría de carga.", "#00897b", "Futuro"),
    ]
    for x, title, body, color, status in boxes:
        rounded_box(ax, x, 0.28, 0.165, 0.5, title, body, color, status)
    ax.annotate("", xy=(0.58, 0.17), xytext=(0.44, 0.17), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
    ax.text(0.51, 0.12, "mejor equilibrio actual entre precisión, trazabilidad y costo", ha="center", color=BLUE, fontsize=10, weight="bold")
    fig.savefig(ASSETS / "fig02_comparacion_metodos.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fig03() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 9.0), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.94, "Arquitectura general del sistema", fontsize=22, weight="bold", color=DARK)
    ax.text(0.04, 0.89, "Flujo completo desde el video de entrada hasta los resultados revisables.", fontsize=11.5, color=MUTED)

    blocks = [
        (0.06, 0.69, 0.24, 0.16, "Video MP4", "Entrada visual", "#274c6b", "#eef2f5"),
        (0.38, 0.69, 0.24, 0.16, "Extracción de frames", "Muestreo temporal", "#2f80ed", "#eaf3fd"),
        (0.70, 0.69, 0.24, 0.16, "YOLO segmentación", "Aislamiento de la carga", "#43a047", "#edf7ed"),
        (0.70, 0.47, 0.24, 0.16, "Normalización y grilla", "896 px · parches 96 × 96", "#00897b", "#e9f7f4"),
        (0.38, 0.47, 0.24, 0.16, "ResNet18", "Fino · medio · grueso", "#7b3fb2", "#f5effb"),
        (0.06, 0.47, 0.24, 0.16, "Agregación", "Resultado por frame y video", "#e7662d", "#fff1e9"),
    ]
    for x, y, w, h, title, tag, edge, fill in blocks:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.8, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + 0.100, title, ha="center", va="center", fontsize=12.2, weight="bold", color="#17324d")
        ax.text(x + w / 2, y + 0.040, tag, ha="center", va="center", fontsize=9.4, color=MUTED)

    arrow = dict(arrowstyle="-|>", mutation_scale=17, lw=2.1, color="#17324d")
    ax.add_patch(FancyArrowPatch((0.30, 0.77), (0.38, 0.77), **arrow))
    ax.add_patch(FancyArrowPatch((0.62, 0.77), (0.70, 0.77), **arrow))
    ax.add_patch(FancyArrowPatch((0.82, 0.69), (0.82, 0.63), **arrow))
    ax.add_patch(FancyArrowPatch((0.70, 0.55), (0.62, 0.55), **arrow))
    ax.add_patch(FancyArrowPatch((0.38, 0.55), (0.30, 0.55), **arrow))

    web = FancyBboxPatch(
        (0.54, 0.20), 0.40, 0.16,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=1.8, edgecolor="#2f80ed", facecolor="#eef6ff",
    )
    ax.add_patch(web)
    ax.text(0.74, 0.305, "Aplicación web local", ha="center", va="center", fontsize=13.5, weight="bold", color="#17324d")
    ax.text(0.74, 0.250, "carga  •  configuración  •  ejecución  •  revisión", ha="center", va="center", fontsize=9.7, color=MUTED)

    results = FancyBboxPatch(
        (0.06, 0.20), 0.42, 0.16,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=1.8, edgecolor="#274c6b", facecolor="#f2f4f6",
    )
    ax.add_patch(results)
    ax.text(0.27, 0.305, "Resultados y trazabilidad", ha="center", va="center", fontsize=13.5, weight="bold", color="#17324d")
    ax.text(0.27, 0.250, "CSV  •  máscaras  •  overlays  •  resumen por video", ha="center", va="center", fontsize=9.5, color=MUTED)

    ax.add_patch(FancyArrowPatch((0.18, 0.47), (0.18, 0.36), **arrow))
    ax.add_patch(FancyArrowPatch((0.48, 0.28), (0.54, 0.28), arrowstyle="<|-|>", mutation_scale=15, lw=2.0, color="#2f80ed"))

    ax.text(0.06, 0.145, "Resultado granulométrico", fontsize=12, weight="bold", color=DARK)
    ax.text(
        0.06, 0.108,
        "porcentajes por clase  •  clase dominante  •  índice granulométrico  •  dispersión",
        fontsize=10.3, color=MUTED,
    )
    ax.text(0.06, 0.065, "Indicadores complementarios", fontsize=12, weight="bold", color=DARK)
    ax.text(
        0.06, 0.025,
        "confianza media  •  frames evaluables  •  tiempo de inferencia  •  trazabilidad por parche",
        fontsize=9.7, color=MUTED, va="center",
    )
    fig.savefig(ASSETS / "fig03_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_segmentation() -> tuple[Image.Image, Image.Image, Image.Image]:
    original = Image.open(SEG_ORIGINAL).convert("RGB")
    mask = Image.open(SEG_MASK).convert("L")
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.Resampling.NEAREST)
    mask = mask.point(lambda value: 255 if value >= 128 else 0)
    normalized = Image.open(SEG_NORMALIZED).convert("RGB")
    return original, mask, normalized


def make_fig04() -> None:
    original, mask, isolated = load_segmentation()
    temp_original = ASSETS / "_seg_original.jpg"
    temp_mask = ASSETS / "_seg_mask.png"
    temp_isolated = ASSETS / "_seg_isolated.jpg"
    original.save(temp_original, quality=95)
    mask.save(temp_mask)
    isolated.save(temp_isolated, quality=95)
    panels = [
        labeled_panel(temp_original, "Frame original", (560, 420)),
        labeled_panel(temp_mask, "Máscara de carga", (560, 420)),
        labeled_panel(temp_isolated, "Carga aislada", (560, 420)),
    ]
    canvas = Image.new("RGB", (1740, 482), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (idx * 580, 0))
    save_rgb(add_header(canvas, "Segmentación y aislamiento de la carga"), ASSETS / "fig04_segmentacion.jpg")
    for path in (temp_original, temp_mask, temp_isolated):
        path.unlink(missing_ok=True)


def make_fig05() -> None:
    original, _, normalized = load_segmentation()
    canvas = Image.new("RGB", (1700, 690), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 20), "Efecto de la normalización por ancho", fill=DARK, font=font(38, True))
    draw.text((35, 70), "La región detectada se recorta y reescala para que todas las cargas tengan una referencia visual comparable.", fill=MUTED, font=font(23))
    before = ImageOps.contain(original, (720, 500), Image.Resampling.LANCZOS)
    before_panel = Image.new("RGB", (720, 500), "#202020")
    before_panel.paste(before, ((720 - before.width) // 2, (500 - before.height) // 2))
    after = ImageOps.contain(normalized, (720, 500), Image.Resampling.LANCZOS)
    after_panel = Image.new("RGB", (720, 500), "black")
    after_panel.paste(after, ((720 - after.width) // 2, (500 - after.height) // 2))
    canvas.paste(before_panel, (35, 145))
    canvas.paste(after_panel, (945, 145))
    draw.text((35, 110), "Antes: frame original", fill=DARK, font=font(25, True))
    draw.text((945, 110), "Después: carga aislada a 896 px", fill=BLUE, font=font(25, True))
    draw.line((785, 385, 895, 385), fill=BLUE, width=8)
    draw.polygon(((895, 385), (865, 365), (865, 405)), fill=BLUE)
    save_rgb(canvas, ASSETS / "fig05_normalizacion.jpg")


def draw_grid(image: Image.Image, patch: int, width: int = 896) -> Image.Image:
    fitted = ImageOps.contain(image, (width, 500), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, 500), "black")
    x0 = (width - fitted.width) // 2
    y0 = (500 - fitted.height) // 2
    canvas.paste(fitted, (x0, y0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    for x in range(0, width + 1, patch):
        draw.line((x, 0, x, 500), fill=(255, 196, 0, 230), width=3)
    for y in range(0, 501, patch):
        draw.line((0, y, width, y), fill=(255, 196, 0, 230), width=3)
    return canvas


def make_fig06() -> None:
    _, _, load = load_segmentation()
    panels = []
    for patch in (64, 96, 128):
        image = draw_grid(load, patch)
        panel = Image.new("RGB", (896, 565), "white")
        panel.paste(image, (0, 65))
        draw = ImageDraw.Draw(panel)
        draw.text((20, 14), f"Parche {patch} × {patch} px", fill=DARK, font=font(30, True))
        draw.text((620, 20), f"≈ {896 // patch} columnas", fill=MUTED, font=font(21))
        panels.append(panel)
    canvas = Image.new("RGB", (2688, 565), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (idx * 896, 0))
    save_rgb(add_header(canvas, "Comparación de escalas de parche"), ASSETS / "fig06_grid_comparison.jpg")


def make_fig07() -> None:
    refs = [
        (
            "FINO",
            ROOT / "data" / "labeling" / "references" / "fino"
            / "45984_2025-10-22_13-52-51.237508+00_8_0.jpg",
            GREEN,
        ),
        (
            "MEDIO",
            ROOT / "data" / "labeling" / "references" / "medio"
            / "46073_2025-10-22_20-20-51.384809+00_8_0.jpg",
            YELLOW,
        ),
        (
            "GRUESO",
            ROOT / "data" / "labeling" / "references" / "grueso"
            / "46175_2025-10-23_15-19-34.287299+00_7_2.jpg",
            RED,
        ),
    ]
    canvas = Image.new("RGB", (1830, 590), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, path, color) in enumerate(refs):
        x = 20 + idx * 610
        image = ImageOps.fit(Image.open(path).convert("RGB"), (570, 430), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, 90))
        draw.rectangle((x, 90, x + 570, 520), outline=color, width=7)
        label_font = font(36, True)
        draw.text((x, 25), label, fill=color, font=label_font)
        subtitle = {"FINO": "textura más cerrada", "MEDIO": "mezcla intermedia", "GRUESO": "fragmentos dominantes"}[label]
        label_box = draw.textbbox((x, 25), label, font=label_font)
        subtitle_x = label_box[2] + 24
        draw.text((subtitle_x, 38), subtitle, fill=MUTED, font=font(21))
    save_rgb(add_header(canvas, "Referencias visuales de las clases"), ASSETS / "fig07_referencias.jpg")


def make_fig08() -> None:
    grid = fit_image(GRID_DIR / "02_grilla_numerada.jpg", (1050, 700), "#111111")
    canvas = Image.new("RGB", (1700, 820), "white")
    canvas.paste(grid, (30, 95))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), "Etiquetado con contexto completo y salida trazable", fill=DARK, font=font(38, True))
    x = 1120
    draw.rounded_rectangle((x, 110, 1665, 720), radius=14, fill="#f6f8fa", outline=BORDER, width=2)
    draw.text((x + 28, 140), "cells.csv", fill=BLUE, font=font(30, True))
    draw.text((x + 28, 195), "Una fila por celda de la grilla", fill=MUTED, font=font(21))
    csv_path = GRID_DIR / "cells.csv"
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader):
            rows.append(row)
            if idx >= 6:
                break
    y = 255
    for idx, row in enumerate(rows):
        values = " | ".join(str(value)[:16] for value in row[:6])
        color = DARK if idx == 0 else "#44515d"
        draw.text((x + 28, y), values, fill=color, font=font(18, idx == 0))
        y += 52
    draw.text((x + 28, 640), "frame • fila • columna • clase\nconfianza • notas", fill=BLUE, font=font(21, True), spacing=8)
    save_rgb(canvas, ASSETS / "fig08_grilla_csv.jpg")


def make_fig09() -> None:
    normalized = Image.open(GRID_DIR / "01_carga_normalizada.jpg").convert("RGB")
    original_candidates = list(
        (ROOT / "data" / "datasets" / "patch_frames" / "batch003").rglob(
            f"{GRID_DIR.name}.jpg"
        )
    )
    original = Image.open(original_candidates[0]).convert("RGB") if original_candidates else normalized
    patch = ImageOps.fit(normalized, (410, 410), Image.Resampling.LANCZOS)
    original_fit = ImageOps.contain(original, (760, 500), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1700, 900), "#10181c")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 300, 900), fill="#17242a")
    draw.text((25, 35), "Granulometría", fill="white", font=font(31, True))
    draw.text((25, 90), "Revisión de parches", fill="#aeb9c2", font=font(20))
    for idx, text in enumerate(("Clases a revisar", "☑ Fino", "☑ Medio", "☑ Grueso", "Confianza: todas", "Solo etiquetados", "Mostrar original")):
        draw.text((25, 165 + idx * 58), text, fill="white" if idx else "#94a4af", font=font(19, idx == 0))
    draw.text((350, 40), "Revisión rápida de patches", fill="white", font=font(42, True))
    draw.text((350, 100), "Patch 326 de 2455", fill="#aeb9c2", font=font(21))
    draw.text((350, 155), "Patch", fill="white", font=font(28, True))
    canvas.paste(patch, (350, 205))
    draw.text((350, 635), "Actual: Medio", fill=YELLOW, font=font(24, True))
    draw.text((350, 680), "Confianza: alta", fill=GREEN, font=font(21, True))
    draw.text((810, 155), "Imagen original", fill="white", font=font(28, True))
    original_panel = Image.new("RGB", (830, 500), "black")
    original_panel.paste(original_fit, ((830 - original_fit.width) // 2, (500 - original_fit.height) // 2))
    canvas.paste(original_panel, (810, 205))
    draw.rectangle((1110, 350, 1210, 450), outline="#2cc7b5", width=5)
    buttons = [("Fino", GREEN), ("Medio", YELLOW), ("Grueso", RED), ("Ambiguo", "#697986"), ("Inválido", "#697986")]
    for idx, (label, color) in enumerate(buttons):
        x = 350 + idx * 255
        draw.rounded_rectangle((x, 760, x + 225, 825), radius=8, fill=color, outline=color)
        text_fill = DARK if label == "Medio" else "white"
        draw.text((x + 65, 778), label, fill=text_fill, font=font(22, True))
    save_rgb(canvas, ASSETS / "fig09_revision_ui.jpg")


def make_fig10() -> None:
    base = "Camera_8_20241024_104003_frame000000_mask_0"
    original = (
        ROOT
        / "data"
        / "legacy"
        / "data_filtrada"
        / "Val"
        / "Camera_8_20241024_104003"
        / "Camera_8_20241024_104003_frame000000.jpg"
    )
    dex = ROOT / "outputs" / "dexined_edges" / "dexined_masked_crops_v3"
    paths = [
        (original, "Entrada"),
        (dex / "edge_maps" / "masked_crops" / f"{base}_edge.png", "Mapa continuo"),
        (dex / "binary_edges" / "masked_crops" / f"{base}_edge_bin.png", "Borde binario"),
        (dex / "overlays" / "masked_crops" / f"{base}_edge_overlay.jpg", "Superposición"),
    ]
    if not all(path.is_file() for path, _ in paths):
        if (ASSETS / "fig10_dexined.jpg").is_file():
            return
        raise FileNotFoundError("No estan disponibles los artefactos historicos de DexiNed.")
    panels = [labeled_panel(path, label, (430, 430)) for path, label in paths]
    canvas = Image.new("RGB", (1760, 492), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (idx * 440, 0))
    save_rgb(add_header(canvas, "Resultado representativo de DexiNed", "Los bordes internos resultaron densos y con contornos de fragmentos incompletos."), ASSETS / "fig10_dexined.jpg")


def make_fig11() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=170)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Clasificador convolucional ResNet18", fontsize=21, weight="bold", color=DARK)
    nodes = [
        (0.04, 0.35, 0.15, 0.32, "Parche RGB", "96 × 96 × 3", BLUE),
        (0.25, 0.30, 0.18, 0.42, "ResNet18", "Transfer learning\nImageNet", "#7b3fb2"),
        (0.49, 0.35, 0.16, 0.32, "Características", "Textura, formas\ny contraste", "#2f80ed"),
        (0.71, 0.35, 0.12, 0.32, "Capa final", "3 neuronas", "#d97706"),
    ]
    for x, y, w, h, title, body, color in nodes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02", edgecolor=color, facecolor="white", linewidth=2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", color=DARK, fontsize=13, weight="bold")
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", color=MUTED, fontsize=10)
    for x1, x2 in ((0.19, 0.25), (0.43, 0.49), (0.65, 0.71), (0.83, 0.865)):
        ax.add_patch(FancyArrowPatch((x1, 0.51), (x2, 0.51), arrowstyle="-|>", mutation_scale=15, lw=1.8, color="#7a8793"))
    for idx, (label, color, value) in enumerate((("Fino", GREEN, "p₁"), ("Medio", YELLOW, "p₂"), ("Grueso", RED, "p₃"))):
        y = 0.66 - idx * 0.16
        ax.add_patch(FancyBboxPatch((0.875, y), 0.095, 0.11, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor=color, edgecolor=color))
        ax.text(0.922, y + 0.055, f"{label}  {value}", ha="center", va="center", fontsize=10.5, weight="bold", color="white" if label != "Medio" else DARK)
    ax.text(0.04, 0.17, "Entrenamiento: cabeza congelada → ajuste fino suave del backbone", fontsize=11.5, color=BLUE, weight="bold")
    fig.savefig(ASSETS / "fig11_resnet_schematic.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fig12() -> None:
    history = pd.read_csv(RUN / "training_history.csv")
    epoch = history["epoch"]
    best_idx = int(history["val_f1_macro"].idxmax())
    best_epoch = int(history.loc[best_idx, "epoch"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=180)
    charts = [
        ("Pérdida", "train_loss", "val_loss", "Pérdida"),
        ("Exactitud", "train_accuracy", "val_accuracy", "Proporción"),
        ("F1 macro", "train_f1_macro", "val_f1_macro", "F1"),
    ]
    for ax, (title, train_col, val_col, ylabel) in zip(axes, charts):
        ax.plot(epoch, history[train_col], color=BLUE, lw=2.2, marker="o", ms=3, label="Entrenamiento")
        ax.plot(epoch, history[val_col], color="#7b3fb2", lw=2.2, marker="o", ms=3, label="Validación")
        ax.axvline(best_epoch, color="#d97706", ls="--", lw=1.5)
        ax.set_title(title, weight="bold", color=DARK)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"Evolución del entrenamiento de ResNet18 · mejor época: {best_epoch}", fontsize=17, weight="bold", color=DARK)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig12_training_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def copy_assets() -> None:
    shutil.copy2(RUN / "confusion_matrix_test.png", ASSETS / "fig13_confusion_matrix.png")
    correct = select_eval("medio", "medio", True)
    shutil.copy2(correct["overlay_path"], ASSETS / "fig14_frame_overlay.jpg")
    shutil.copy2(TECH / "evaluacion" / "reporte_prueba_tecnica.png", ASSETS / "fig16_prueba_tecnica.png")


def make_fig17() -> None:
    margin = np.array([0.00, 0.10, 0.15, 0.20, 0.25, 0.30])
    coverage = np.array([100.0, 76.7, 64.4, 60.0, 52.2, 44.4])
    accuracy = np.array([76.7, 79.7, 87.9, 92.6, 95.7, 97.5])
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=180)
    ax.plot(coverage, accuracy, color=BLUE, lw=3, marker="o", ms=8)
    for x, y, m in zip(coverage, accuracy, margin):
        ax.annotate(f"margen {m:.2f}", (x, y), xytext=(7, 8), textcoords="offset points", fontsize=9, color=MUTED)
    ax.set_xlabel("Cobertura de imágenes (%)", fontsize=11)
    ax.set_ylabel("Exactitud entre imágenes aceptadas (%)", fontsize=11)
    ax.set_title("Compromiso entre cobertura y exactitud", fontsize=17, weight="bold", color=DARK)
    ax.set_xlim(40, 103)
    ax.set_ylim(72, 100)
    ax.grid(alpha=0.25)
    ax.text(0.02, -0.18, "Al exigir mayor separación entre las dos clases más probables, aumenta la exactitud y disminuye la cobertura.", transform=ax.transAxes, fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig17_coverage_accuracy.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fig18() -> None:
    examples = [
        (select_eval("medio", "fino", False), "Medio → fino"),
        (select_eval("medio", "grueso", False), "Medio → grueso"),
    ]
    canvas = Image.new("RGB", (1660, 720), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), "Errores representativos entre clases adyacentes", fill=DARK, font=font(38, True))
    draw.text((30, 70), "La clase medio concentra la mayor parte de las confusiones de la prueba técnica.", fill=MUTED, font=font(23))
    for idx, (row, label) in enumerate(examples):
        image = fit_image(Path(row["comparison_path"]), (790, 560), "white")
        x = 25 + idx * 820
        canvas.paste(image, (x, 125))
        color = GREEN if "fino" in label else RED
        draw.rectangle((x, 125, x + 790, 685), outline=color, width=5)
        draw.rounded_rectangle((x + 20, 140, x + 250, 195), radius=10, fill=color)
        draw.text((x + 42, 153), label, fill="white", font=font(24, True))
    save_rgb(canvas, ASSETS / "fig18_errores.jpg")


def make_fig19() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.94, "Arquitectura general del sistema", fontsize=22, weight="bold", color=DARK)
    ax.text(0.04, 0.89, "Flujo desde el video de entrada hasta los resultados revisables.", fontsize=11.5, color=MUTED)

    blocks = [
        (0.04, 0.59, 0.145, 0.20, "Video\nMP4", "Entrada", "#274c6b", "#eef2f5"),
        (0.225, 0.59, 0.155, 0.20, "Extracción\nde frames", "Proceso local", "#2f80ed", "#eaf3fd"),
        (0.42, 0.59, 0.155, 0.20, "YOLO\nsegmentación", "Modelo de percepción", "#43a047", "#edf7ed"),
        (0.615, 0.59, 0.155, 0.20, "Normalización\ny grilla 96 px", "Preparación", "#00897b", "#e9f7f4"),
        (0.81, 0.59, 0.15, 0.20, "ResNet18\ny agregación", "Clasificación", "#e7662d", "#fff1e9"),
    ]
    for x, y, w, h, title, tag, edge, fill in blocks:
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.8, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + 0.115, title, ha="center", va="center", fontsize=12, weight="bold", color="#17324d", linespacing=1.15)
        ax.text(x + w / 2, y + 0.035, tag, ha="center", va="center", fontsize=8.8, color=MUTED)

    arrow = dict(arrowstyle="-|>", mutation_scale=16, lw=2.0, color="#17324d")
    for x1, x2 in ((0.185, 0.225), (0.38, 0.42), (0.575, 0.615), (0.77, 0.81)):
        ax.add_patch(FancyArrowPatch((x1, 0.69), (x2 - 0.008, 0.69), **arrow))

    web = FancyBboxPatch(
        (0.18, 0.25), 0.31, 0.17,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=1.8, edgecolor="#2f80ed", facecolor="#eef6ff",
    )
    ax.add_patch(web)
    ax.text(0.335, 0.36, "Aplicación web local", ha="center", va="center", fontsize=13, weight="bold", color="#17324d")
    ax.text(0.335, 0.305, "carga  •  configuración  •  ejecución  •  revisión", ha="center", va="center", fontsize=9.5, color=MUTED)

    results = FancyBboxPatch(
        (0.60, 0.25), 0.36, 0.17,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        linewidth=1.8, edgecolor="#274c6b", facecolor="#f2f4f6",
    )
    ax.add_patch(results)
    ax.text(0.78, 0.36, "Resultados y trazabilidad", ha="center", va="center", fontsize=13, weight="bold", color="#17324d")
    ax.text(0.78, 0.305, "CSV  •  máscaras  •  overlays  •  resumen por video", ha="center", va="center", fontsize=9.5, color=MUTED)

    ax.add_patch(FancyArrowPatch((0.49, 0.335), (0.60, 0.335), **arrow))
    ax.add_patch(FancyArrowPatch((0.885, 0.59), (0.885, 0.42), **arrow))

    ax.text(0.04, 0.12, "Procesamiento local", fontsize=9.5, color="#2f80ed", weight="bold")
    ax.text(0.205, 0.12, "Modelos de visión", fontsize=9.5, color="#43a047", weight="bold")
    ax.text(0.355, 0.12, "Clasificación y agregación", fontsize=9.5, color="#e7662d", weight="bold")
    ax.text(0.58, 0.12, "Artefactos auditables", fontsize=9.5, color="#274c6b", weight="bold")
    fig.savefig(ASSETS / "fig19_arquitectura_sistema.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fig20() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.035, 0.95, "Secuencia de ejecución del pipeline", fontsize=22, weight="bold", color=DARK)
    ax.text(0.035, 0.905, "Interacción entre el usuario, la aplicación local, los modelos y los archivos de salida.", fontsize=11.5, color=MUTED)

    actors = [
        (0.07, "Usuario", "#274c6b", "#eef2f5"),
        (0.235, "Web app", "#2f80ed", "#eaf3fd"),
        (0.405, "Orquestador\nlocal", "#274c6b", "#eef2f5"),
        (0.575, "YOLO-seg", "#43a047", "#edf7ed"),
        (0.755, "Analítico\ngranulométrico", "#e7662d", "#fff1e9"),
        (0.925, "Resultados", "#5f6b78", "#f2f4f6"),
    ]
    for x, label, edge, fill in actors:
        width = 0.145 if "Analítico" not in label else 0.165
        box = FancyBboxPatch(
            (x - width / 2, 0.79), width, 0.085,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.6, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x, 0.832, label, ha="center", va="center", fontsize=10.5, weight="bold", color="#17324d", linespacing=1.0)
        ax.plot([x, x], [0.79, 0.10], color="#aeb8c2", lw=1.2, zorder=0)

    def sequence_arrow(x1, x2, y, label, color="#17324d"):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=14, lw=1.8, color=color))
        ax.text((x1 + x2) / 2, y + 0.022, label, ha="center", va="bottom", fontsize=9.5, weight="bold", color=DARK)

    sequence_arrow(0.07, 0.235, 0.72, "1. Carga el video")
    sequence_arrow(0.235, 0.405, 0.64, "2. Solicita la corrida")

    ax.plot([0.405, 0.465], [0.57, 0.57], color="#2f80ed", lw=1.8)
    ax.plot([0.465, 0.465], [0.57, 0.535], color="#2f80ed", lw=1.8)
    ax.add_patch(FancyArrowPatch((0.465, 0.535), (0.405, 0.535), arrowstyle="-|>", mutation_scale=14, lw=1.8, color="#2f80ed"))
    ax.text(0.435, 0.59, "3. Extrae frames", ha="center", va="bottom", fontsize=9.5, weight="bold", color=DARK)

    sequence_arrow(0.405, 0.575, 0.48, "4. Ejecuta segmentación")
    sequence_arrow(0.575, 0.405, 0.40, "5. Devuelve máscaras", color="#2f80ed")
    sequence_arrow(0.405, 0.755, 0.32, "6. Normaliza, genera parches y clasifica")
    sequence_arrow(0.755, 0.925, 0.24, "7. CSV, overlays y resumen")
    sequence_arrow(0.925, 0.235, 0.16, "8. Métricas y galería", color="#2f80ed")
    sequence_arrow(0.235, 0.07, 0.10, "9. Revisión y descarga", color="#2f80ed")

    fig.savefig(ASSETS / "fig20_secuencia_ejecucion.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    make_fig01()
    make_fig02()
    make_fig03()
    make_fig04()
    make_fig05()
    make_fig06()
    make_fig07()
    make_fig08()
    make_fig09()
    make_fig10()
    make_fig11()
    make_fig12()
    copy_assets()
    make_fig17()
    make_fig18()
    make_fig19()
    make_fig20()
    print(f"Assets creados en: {ASSETS}")
    for path in sorted(ASSETS.glob("fig*")):
        print(path.name)


if __name__ == "__main__":
    main()
