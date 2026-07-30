from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from granulometria.metrics import compute_granulometry_metrics


EXPECTED_CLASSES = ("fino", "medio", "grueso")
PREDICTED_CLASSES = (*EXPECTED_CLASSES, "no_evaluable")
DISPLAY_NAMES = {
    "fino": "Fino",
    "medio": "Medio",
    "grueso": "Grueso",
    "no_evaluable": "No evaluable",
}
OUTPUT_NAMES = {
    "fino": "Fino",
    "medio": "Media",
    "grueso": "Grueso",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalua una prueba tecnica de granulometria usando las carpetas "
            "Fino, Media y Grueso como referencia visual esperada."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Corrida generada por predict_granulometria_tolva.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reemplaza resultados_ordenados y evaluacion si ya existen.",
    )
    parser.add_argument(
        "--training-split-csv",
        type=Path,
        default=None,
        help="Split del entrenamiento para comprobar coincidencias exactas de frames.",
    )
    return parser.parse_args()


def normalize_class(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "fina": "fino",
        "fino": "fino",
        "media": "medio",
        "medio": "medio",
        "gruesa": "grueso",
        "grueso": "grueso",
    }
    return aliases.get(text, "no_evaluable")


def expected_class_from_path(path: Path) -> str:
    for part in reversed(path.parts[:-1]):
        normalized = normalize_class(part)
        if normalized in EXPECTED_CLASSES:
            return normalized
    raise ValueError(f"No se encontro clase esperada en la ruta: {path}")


def bool_from_csv(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "si", "yes"}


def int_from_csv(value: Any) -> int:
    try:
        return int(float(str(value or "0").strip()))
    except ValueError:
        return 0


def float_from_csv(value: Any) -> float:
    try:
        return float(str(value or "0").strip())
    except ValueError:
        return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"El directorio ya existe: {path}. Usa --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return copy


def make_comparison(
    original_path: Path,
    overlay_path: Path | None,
    target_path: Path,
    expected: str,
    predicted: str,
    dominant_share: float,
) -> None:
    canvas_width, canvas_height = 1600, 560
    header_height = 70
    canvas = Image.new("RGB", (canvas_width, canvas_height), (246, 247, 249))
    draw = ImageDraw.Draw(canvas)
    title_font = font(27, bold=True)
    label_font = font(20)

    original = Image.open(original_path).convert("RGB")
    overlay = Image.open(overlay_path).convert("RGB") if overlay_path and overlay_path.exists() else None
    left = fit_image(original, 760, 450)
    right = fit_image(overlay, 760, 450) if overlay else None

    draw.text((28, 20), f"Esperado: {DISPLAY_NAMES[expected]}", font=title_font, fill=(30, 34, 40))
    result_color = (44, 135, 74) if expected == predicted else (204, 62, 62)
    draw.text(
        (820, 20),
        f"Prediccion: {DISPLAY_NAMES[predicted]} ({dominant_share:.0%})",
        font=title_font,
        fill=result_color,
    )

    left_x = 20 + (760 - left.width) // 2
    left_y = header_height + (450 - left.height) // 2
    canvas.paste(left, (left_x, left_y))
    draw.text((28, 530), "Frame original", font=label_font, fill=(85, 90, 98))

    if right is not None:
        right_x = 820 + (760 - right.width) // 2
        right_y = header_height + (450 - right.height) // 2
        canvas.paste(right, (right_x, right_y))
        draw.text((828, 530), "Resultado del pipeline", font=label_font, fill=(85, 90, 98))
    else:
        draw.text((1030, 280), "Sin overlay", font=title_font, fill=(150, 80, 80))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target_path, quality=94)


def organize_results(
    rows: list[dict[str, str]], output_dir: Path
) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        original_path = Path(row["image_path"])
        expected = expected_class_from_path(original_path)
        predicted = normalize_class(row.get("dominant_label"))
        has_mask = bool_from_csv(row.get("has_mask"))
        patch_count = int_from_csv(row.get("patch_count"))
        is_evaluable = has_mask and patch_count > 0 and predicted in EXPECTED_CLASSES
        if not is_evaluable:
            predicted = "no_evaluable"
        is_correct = is_evaluable and predicted == expected

        class_dir = output_dir / OUTPUT_NAMES[expected]
        originals_dir = class_dir / "imagenes_originales"
        overlays_dir = class_dir / "resultados_pipeline"
        comparisons_dir = class_dir / "comparaciones"
        originals_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        comparisons_dir.mkdir(parents=True, exist_ok=True)

        original_target = originals_dir / original_path.name
        if original_path.exists():
            shutil.copy2(original_path, original_target)

        overlay_raw = str(row.get("granulometry_overlay_path", "")).strip()
        overlay_path = Path(overlay_raw) if overlay_raw else None
        overlay_target = overlays_dir / f"{original_path.stem}_resultado.jpg"
        if overlay_path is not None and overlay_path.exists():
            shutil.copy2(overlay_path, overlay_target)

        comparison_target = comparisons_dir / f"{original_path.stem}_comparacion.jpg"
        if original_target.exists():
            make_comparison(
                original_path=original_target,
                overlay_path=overlay_target if overlay_target.exists() else None,
                target_path=comparison_target,
                expected=expected,
                predicted=predicted,
                dominant_share=float_from_csv(row.get("dominant_share")),
            )

        shares = {
            "fino": float_from_csv(row.get("share_fino")),
            "medio": float_from_csv(row.get("share_medio")),
            "grueso": float_from_csv(row.get("share_grueso")),
        }
        calculated_metrics = compute_granulometry_metrics(shares)
        index_raw = str(row.get("indice_granulometrico", "")).strip()
        dispersion_raw = str(row.get("dispersion_granulometrica", "")).strip()
        evaluated.append(
            {
                "image_name": original_path.name,
                "original_image_path": str(original_path),
                "expected_class": expected,
                "predicted_class": predicted,
                "is_evaluable": is_evaluable,
                "is_correct": is_correct,
                "patch_count": patch_count,
                "share_fino": shares["fino"],
                "share_medio": shares["medio"],
                "share_grueso": shares["grueso"],
                "indice_granulometrico": (
                    float_from_csv(index_raw)
                    if index_raw
                    else calculated_metrics["indice_granulometrico"]
                ),
                "dispersion_granulometrica": (
                    float_from_csv(dispersion_raw)
                    if dispersion_raw
                    else calculated_metrics["dispersion_granulometrica"]
                ),
                "dominant_share": float_from_csv(row.get("dominant_share")),
                "dominant_margin": float_from_csv(row.get("dominant_margin")),
                "mean_patch_confidence": float_from_csv(row.get("mean_patch_confidence")),
                "max_yolo_conf": float_from_csv(row.get("max_yolo_conf")),
                "comparison_path": str(comparison_target) if comparison_target.exists() else "",
                "overlay_path": str(overlay_target) if overlay_target.exists() else "",
            }
        )
    return evaluated


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    evaluable = sum(bool(row["is_evaluable"]) for row in rows)
    correct = sum(bool(row["is_correct"]) for row in rows)
    confusion = {
        expected: {
            predicted: sum(
                1
                for row in rows
                if row["expected_class"] == expected and row["predicted_class"] == predicted
            )
            for predicted in PREDICTED_CLASSES
        }
        for expected in EXPECTED_CLASSES
    }

    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in EXPECTED_CLASSES:
        selected = [row for row in rows if row["expected_class"] == label]
        selected_evaluable = [row for row in selected if row["is_evaluable"]]
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in EXPECTED_CLASSES if other != label)
        false_negative = len(selected) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "total": len(selected),
            "evaluable": len(selected_evaluable),
            "correct": true_positive,
            "accuracy_total": true_positive / len(selected) if selected else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_patch_confidence": (
                sum(float(row["mean_patch_confidence"]) for row in selected_evaluable)
                / len(selected_evaluable)
                if selected_evaluable
                else 0.0
            ),
            "mean_indice_granulometrico": (
                sum(float(row["indice_granulometrico"]) for row in selected_evaluable)
                / len(selected_evaluable)
                if selected_evaluable
                else 0.0
            ),
            "mean_dispersion_granulometrica": (
                sum(float(row["dispersion_granulometrica"]) for row in selected_evaluable)
                / len(selected_evaluable)
                if selected_evaluable
                else 0.0
            ),
            "mean_shares": {
                share_label: (
                    sum(float(row[f"share_{share_label}"]) for row in selected) / len(selected)
                    if selected
                    else 0.0
                )
                for share_label in EXPECTED_CLASSES
            },
        }

    return {
        "total": total,
        "evaluable": evaluable,
        "no_evaluable": total - evaluable,
        "correct": correct,
        "evaluable_ratio": evaluable / total if total else 0.0,
        "accuracy_total": correct / total if total else 0.0,
        "accuracy_evaluable": correct / evaluable if evaluable else 0.0,
        "f1_macro": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "mean_patch_confidence": (
            sum(float(row["mean_patch_confidence"]) for row in rows if row["is_evaluable"])
            / evaluable
            if evaluable
            else 0.0
        ),
        "mean_indice_granulometrico": (
            sum(float(row["indice_granulometrico"]) for row in rows if row["is_evaluable"])
            / evaluable
            if evaluable
            else 0.0
        ),
        "mean_dispersion_granulometrica": (
            sum(float(row["dispersion_granulometrica"]) for row in rows if row["is_evaluable"])
            / evaluable
            if evaluable
            else 0.0
        ),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "prediction_counts": dict(Counter(row["predicted_class"] for row in rows)),
    }


def build_margin_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analysis: list[dict[str, Any]] = []
    for threshold in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3):
        selected = [row for row in rows if float(row["dominant_margin"]) >= threshold]
        correct = sum(bool(row["is_correct"]) for row in selected)
        analysis.append(
            {
                "margen_minimo": threshold,
                "frames_aceptados": len(selected),
                "cobertura": len(selected) / len(rows) if rows else 0.0,
                "correctos": correct,
                "accuracy": correct / len(selected) if selected else 0.0,
            }
        )
    return analysis


def training_overlap(rows: list[dict[str, Any]], split_csv: Path | None) -> dict[str, Any]:
    if split_csv is None:
        return {"checked": False, "training_split_csv": "", "matching_frames": []}
    split_csv = split_csv.resolve()
    if not split_csv.is_file():
        raise FileNotFoundError(f"No existe training-split-csv: {split_csv}")
    training_rows = read_csv(split_csv)
    training_frames = {
        Path(str(row.get("source_image") or row.get("group") or "")).name.lower()
        for row in training_rows
        if str(row.get("source_image") or row.get("group") or "").strip()
    }
    test_frames = {str(row["image_name"]).lower() for row in rows}
    matches = sorted(test_frames & training_frames)
    return {
        "checked": True,
        "training_split_csv": str(split_csv),
        "matching_frames": matches,
        "matching_count": len(matches),
    }


def write_analysis_report(
    path: Path,
    metrics: dict[str, Any],
    margin_rows: list[dict[str, Any]],
    overlap: dict[str, Any],
) -> None:
    errors = metrics["total"] - metrics["correct"]
    severe_errors = (
        metrics["confusion_matrix"]["fino"]["grueso"]
        + metrics["confusion_matrix"]["grueso"]["fino"]
    )
    adjacent_errors = errors - metrics["no_evaluable"] - severe_errors
    lines = [
        "# Analisis de la prueba tecnica de granulometria",
        "",
        "## Resultado general",
        "",
        f"- Frames: {metrics['total']} ({metrics['evaluable']} evaluables).",
        f"- Acierto a nivel de frame: {metrics['correct']}/{metrics['total']} "
        f"({metrics['accuracy_total']:.1%}).",
        f"- F1 macro: {metrics['f1_macro']:.3f}.",
        f"- Confianza media de los patches: {metrics['mean_patch_confidence']:.1%}.",
        f"- Indice granulometrico visual medio: "
        f"{metrics['mean_indice_granulometrico']:.3f}.",
        f"- Dispersion granulometrica media: "
        f"{metrics['mean_dispersion_granulometrica']:.3f}.",
        "",
        "## Resultado por clase",
        "",
    ]
    for label in EXPECTED_CLASSES:
        item = metrics["per_class"][label]
        shares = item["mean_shares"]
        lines.append(
            f"- {DISPLAY_NAMES[label]}: {item['correct']}/{item['total']} correctas "
            f"({item['accuracy_total']:.1%}); precision {item['precision']:.1%}; "
            f"recall {item['recall']:.1%}; F1 {item['f1']:.3f}. Promedio de patches: "
            f"fino {shares['fino']:.1%}, medio {shares['medio']:.1%}, "
            f"grueso {shares['grueso']:.1%}; IG "
            f"{item['mean_indice_granulometrico']:.3f}; dispersion "
            f"{item['mean_dispersion_granulometrica']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Lectura tecnica",
            "",
            f"- {adjacent_errors} de {errors} errores fueron entre clases vecinas "
            "(fino-medio o medio-grueso).",
            f"- Hubo {severe_errors} error(es) extremo(s) entre fino y grueso.",
            "- La clase medio es la mas dificil: su composicion promedio queda repartida "
            "entre las tres salidas, por lo que el voto dominante tiene poco margen.",
            "- El margen entre la primera y segunda clase es un mejor indicador de duda "
            "que la confianza individual de los patches.",
            "",
            "## Politica exploratoria por margen",
            "",
            "| Margen minimo | Frames aceptados | Cobertura | Acierto aceptados |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in margin_rows:
        lines.append(
            f"| {row['margen_minimo']:.2f} | {row['frames_aceptados']} | "
            f"{row['cobertura']:.1%} | {row['accuracy']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"Estos umbrales son exploratorios. Con solo {metrics['total']} frames no deben fijarse como "
            "regla operacional; sirven para orientar una futura salida `incierto/mixto` y "
            "la agregacion de varios frames del mismo camion.",
            "",
            "## Independencia respecto al entrenamiento",
            "",
        ]
    )
    if overlap.get("checked"):
        count = int(overlap.get("matching_count", 0))
        lines.append(
            f"Se compararon los nombres de frame contra el split de entrenamiento y se "
            f"encontraron {count} coincidencias exactas."
        )
        if count:
            lines.extend(["", *[f"- {name}" for name in overlap["matching_frames"]]])
    else:
        lines.append("No se entrego un split de entrenamiento para comprobar coincidencias.")
    lines.extend(
        [
            "",
            "## Alcance",
            "",
            "La referencia de esta prueba es una seleccion visual de la clase mayoritaria "
            "del frame. Por lo tanto, las metricas miden concordancia con ese criterio y no "
            "una distribucion granulometrica fisica validada por harnero o medicion en mm.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((xy[0] - width // 2, xy[1] - height // 2), text, font=text_font, fill=fill)


def draw_report(metrics: dict[str, Any], output_path: Path) -> None:
    width, height = 1600, 1250
    background = (248, 249, 250)
    white = (255, 255, 255)
    dark = (31, 35, 40)
    muted = (92, 99, 108)
    border = (216, 220, 225)
    green = (61, 145, 72)
    pale_green = (205, 237, 211)
    pale_red = (249, 221, 224)
    blue = (46, 111, 190)

    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    title = font(43, bold=True)
    subtitle = font(23)
    section = font(29, bold=True)
    body = font(20)
    body_bold = font(21, bold=True)
    small = font(17)
    card_label = font(18)
    card_value = font(39, bold=True)
    matrix_value = font(28, bold=True)

    draw.text((70, 42), "Prueba tecnica - Granulometria en Tolva", font=title, fill=dark)
    draw.text(
        (70, 100),
        "Comparacion entre seleccion visual de la carga y resultado dominante del pipeline",
        font=subtitle,
        fill=muted,
    )

    cards = [
        ("Imagenes", f"{metrics['total']}", dark),
        ("Evaluables", f"{metrics['evaluable_ratio']:.1%}", blue),
        ("Acierto total", f"{metrics['accuracy_total']:.1%}", green),
        ("F1 macro", f"{metrics['f1_macro']:.3f}", green),
    ]
    card_y, card_h, card_gap = 150, 125, 24
    card_w = (width - 140 - card_gap * 3) // 4
    for index, (label, value, value_color) in enumerate(cards):
        x = 70 + index * (card_w + card_gap)
        draw.rectangle((x, card_y, x + card_w, card_y + card_h), fill=white, outline=border)
        draw.text((x + 20, card_y + 19), label, font=card_label, fill=muted)
        draw.text((x + 20, card_y + 56), value, font=card_value, fill=value_color)

    draw.text((70, 320), "Matriz de confusion", font=section, fill=dark)
    draw.text((70, 360), "Filas: clase esperada | Columnas: prediccion", font=small, fill=muted)
    matrix_x, matrix_y = 245, 410
    cell_w, cell_h = 190, 95
    for col, predicted in enumerate(PREDICTED_CLASSES):
        centered_text(
            draw,
            (matrix_x + col * cell_w + cell_w // 2, matrix_y - 25),
            DISPLAY_NAMES[predicted],
            small,
            dark,
        )
    max_value = max(
        metrics["confusion_matrix"][expected][predicted]
        for expected in EXPECTED_CLASSES
        for predicted in PREDICTED_CLASSES
    )
    for row_index, expected in enumerate(EXPECTED_CLASSES):
        y = matrix_y + row_index * cell_h
        draw.text((70, y + 32), DISPLAY_NAMES[expected], font=body_bold, fill=dark)
        for col_index, predicted in enumerate(PREDICTED_CLASSES):
            x = matrix_x + col_index * cell_w
            value = metrics["confusion_matrix"][expected][predicted]
            intensity = value / max(max_value, 1)
            if expected == predicted:
                fill = tuple(int(white[i] + (pale_green[i] - white[i]) * intensity) for i in range(3))
            elif value:
                fill = tuple(int(white[i] + (pale_red[i] - white[i]) * intensity) for i in range(3))
            else:
                fill = white
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=fill, outline=border)
            centered_text(draw, (x + cell_w // 2, y + cell_h // 2), str(value), matrix_value, dark)

    chart_x, chart_y = 1080, 410
    chart_w, chart_h = 420, 300
    draw.text((1080, 320), "Acierto por clase", font=section, fill=dark)
    draw.line((chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h), fill=dark, width=2)
    bar_w = 90
    for index, expected in enumerate(EXPECTED_CLASSES):
        accuracy = metrics["per_class"][expected]["accuracy_total"]
        x = chart_x + 35 + index * 135
        bar_h = int(chart_h * accuracy)
        y = chart_y + chart_h - bar_h
        draw.rectangle((x, y, x + bar_w, chart_y + chart_h), fill=green)
        centered_text(draw, (x + bar_w // 2, y - 18), f"{accuracy:.1%}", small, dark)
        centered_text(
            draw,
            (x + bar_w // 2, chart_y + chart_h + 28),
            DISPLAY_NAMES[expected],
            small,
            dark,
        )

    summary_top = 790
    draw.rectangle((70, summary_top, 1530, 990), fill=white, outline=border)
    draw.text((95, summary_top + 22), "Resumen por clase", font=section, fill=dark)
    y = summary_top + 70
    for expected in EXPECTED_CLASSES:
        item = metrics["per_class"][expected]
        line = (
            f"{DISPLAY_NAMES[expected]}: {item['correct']}/{item['total']} correctas | "
            f"precision {item['precision']:.1%} | recall {item['recall']:.1%} | "
            f"F1 {item['f1']:.3f}"
        )
        draw.text((105, y), line, font=body, fill=dark)
        y += 40

    indicators_top = 1015
    indicators_bottom = 1195
    draw.rectangle(
        (70, indicators_top, 1530, indicators_bottom),
        fill=white,
        outline=border,
    )
    draw.text(
        (95, indicators_top + 20),
        "Indicadores granulometricos por clase visual esperada",
        font=section,
        fill=dark,
    )
    indicator_width = 440
    indicator_gap = 40
    indicator_x = 95
    for expected in EXPECTED_CLASSES:
        item = metrics["per_class"][expected]
        draw.text(
            (indicator_x, indicators_top + 70),
            DISPLAY_NAMES[expected],
            font=body_bold,
            fill=dark,
        )
        draw.text(
            (indicator_x, indicators_top + 108),
            f"IG promedio: {item['mean_indice_granulometrico']:.3f}",
            font=body,
            fill=dark,
        )
        draw.text(
            (indicator_x, indicators_top + 140),
            f"Dispersion promedio: {item['mean_dispersion_granulometrica']:.3f}",
            font=body,
            fill=dark,
        )
        indicator_x += indicator_width + indicator_gap

    footer = (
        "IG: 1=fino, 2=medio, 3=grueso. Dispersion: 0=homogenea, "
        "1=maxima separacion ordinal. Valores promediados por frame."
    )
    draw.text((70, 1215), footer, font=small, fill=muted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.is_file():
        raise FileNotFoundError(f"No existe el resumen de inferencia: {summary_csv}")

    ordered_dir = run_dir / "resultados_ordenados"
    evaluation_dir = run_dir / "evaluacion"
    prepare_directory(ordered_dir, args.overwrite)
    prepare_directory(evaluation_dir, args.overwrite)

    source_rows = read_csv(summary_csv)
    if not source_rows:
        raise ValueError(f"El CSV no contiene resultados: {summary_csv}")
    evaluated = organize_results(source_rows, ordered_dir)
    metrics = build_metrics(evaluated)
    margin_rows = build_margin_analysis(evaluated)
    overlap = training_overlap(evaluated, args.training_split_csv)
    metrics["training_overlap"] = overlap

    write_csv(
        evaluation_dir / "evaluacion_por_imagen.csv",
        evaluated,
        [
            "image_name",
            "original_image_path",
            "expected_class",
            "predicted_class",
            "is_evaluable",
            "is_correct",
            "patch_count",
            "share_fino",
            "share_medio",
            "share_grueso",
            "indice_granulometrico",
            "dispersion_granulometrica",
            "dominant_share",
            "dominant_margin",
            "mean_patch_confidence",
            "max_yolo_conf",
            "comparison_path",
            "overlay_path",
        ],
    )
    write_csv(
        evaluation_dir / "matriz_confusion.csv",
        [
            {
                "clase_esperada": expected,
                **{
                    f"pred_{predicted}": metrics["confusion_matrix"][expected][predicted]
                    for predicted in PREDICTED_CLASSES
                },
            }
            for expected in EXPECTED_CLASSES
        ],
        ["clase_esperada", *[f"pred_{label}" for label in PREDICTED_CLASSES]],
    )
    write_csv(
        evaluation_dir / "analisis_margen.csv",
        margin_rows,
        ["margen_minimo", "frames_aceptados", "cobertura", "correctos", "accuracy"],
    )
    (evaluation_dir / "resumen_evaluacion.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_path = evaluation_dir / "reporte_prueba_tecnica.png"
    draw_report(metrics, report_path)
    write_analysis_report(
        evaluation_dir / "analisis_prueba_tecnica.md",
        metrics,
        margin_rows,
        overlap,
    )

    print("Evaluacion completada")
    print(f"Imagenes: {metrics['total']}")
    print(f"Evaluables: {metrics['evaluable']}/{metrics['total']}")
    print(f"Correctas: {metrics['correct']}/{metrics['total']}")
    print(f"Acierto total: {metrics['accuracy_total']:.2%}")
    print(f"F1 macro: {metrics['f1_macro']:.3f}")
    for expected in EXPECTED_CLASSES:
        item = metrics["per_class"][expected]
        print(
            f"{DISPLAY_NAMES[expected]}: {item['correct']}/{item['total']} "
            f"({item['accuracy_total']:.2%})"
        )
    print(f"Reporte: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
