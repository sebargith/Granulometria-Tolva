from __future__ import annotations

import math
from collections.abc import Mapping


GRANULOMETRY_CLASS_SCORES = {
    "fino": 1.0,
    "medio": 2.0,
    "grueso": 3.0,
}


def compute_granulometry_metrics(
    shares: Mapping[str, float],
) -> dict[str, float | None]:
    """Calcula tendencia central y dispersion de la composicion granulometrica."""
    values = {
        label: max(0.0, float(shares.get(label, 0.0)))
        for label in GRANULOMETRY_CLASS_SCORES
    }
    total = sum(values.values())
    if total <= 0.0:
        return {
            "indice_granulometrico": None,
            "dispersion_granulometrica": None,
        }

    proportions = {label: value / total for label, value in values.items()}
    index = sum(
        proportions[label] * score
        for label, score in GRANULOMETRY_CLASS_SCORES.items()
    )
    variance = sum(
        proportions[label] * (score - index) ** 2
        for label, score in GRANULOMETRY_CLASS_SCORES.items()
    )
    return {
        "indice_granulometrico": index,
        "dispersion_granulometrica": math.sqrt(max(variance, 0.0)),
    }


def metric_definitions() -> dict[str, object]:
    return {
        "class_scores": GRANULOMETRY_CLASS_SCORES,
        "indice_granulometrico": "sum(p_clase * puntaje_clase)",
        "dispersion_granulometrica": (
            "sqrt(sum(p_clase * (puntaje_clase - indice_granulometrico)^2))"
        ),
        "interpretation": (
            "El indice varia entre 1 (fino) y 3 (grueso). "
            "La dispersion varia entre 0 (homogeneo) y 1 (maxima separacion ordinal)."
        ),
    }
