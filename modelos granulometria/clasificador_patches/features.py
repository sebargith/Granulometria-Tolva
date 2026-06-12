from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


FEATURE_VERSION = "opencv_edges_v1"


def _normalized_histogram(image: np.ndarray, bins: int) -> np.ndarray:
    histogram = cv2.calcHist([image], [0], None, [bins], [0, 256]).ravel()
    total = float(histogram.sum())
    return histogram / total if total > 0 else histogram


def _append_stats(features: OrderedDict[str, float], prefix: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32).ravel()
    percentiles = np.percentile(values, [10, 25, 50, 75, 90])
    features[f"{prefix}_mean"] = float(values.mean())
    features[f"{prefix}_std"] = float(values.std())
    for percentile, value in zip((10, 25, 50, 75, 90), percentiles):
        features[f"{prefix}_p{percentile}"] = float(value)


def _lbp_histogram(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    neighbors = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
    histogram, _ = np.histogram(lbp, bins=16, range=(0, 256))
    histogram = histogram.astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def _edge_component_features(binary: np.ndarray) -> dict[str, float]:
    height, width = binary.shape
    image_area = float(height * width)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(contour) for contour in contours], dtype=np.float32)
    perimeters = np.array(
        [cv2.arcLength(contour, closed=True) for contour in contours], dtype=np.float32
    )

    result = {
        "contour_count_per_kpx": len(contours) / max(image_area / 1000.0, 1.0),
        "contour_area_sum_fraction": float(areas.sum() / image_area) if areas.size else 0.0,
        "contour_area_mean": float(areas.mean()) if areas.size else 0.0,
        "contour_area_median": float(np.median(areas)) if areas.size else 0.0,
        "contour_area_p90": float(np.percentile(areas, 90)) if areas.size else 0.0,
        "contour_perimeter_mean": float(perimeters.mean()) if perimeters.size else 0.0,
        "contour_perimeter_p90": float(np.percentile(perimeters, 90)) if perimeters.size else 0.0,
    }

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    result.update(
        {
            "component_count_per_kpx": (component_count - 1)
            / max(image_area / 1000.0, 1.0),
            "component_area_mean": float(component_areas.mean())
            if component_areas.size
            else 0.0,
            "component_area_p90": float(np.percentile(component_areas, 90))
            if component_areas.size
            else 0.0,
        }
    )
    return result


def extract_features_from_image(image_bgr: np.ndarray) -> OrderedDict[str, float]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("La imagen esta vacia.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    features: OrderedDict[str, float] = OrderedDict()

    _append_stats(features, "gray", gray)
    _append_stats(features, "clahe", enhanced)
    for index, value in enumerate(_normalized_histogram(enhanced, bins=16)):
        features[f"gray_hist_{index:02d}"] = float(value)

    probabilities = _normalized_histogram(enhanced, bins=256)
    nonzero = probabilities[probabilities > 0]
    features["gray_entropy"] = float(-(nonzero * np.log2(nonzero)).sum())

    laplacian = cv2.Laplacian(enhanced, cv2.CV_32F)
    features["laplacian_variance"] = float(laplacian.var())
    features["laplacian_abs_mean"] = float(np.abs(laplacian).mean())

    sobel_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    _append_stats(features, "gradient", gradient)
    for threshold in (20, 40, 80, 120):
        features[f"gradient_density_{threshold}"] = float((gradient >= threshold).mean())

    for low, high in ((30, 90), (60, 150), (100, 200)):
        canny = cv2.Canny(enhanced, low, high)
        prefix = f"canny_{low}_{high}"
        features[f"{prefix}_density"] = float((canny > 0).mean())
        for name, value in _edge_component_features(canny).items():
            features[f"{prefix}_{name}"] = value

    for index, value in enumerate(_lbp_histogram(enhanced)):
        features[f"lbp_{index:02d}"] = float(value)

    enhanced_float = enhanced.astype(np.float32) / 255.0
    for orientation, theta in enumerate(np.linspace(0, np.pi, 4, endpoint=False)):
        kernel = cv2.getGaborKernel((9, 9), 3.0, float(theta), 6.0, 0.5, 0)
        response = cv2.filter2D(enhanced_float, cv2.CV_32F, kernel)
        absolute = np.abs(response)
        features[f"gabor_{orientation}_mean"] = float(absolute.mean())
        features[f"gabor_{orientation}_std"] = float(absolute.std())

    resized = cv2.resize(enhanced_float, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized)
    for row in range(4):
        for column in range(4):
            if row == 0 and column == 0:
                continue
            features[f"dct_{row}_{column}"] = float(dct[row, column])

    return features


def extract_features(image_path: Path) -> OrderedDict[str, float]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer el parche: {image_path}")
    return extract_features_from_image(image)

