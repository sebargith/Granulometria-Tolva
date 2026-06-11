from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PreprocessMode = Literal["rgb", "gray", "gray_clahe"]
ThresholdMode = Literal["none", "fixed", "otsu", "percentile"]
InputNormalization = Literal["bgr255_mean", "zero_one", "imagenet"]
OutputTransform = Literal["sigmoid", "none"]


@dataclass(frozen=True)
class DexiNedConfig:
    """Configuracion de inferencia DexiNed para mapas de bordes."""

    weights: Path | None = None
    pretrained: bool = True
    device: str | None = None
    image_size: tuple[int, int] | None = None
    preprocess: PreprocessMode = "gray_clahe"
    normalization: InputNormalization = "bgr255_mean"
    output_transform: OutputTransform = "sigmoid"
    contrast_stretch: bool = True
    contrast_percentiles: tuple[float, float] = (1.0, 99.0)
    output_index: int = -1
    threshold_mode: ThresholdMode = "otsu"
    threshold_value: int = 128
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    strict_weights: bool = True


def parse_image_size(value: str | int | None) -> tuple[int, int] | None:
    """Convierte valores tipo 512 o 512x384 a tupla (alto, ancho)."""

    if value is None:
        return None
    if isinstance(value, int):
        return None if value <= 0 else (value, value)

    text = str(value).strip().lower()
    if text in {"", "none", "original", "0"}:
        return None
    if "x" in text:
        width_text, height_text = text.split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
        return (height, width)

    side = int(text)
    return (side, side)


def _load_cv_numpy_torch():
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Faltan dependencias para DexiNed. Instala con: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return cv2, np, torch


def _load_dexined_class():
    candidates = (
        ("kornia.filters", "DexiNed"),
        ("kornia.models", "DexiNed"),
        ("kornia.models.dexined", "DexiNed"),
    )
    last_error: Exception | None = None

    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as exc:  # pragma: no cover - depende de version de kornia.
            last_error = exc

    raise RuntimeError(
        "No se pudo importar DexiNed desde Kornia. Revisa que 'kornia' este instalado."
    ) from last_error


def _select_device(requested_device: str | None):
    _, _, torch = _load_cv_numpy_torch()
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("El checkpoint no contiene un state_dict compatible.")

    cleaned = {}
    for key, value in checkpoint.items():
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def load_dexined_model(config: DexiNedConfig):
    """Carga DexiNed desde Kornia, con pesos preentrenados o checkpoint local."""

    _, _, torch = _load_cv_numpy_torch()
    device = _select_device(config.device)
    dexined_cls = _load_dexined_class()

    use_pretrained = bool(config.pretrained and config.weights is None)
    try:
        model = dexined_cls(pretrained=use_pretrained)
    except TypeError:
        model = dexined_cls()

    if config.weights is not None:
        checkpoint_path = config.weights.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No existe el checkpoint DexiNed: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=config.strict_weights)

    model.to(device)
    model.eval()
    return model, device


def read_image_bgr(image_path: Path):
    cv2, _, _ = _load_cv_numpy_torch()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    return image


def preprocess_image(
    image_bgr,
    mode: PreprocessMode,
    image_size: tuple[int, int] | None,
    clahe_clip_limit: float,
    clahe_tile_grid_size: tuple[int, int],
):
    """Preprocesa imagen para DexiNed y devuelve imagen RGB uint8."""

    cv2, np, _ = _load_cv_numpy_torch()

    if mode == "rgb":
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif mode in {"gray", "gray_clahe"}:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if mode == "gray_clahe":
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_tile_grid_size,
            )
            gray = clahe.apply(gray)
        image_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Modo de preprocesamiento no soportado: {mode}")

    original_shape = image_rgb.shape[:2]
    if image_size is not None and original_shape != image_size:
        height, width = image_size
        interpolation = cv2.INTER_AREA if height < original_shape[0] else cv2.INTER_LINEAR
        image_rgb = cv2.resize(image_rgb, (width, height), interpolation=interpolation)

    return image_rgb, original_shape


def image_to_tensor(image_rgb, device, normalization: InputNormalization):
    _, np, torch = _load_cv_numpy_torch()
    image_float = image_rgb.astype(np.float32)

    if normalization == "zero_one":
        image_float = image_float / 255.0
    elif normalization == "imagenet":
        image_float = image_float / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_float = (image_float - mean) / std
    elif normalization == "bgr255_mean":
        image_float = image_float[:, :, ::-1].copy()
        bgr_mean = np.array([103.939, 116.779, 123.680], dtype=np.float32)
        image_float = image_float - bgr_mean
    else:
        raise ValueError(f"Normalizacion no soportada: {normalization}")

    tensor = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def _to_display_edge(edge_values, contrast_stretch: bool, percentiles: tuple[float, float]):
    _, np, _ = _load_cv_numpy_torch()
    edge = np.asarray(edge_values, dtype=np.float32)
    edge = np.nan_to_num(edge, nan=0.0, posinf=0.0, neginf=0.0)

    if contrast_stretch:
        low_pct, high_pct = percentiles
        low = float(np.percentile(edge, low_pct))
        high = float(np.percentile(edge, high_pct))
        if high > low:
            edge = (edge - low) / (high - low)

    edge = np.clip(edge, 0.0, 1.0)
    return (edge * 255.0).astype(np.uint8)


def infer_edge_map(image_bgr, model, device, config: DexiNedConfig):
    """Retorna mapa DexiNed uint8 en el tamano original de la imagen."""

    cv2, np, torch = _load_cv_numpy_torch()
    image_rgb, original_shape = preprocess_image(
        image_bgr=image_bgr,
        mode=config.preprocess,
        image_size=config.image_size,
        clahe_clip_limit=config.clahe_clip_limit,
        clahe_tile_grid_size=config.clahe_tile_grid_size,
    )
    tensor = image_to_tensor(image_rgb, device, config.normalization)

    with torch.inference_mode():
        outputs = model(tensor)
        if isinstance(outputs, (list, tuple)):
            output = outputs[config.output_index]
        else:
            output = outputs
        if config.output_transform == "sigmoid":
            edge_prob = torch.sigmoid(output)
        elif config.output_transform == "none":
            edge_prob = output
        else:
            raise ValueError(f"Transformacion de salida no soportada: {config.output_transform}")
        if tuple(edge_prob.shape[-2:]) != tuple(original_shape):
            edge_prob = torch.nn.functional.interpolate(
                edge_prob,
                size=original_shape,
                mode="bilinear",
                align_corners=False,
            )

    edge = edge_prob.squeeze().detach().cpu().numpy()
    return _to_display_edge(
        edge,
        contrast_stretch=config.contrast_stretch,
        percentiles=config.contrast_percentiles,
    )


def make_foreground_mask(
    image_bgr,
    threshold: int = 8,
    erode_iterations: int = 8,
    keep_largest: bool = True,
):
    """Crea mascara de primer plano para recortes con fondo negro."""

    cv2, np, _ = _load_cv_numpy_torch()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > int(threshold)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest_label).astype(np.uint8) * 255

    if erode_iterations > 0:
        mask = cv2.erode(mask, kernel, iterations=int(erode_iterations))
    return mask


def apply_mask_to_edge(edge_map, mask):
    cv2, _, _ = _load_cv_numpy_torch()
    return cv2.bitwise_and(edge_map, edge_map, mask=mask)


def binarize_edge_map(
    edge_map,
    mode: ThresholdMode,
    threshold_value: int = 128,
    valid_mask=None,
):
    cv2, np, _ = _load_cv_numpy_torch()
    if mode == "none":
        return None
    values = edge_map[valid_mask > 0] if valid_mask is not None else edge_map.ravel()
    values = values[values > 0]
    if values.size == 0:
        return np.zeros_like(edge_map, dtype=np.uint8)
    if mode == "fixed":
        _, binary = cv2.threshold(edge_map, int(threshold_value), 255, cv2.THRESH_BINARY)
        return binary
    if mode == "otsu":
        target = edge_map if valid_mask is None else cv2.bitwise_and(edge_map, edge_map, mask=valid_mask)
        _, binary = cv2.threshold(target, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    if mode == "percentile":
        threshold = float(np.percentile(values, float(threshold_value)))
        _, binary = cv2.threshold(edge_map, threshold, 255, cv2.THRESH_BINARY)
        return binary
    raise ValueError(f"Modo de binarizacion no soportado: {mode}")


def refine_binary_edges(
    binary_edge,
    open_iterations: int = 0,
    close_iterations: int = 0,
    thin: bool = False,
):
    cv2, np, _ = _load_cv_numpy_torch()
    refined = binary_edge.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)

    if open_iterations > 0:
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=int(open_iterations))
    if close_iterations > 0:
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=int(close_iterations))
    if thin:
        refined = skeletonize_binary(refined)
    return refined


def skeletonize_binary(binary_edge):
    """Adelgaza bordes binarios usando esqueleto morfologico OpenCV."""

    cv2, np, _ = _load_cv_numpy_torch()
    image = (binary_edge > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(image) > 0:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(image, opened)
        eroded = cv2.erode(image, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded
    return skeleton


def make_edge_overlay(image_bgr, edge_map, binary_edge=None, alpha: float = 0.75):
    """Dibuja bordes DexiNed sobre la imagen original para control visual."""

    cv2, np, _ = _load_cv_numpy_torch()
    overlay = image_bgr.copy()
    if binary_edge is None:
        mask = edge_map > 128
        intensity = edge_map
    else:
        mask = binary_edge > 0
        intensity = binary_edge

    color_layer = np.zeros_like(image_bgr)
    color_layer[:, :, 1] = intensity
    color_layer[:, :, 2] = intensity
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, color_layer, alpha, 0)
    overlay[mask] = blended[mask]
    return overlay


def write_image(path: Path, image) -> None:
    cv2, _, _ = _load_cv_numpy_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"No se pudo guardar imagen: {path}")
