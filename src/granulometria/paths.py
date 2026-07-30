from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LEGACY_DATA_DIR = DATA_DIR / "legacy"
PATCH_FRAME_DATASET_DIR = DATA_DIR / "datasets" / "patch_frames"
TECHNICAL_TEST_DATASET_DIR = DATA_DIR / "evaluation" / "prueba_tecnica"
LABELING_DIR = DATA_DIR / "labeling"
LABELING_BATCHES_DIR = LABELING_DIR / "batches"
LABELING_REFERENCES_DIR = LABELING_DIR / "references"
LABELING_NORMALIZED_REFERENCES_DIR = (
    LABELING_DIR / "references_normalized_20260618_203838"
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PATCH_DATASETS_DIR = OUTPUTS_DIR / "patches"
FINAL_PATCH_DATASET_DIR = PATCH_DATASETS_DIR / "grid96_batch003_data3_mixto_ai"
VIDEO_PIPELINE_OUTPUT_DIR = OUTPUTS_DIR / "video_pipeline"

PERCEPTION_MODEL = (
    PROJECT_ROOT
    / "modelos percepcion"
    / "carga_tolva_yolo"
    / "haul_truck_segment_m_736_low_data_aug_07012025.pt"
)

CLASSIFIER_DIR = PROJECT_ROOT / "modelos granulometria" / "clasificador_patches"
CLASSIFIER_RUNS_DIR = CLASSIFIER_DIR / "runs"
FINAL_CNN_RUN_DIR = (
    CLASSIFIER_RUNS_DIR
    / "resnet18_rgb_grid96_batch003_revisado_cuda_20260702"
)
FINAL_CNN_MODEL = FINAL_CNN_RUN_DIR / "model.pt"
FINAL_RF_RUN_DIR = CLASSIFIER_RUNS_DIR / "rf_grid96_batch003_revisado_20260702"
FINAL_RF_MODEL = FINAL_RF_RUN_DIR / "model.joblib"

INFERENCE_SCRIPT = (
    PROJECT_ROOT
    / "modelos granulometria"
    / "inferencia_tolva"
    / "predict_granulometria_tolva.py"
)

FINAL_REPORT_DIR = PROJECT_ROOT / "docs" / "informe_final"
FINAL_REPORT_ASSETS_DIR = FINAL_REPORT_DIR / "assets"
FINAL_REPORT_PATH = FINAL_REPORT_DIR / "Capitulo_Granulometria_Informe_Final.docx"
