# Aplicacion Streamlit de granulometria

La aplicacion integra tres funciones:

1. etiquetado de patches;
2. revision y correccion del dataset;
3. pipeline completo de video.

Abrir desde la raiz:

Hacer doble clic en `Abrir Webapp Granulometria de Carga.lnk`.

Desde PowerShell:

```powershell
& "tools\patch_labeler\abrir_etiquetador.cmd" `
  "outputs\patches\grid96_batch003_data3_mixto_ai"
```

El launcher usa el dataset final:

```text
outputs/patches/grid96_batch003_data3_mixto_ai
```

Tambien puede abrirse otra carpeta:

```powershell
tools\patch_labeler\abrir_etiquetador.cmd "ruta\al\dataset"
```

Si no se entrega una ruta, `launcher.py` selecciona el dataset generado mas
reciente y usa `data/datasets/patch_frames` como ultimo fallback.

Modelos por defecto:

```text
CNN: modelos granulometria/clasificador_patches/runs/
     resnet18_rgb_grid96_batch003_revisado_cuda_20260702/model.pt

RF:  modelos granulometria/clasificador_patches/runs/
     rf_grid96_batch003_revisado_20260702/model.joblib

YOLO: modelos percepcion/carga_tolva_yolo/
      haul_truck_segment_m_736_low_data_aug_07012025.pt
```

Las rutas compartidas se obtienen desde `src/granulometria/paths.py`.
