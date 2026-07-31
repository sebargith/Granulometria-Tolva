# Percepcion YOLO de carga en tolva

Modelo entrenado de segmentacion utilizado por el pipeline:

```text
modelos percepcion/carga_tolva_yolo/
  haul_truck_segment_m_736_low_data_aug_07012025.pt
```

`best.pt` se conserva solo como artefacto historico y no se considera un
segundo modelo del proyecto ni parte del flujo vigente.

Inferencia de carga:

```powershell
python "modelos percepcion\carga_tolva_yolo\predict_carga.py" `
  --source "input\prueba_cnn_50_frames" `
  --run-name "segmentacion_carga"
```

Preparacion directa de patches:

```powershell
python "modelos percepcion\carga_tolva_yolo\preparar_patches.py" `
  --source "data\datasets\patch_frames\batch003" `
  --run-name "patches_carga" `
  --patch-size 96 `
  --stride 96
```

Para el pipeline completo se recomienda usar
`modelos granulometria/inferencia_tolva/predict_granulometria_tolva.py`.
