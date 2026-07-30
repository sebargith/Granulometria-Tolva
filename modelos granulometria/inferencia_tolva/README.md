# Inferencia de granulometria en tolva

`predict_granulometria_tolva.py` ejecuta el pipeline:

```text
frame -> YOLO-seg -> normalizacion -> patches -> clasificador -> agregacion
```

Configuracion por defecto vigente:

```text
clasificador: ResNet18 Batch 003 revisado
patch: 96 x 96 px
stride: 96 px
ancho normalizado: 896 px
clases YOLO: 1, material
```

Ejemplo:

```powershell
python "modelos granulometria\inferencia_tolva\predict_granulometria_tolva.py" `
  --source "input\prueba_cnn_50_frames" `
  --run-name "cnn_prueba_actual" `
  --device 0 `
  --patch-device cuda
```

Para usar el baseline Random Forest:

```powershell
python "modelos granulometria\inferencia_tolva\predict_granulometria_tolva.py" `
  --source "input\prueba_cnn_50_frames" `
  --patch-model "modelos granulometria\clasificador_patches\runs\rf_grid96_batch003_revisado_20260702\model.joblib" `
  --patch-model-type rf `
  --run-name "rf_prueba_actual"
```

`generar_grillas_normalizadas.py` prepara cargas para etiquetado. Sus valores
por defecto son ancho 896 px y celdas de 96 px.
