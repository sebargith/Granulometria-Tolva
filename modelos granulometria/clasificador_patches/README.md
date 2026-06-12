# Clasificador de patches

Baseline reproducible para clasificar patches como `fino`, `medio` o
`grueso`. Usa caracteristicas de escala de grises, CLAHE, gradientes, Canny,
contornos, componentes conectados, LBP, Gabor y DCT. No usa color como senal
principal.

El split se realiza por `source_image`: todos los patches de un mismo frame
quedan en un solo conjunto para evitar fuga de informacion.

## Entrenar

```powershell
python "modelos granulometria\clasificador_patches\train_random_forest.py" `
  --dataset "outputs\patches\data_parches_test_carga_128"
```

La salida se versiona automaticamente en `runs/v001_random_forest`,
`runs/v002_random_forest`, etc. Cada corrida guarda el modelo, split,
metricas, matriz de confusion, predicciones de test e importancia de features.

Para reentrenar con datos nuevos manteniendo el test original:

```powershell
python "modelos granulometria\clasificador_patches\train_random_forest.py" `
  --dataset "outputs\patches\data_parches_test_carga_128" "ruta\dataset_nuevo" `
  --reuse-split "modelos granulometria\clasificador_patches\runs\v002_random_forest\group_split.csv"
```

Los frames nuevos se asignan a `train`; los frames ya conocidos conservan su
split. Para una evaluacion mas solida sera necesario crear un test fijo con
mas secuencias y mas ejemplos `grueso`.

## Inferir

```powershell
python "modelos granulometria\clasificador_patches\predict_patches.py" `
  --model "modelos granulometria\clasificador_patches\runs\v002_random_forest\model.joblib" `
  --dataset "outputs\patches\data_parches_test_carga_128"
```

Genera `predictions.csv` y `summary_by_frame.csv`.
