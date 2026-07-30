# Granulometria visual en tolva

Pipeline experimental para estimar la granulometria visual aparente del material
visible en la tolva de camiones CAEX. La entrada operacional es un video o una
carpeta de frames; la salida incluye proporciones de material fino, medio y
grueso, clase dominante, indice granulometrico, dispersion y trazabilidad visual.

El resultado describe la superficie visible de la carga. No corresponde a una
medicion fisica en milimetros ni a la distribucion granulometrica de todo el
volumen transportado.

## Pipeline actual

```text
video o imagenes
-> extraccion de frames
-> YOLO-seg: mascara de carga
-> normalizacion de la carga a 896 px de ancho
-> grilla de patches RGB de 96 x 96 px
-> ResNet18: fino / medio / grueso
-> agregacion por frame y por video
-> porcentajes, dominante, indice, dispersion, confianza y overlays
```

Modelo principal:

```text
modelos granulometria/clasificador_patches/runs/
  resnet18_rgb_grid96_batch003_revisado_cuda_20260702/model.pt
```

Baseline conservado:

```text
modelos granulometria/clasificador_patches/runs/
  rf_grid96_batch003_revisado_20260702/model.joblib
```

Modelo de percepcion:

```text
modelos percepcion/carga_tolva_yolo/
  haul_truck_segment_m_736_low_data_aug_07012025.pt
```

## Estructura

```text
Granulometria/
|-- data/
|   |-- raw/evidencias_caex/              # evidencia original
|   |-- datasets/patch_frames/             # frames de Batch 001, 002 y 003
|   |-- evaluation/prueba_tecnica/         # 90 frames de evaluacion
|   |-- labeling/                          # batches, referencias y respuestas IA
|   `-- legacy/                            # datos historicos no usados por el pipeline actual
|-- docs/
|   |-- informe_final/                     # Word, figuras y referencias del informe
|   |-- guides/                            # comandos actuales
|   |-- labeling/                          # prompt e indice de batches
|   |-- methodology/                       # documentos metodologicos historicos
|   |-- presentations/
|   `-- results/
|-- input/                                 # entradas manuales de prueba
|-- modelos granulometria/                 # entrenamiento e inferencia granulometrica
|-- modelos percepcion/                    # herramientas y pesos YOLO de carga
|-- outputs/                               # datasets procesados y corridas vigentes
|-- scripts/
|   |-- labeling/
|   |-- reporting/
|   |-- evaluate_granulometry_technical_test.py
|   `-- run_dexined_edges.py
|-- src/granulometria/                     # metricas, rutas compartidas y DexiNed
|-- tests/
|-- tools/patch_labeler/                   # aplicacion Streamlit
|-- tools/patch_labeler/crear_acceso_directo.ps1 # genera el acceso local
`-- requirements.txt
```

## Entorno

Desde la raiz del repositorio:

```powershell
conda activate memoria-dexined
pip install -r requirements.txt
```

Para CNN con GPU se requiere una instalacion de PyTorch compatible con CUDA.

## Abrir la aplicacion

El acceso directo no se versiona porque contiene una ruta local. Para crearlo:

```powershell
powershell -ExecutionPolicy Bypass -File "tools\patch_labeler\crear_acceso_directo.ps1"
```

Luego se puede abrir `Abrir Webapp Granulometria de Carga.lnk`. Para iniciar
directamente desde PowerShell:

```powershell
& "tools\patch_labeler\abrir_etiquetador.cmd" `
  "outputs\patches\grid96_batch003_data3_mixto_ai"
```

La aplicacion permite:

- revisar y corregir etiquetas de patches;
- filtrar por clase y confianza;
- ejecutar el pipeline completo sobre video;
- revisar overlays y resultados por frame;
- consultar el resultado agregado del video;
- descargar CSV y artefactos de la corrida.

Las correcciones se guardan en:

```text
outputs/patches/grid96_batch003_data3_mixto_ai/labels.csv
```

## Inferencia sobre imagenes

El predictor usa por defecto el modelo ResNet18 final, patches de 96 px,
stride 96 y normalizacion a 896 px:

```powershell
python "modelos granulometria\inferencia_tolva\predict_granulometria_tolva.py" `
  --source "input\prueba_cnn_50_frames" `
  --run-name "cnn_prueba_actual" `
  --device 0 `
  --patch-device cuda
```

La salida queda en:

```text
outputs/granulometria_tolva/<run-name>/
```

## Entrenamiento

ResNet18 RGB:

```powershell
python "modelos granulometria\clasificador_patches\train_resnet.py" `
  --dataset "outputs\patches\grid96_batch003_data3_mixto_ai" `
  --run-name "resnet18_rgb_grid96_nuevo" `
  --image-size 96 `
  --batch-size 64 `
  --freeze-epochs 8 `
  --finetune-epochs 8 `
  --device cuda
```

Random Forest:

```powershell
python "modelos granulometria\clasificador_patches\train_random_forest.py" `
  --dataset "outputs\patches\grid96_batch003_data3_mixto_ai" `
  --run-name "rf_grid96_nuevo" `
  --n-jobs 1
```

El split se realiza por frame, no por patch, para evitar fuga de informacion.
Las etiquetas `invalido` y `ambiguo` se excluyen del entrenamiento.

## Preparar un nuevo batch para etiquetado

1. Generar grillas normalizadas:

```powershell
python "modelos granulometria\inferencia_tolva\generar_grillas_normalizadas.py" `
  --source "data\datasets\patch_frames\batch003" `
  --output-dir "data\labeling\batches\nuevo_batch\grillas" `
  --run-name "grillas" `
  --widths 896 `
  --cell-size 96 `
  --recursive
```

2. Preparar paquetes:

```powershell
python "scripts\labeling\prepare_grid_ai_packages.py" `
  --grid-run "data\labeling\batches\nuevo_batch\grillas\grillas" `
  --batch-dir "data\labeling\batches\nuevo_batch" `
  --package-prefix "batch004" `
  --mix-by-source-parent
```

3. Compilar las respuestas:

```powershell
python "scripts\labeling\compile_grid_ai_labels.py" `
  --batch-dir "data\labeling\batches\nuevo_batch" `
  --output-dataset "outputs\patches\grid96_batch004" `
  --overwrite
```

## Prueba tecnica

```powershell
python "modelos granulometria\inferencia_tolva\predict_granulometria_tolva.py" `
  --source "data\evaluation\prueba_tecnica" `
  --recursive `
  --run-name "prueba_tecnica_actual" `
  --output-dir "outputs\prueba_tecnica"

python "scripts\evaluate_granulometry_technical_test.py" `
  --run-dir "outputs\prueba_tecnica\prueba_tecnica_actual" `
  --overwrite
```

## Verificacion

```powershell
python -m unittest discover -s tests -v
python -m compileall src tools scripts "modelos granulometria" "modelos percepcion"
```

El informe vigente se encuentra en:

```text
docs/informe_final/Capitulo_Granulometria_Informe_Final.docx
```

Los detalles operativos adicionales estan en `docs/guides/USO_ACTUAL.md`.
Los modelos y datasets externos necesarios se describen en
`docs/guides/ARTEFACTOS_EXTERNOS.md`.
