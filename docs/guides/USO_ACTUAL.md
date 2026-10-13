# Uso actual del proyecto

Para instalar desde un clon nuevo o ejecutar la demo sin pesos, consulte
[QUICKSTART.md](QUICKSTART.md). Los comandos de entrenamiento requieren datasets
y modelos externos. Pase siempre `--dataset` para evitar defaults historicos.

Todos los comandos se ejecutan desde la raiz del repositorio:

```powershell
conda activate memoria-dexined
```

## Aplicacion web

Crear una vez el acceso directo local:

```powershell
powershell -ExecutionPolicy Bypass -File "tools\patch_labeler\crear_acceso_directo.ps1"
```

Luego hacer doble clic en `Abrir Webapp Granulometria de Carga.lnk`.

Desde PowerShell:

```powershell
& "tools\patch_labeler\abrir_etiquetador.cmd" `
  "outputs\patches\grid96_batch003_data3_mixto_ai"
```

La aplicacion abre el dataset final revisado:

```text
outputs/patches/grid96_batch003_data3_mixto_ai
```

## Inferencia de imagenes

```powershell
python "modelos granulometria\inferencia_tolva\predict_granulometria_tolva.py" `
  --source "input\prueba_cnn_50_frames" `
  --run-name "cnn_prueba_actual" `
  --device 0 `
  --patch-device cuda
```

Valores por defecto vigentes:

```text
modelo de patches: ResNet18 Batch 003 revisado
patch: 96 x 96 px
stride: 96 px
normalizacion: 896 px de ancho
modelo YOLO: modelos percepcion/carga_tolva_yolo/
             haul_truck_segment_m_736_low_data_aug_07012025.pt
```

## Video

Abrir `Abrir Webapp Granulometria de Carga.lnk`, seleccionar `Pipeline de video`, cargar el archivo y
ejecutar. Las corridas se guardan en:

```text
outputs/video_pipeline/
```

## Entrenamiento ResNet18

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

## Entrenamiento Random Forest

```powershell
python "modelos granulometria\clasificador_patches\train_random_forest.py" `
  --dataset "outputs\patches\grid96_batch003_data3_mixto_ai" `
  --run-name "rf_grid96_nuevo" `
  --jobs 1
```

## Nuevo batch de etiquetado

```powershell
python "modelos granulometria\inferencia_tolva\generar_grillas_normalizadas.py" `
  --source "data\datasets\patch_frames\batch003" `
  --output-dir "data\labeling\batches\nuevo_batch\grillas" `
  --run-name "grillas" `
  --widths 896 `
  --cell-size 96 `
  --recursive

python "scripts\labeling\prepare_grid_ai_packages.py" `
  --grid-run "data\labeling\batches\nuevo_batch\grillas\grillas" `
  --batch-dir "data\labeling\batches\nuevo_batch" `
  --package-prefix "batch004" `
  --mix-by-source-parent

python "scripts\labeling\compile_grid_ai_labels.py" `
  --batch-dir "data\labeling\batches\nuevo_batch" `
  --output-dataset "outputs\patches\grid96_batch004"
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
