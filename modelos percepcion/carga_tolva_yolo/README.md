# Modelo YOLO de carga visible en tolva

Componente de percepcion para segmentar la superficie visible de carga en la
tolva de un camion.

## Modelo

Modelo principal:

```text
models/haul_truck_segment_m_736_low_data_aug_07012025.pt
task: segment
imgsz: 736
class 0: caex
class 1: material
```

`best.pt` se conserva localmente como modelo alternativo, pero no es el modelo
principal del pipeline.

## Inferencia

Desde la raiz del proyecto:

```powershell
conda activate memoria-dexined

python "modelos percepcion\carga_tolva_yolo\predict_carga.py" `
  --source "input\Data test 1" `
  --conf 0.25
```

Tambien se conserva el comando compatible:

```powershell
python predict_tolva.py `
  --source "input\Data test 1" `
  --conf 0.25
```

Los valores por defecto ya corresponden al modelo:

```text
model: models/haul_truck_segment_m_736_low_data_aug_07012025.pt
imgsz: 736
class-ids: 1
output-dir: outputs/yolo_carga
```

## Salidas

Cada fuente genera una carpeta independiente:

```text
outputs/yolo_carga/<nombre-input>/
|-- masks/
|-- overlays/
|-- previews/
|-- params.json
`-- predictions.csv
```

Las mascaras binarias representan la clase `material` y son la entrada esperada
para generar patches internos a la region de material.

## Preparar patches para Streamlit

El comando integrado ejecuta la segmentacion y genera patches `256x256` con
50% de solapamiento y al menos 80% de carga:

```powershell
python "modelos percepcion\carga_tolva_yolo\preparar_patches.py" `
  --source "Data parches test" `
  --run-name data_parches_test_carga `
  --open-labeler
```

Salidas:

```text
outputs/yolo_carga/data_parches_test_carga/   mascaras y overlays
outputs/patches/data_parches_test_carga/      patches.csv, images y labels.csv
```
