# Modelo YOLO de carga visible en tolva

Componente de percepcion para segmentar la superficie visible de carga en la
tolva de un camion.

## Modelo

```text
best.pt
```

Metadata verificada:

```text
task: segment
imgsz: 640
class 0: Carga
```

El peso se mantiene local y no se publica en Git.

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
model: modelos percepcion/carga_tolva_yolo/best.pt
imgsz: 640
class-ids: 0
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

Las mascaras binarias representan la clase `Carga` y son la entrada esperada
para generar patches internos a la region de material.
