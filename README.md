# Granulometria visible en tolva

Proyecto experimental de vision computacional para estimar la granulometria
visual aparente del material visible en la tolva de camiones mineros.

El objetivo no es medir directamente una distribucion fisica en milimetros,
sino construir un indicador visual-operacional basado en segmentacion de la
carga, mapas de bordes, contornos y analisis local.

## Estado actual

Actualmente estan implementadas dos etapas:

1. Inferencia YOLO-seg para detectar el camion y la region `material`.
2. Inferencia DexiNed para generar mapas de bordes sobre imagenes o recortes.

Todavia falta conectar ambas etapas, generar patches, extraer features de
contornos y entrenar el clasificador fina/media/gruesa.

## Estructura

```text
.
|-- predict_tolva.py
|-- scripts/
|   `-- run_dexined_edges.py
|-- src/granulometria/
|   `-- dexined.py
|-- models/                 # pesos locales, ignorados por Git
|-- input/                  # imagenes de prueba, ignoradas por Git
|-- outputs/                # resultados, ignorados por Git
|-- data/                   # datos locales auxiliares, ignorados por Git
|-- Data filtrada/          # dataset local, ignorado por Git
|-- Data hecha anteriormente por psinet/ # dataset local, ignorado por Git
|-- requirements.txt
`-- para correr.txt
```

## Entorno

Se ha utilizado el entorno Conda `memoria-dexined`:

```powershell
cd "D:\Memoria\Granulometria"
conda activate memoria-dexined
python -m pip install -r requirements.txt
```

## Inferencia YOLO de material en tolva

El modelo esperado es:

```text
models/haul_truck_segment_m_736_low_data_aug_07012025.pt
```

Las imagenes originales se organizan por experimento:

```text
input/Data test 1/
input/Data test 2/
```

Ejemplo recomendado, filtrando solo la clase `material` (`class-id 1`):

```powershell
python predict_tolva.py `
  --model "models\haul_truck_segment_m_736_low_data_aug_07012025.pt" `
  --source "input\Data test 2" `
  --imgsz 736 `
  --conf 0.25 `
  --class-ids 1
```

Si se omite `--run-name`, la carpeta de salida toma el nombre del input. Si ya
existe, se agrega fecha y hora para no sobrescribir resultados.

```text
outputs/yolo_tolva/data_test_2/
|-- masks/
|-- overlays/
|-- previews/
|-- params.json
`-- predictions.csv
```

## Inferencia DexiNed

DexiNed se ejecuta mediante:

```powershell
python scripts\run_dexined_edges.py `
  --source "D:\ruta\a\imagenes_o_recortes" `
  --run-name dexined_prueba `
  --image-size original `
  --preprocess gray_clahe `
  --normalization bgr255_mean `
  --threshold percentile `
  --threshold-value 90
```

Salidas:

```text
outputs/dexined_edges/<run-name>/
|-- edge_maps/
|-- binary_edges/
|-- overlays/
|-- params.json
`-- edge_outputs.csv
```

Opciones relevantes para recortes con fondo negro o elementos ajenos a la
carga:

```text
--mask-black-background
--foreground-erode 12
--roi-rel "0.08,0.02,0.92,0.86"
--binary-open-iterations 1
--thin-binary
```

## Modelos locales

Actualmente se dispone localmente de:

```text
haul_truck_segment_m_736_low_data_aug_07012025.pt
oversized_rocks_segment_20241205-004357.pt
```

Los archivos de pesos no se suben al repositorio por su tamano. Deben copiarse
manualmente dentro de `models/` al preparar un entorno nuevo.

## Proximas etapas

```text
frame original
-> mascara YOLO de material
-> DexiNed/OpenCV solo dentro de la carga
-> patches validos
-> features de bordes y contornos
-> clasificador fina/media/gruesa
-> agregacion por frame y por camion
```

Las tareas tecnicas siguientes son:

1. Integrar automaticamente las mascaras YOLO con DexiNed.
2. Generar patches internos a la mascara de material.
3. Extraer features de bordes, areas, perimetros y fragmentacion.
4. Crear etiquetas fina/media/gruesa/ambiguo.
5. Entrenar y evaluar Random Forest/SVM como baseline.
6. Agregar resultados temporalmente por secuencia o camion.
7. Incorporar el modelo de oversized rocks como indicador complementario.

## Resultado esperado

```text
Fina: 22%
Media: 47%
Gruesa: 31%
Predominante: media
Indice granulometrico visual: 2.09
Oversized rocks: detectadas/no detectadas
```

Este resultado debe interpretarse como granulometria visual aparente. Obtener
tamanos fisicos reales requiere calibracion geometrica y una referencia de
escala.
