# Granulometria visible en tolva

Proyecto experimental de vision computacional para estimar la granulometria
visual aparente del material visible en la tolva de camiones mineros.

El objetivo no es medir directamente una distribucion fisica en milimetros,
sino construir un indicador visual-operacional basado en segmentacion de la
carga, mapas de bordes, contornos y analisis local.

## Estado actual

Actualmente estan implementados tres componentes:

1. Inferencia YOLO-seg para detectar la region visible de `Carga`.
2. Inferencia DexiNed para generar mapas de bordes sobre imagenes o recortes.
3. Generacion trazable y etiquetado manual de patches.

Todavia falta conectar automaticamente las mascaras YOLO con los patches,
extraer features y entrenar el clasificador fina/media/gruesa.

## Estructura

```text
.
|-- predict_tolva.py
|-- modelos percepcion/
|   `-- carga_tolva_yolo/
|       |-- best.pt              # peso local, ignorado por Git
|       |-- predict_carga.py
|       `-- README.md
|-- scripts/
|   `-- run_dexined_edges.py
|-- tools/
|   `-- patch_labeler/
|       |-- app.py
|       |-- generate_patches.py
|       |-- patches.py
|       |-- launcher.py
|       |-- abrir_etiquetador.cmd
|       `-- assets/patch_labeler.ico
|-- src/granulometria/
|   `-- dexined.py
|-- Etiquetar parches.lnk
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

El modelo de percepcion principal esta encapsulado en:

```text
modelos percepcion/carga_tolva_yolo/
```

Metadata verificada:

```text
modelo: best.pt
task: segment
imgsz: 640
class 0: Carga
```

Las imagenes originales se organizan por experimento:

```text
input/Data test 1/
input/Data test 2/
```

Ejemplo recomendado:

```powershell
python predict_tolva.py `
  --source "input\Data test 2" `
  --conf 0.25
```

`predict_tolva.py` conserva compatibilidad con los comandos anteriores y
ejecuta internamente:

```text
modelos percepcion/carga_tolva_yolo/predict_carga.py
```

Si se omite `--run-name`, la carpeta de salida toma el nombre del input. Si ya
existe, se agrega fecha y hora para no sobrescribir resultados.

```text
outputs/yolo_carga/data_test_2/
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

## Generacion de patches

Para generar ventanas `256x256` con 50% de solapamiento desde recortes de
carga:

```powershell
python tools\patch_labeler\generate_patches.py `
  --source "D:\ruta\a\recortes_de_carga" `
  --run-name prueba_patches `
  --patch-size 256 `
  --stride 128 `
  --min-valid-fraction 0.8
```

Si los recortes tienen fondo negro, agregar:

```text
--mask-black-background --foreground-threshold 8
```

Tambien se puede usar `--mask-dir` para aceptar solamente ventanas cubiertas
por mascaras binarias. La salida contiene las imagenes y su trazabilidad:

```text
outputs/patches/prueba_patches/
|-- images/
|-- params.json
`-- patches.csv
```

## Etiquetado de patches

La aplicacion local permite asignar `fino`, `medio`, `grueso`, `mixto`,
`invalido` o `ambiguo`, junto con confianza y notas:

La forma mas simple de abrirla es hacer doble clic en:

```text
Etiquetar parches.lnk
```

El acceso directo carga por defecto `Data parches test`, abre el navegador y
usa el entorno Conda `memoria-dexined`.

Desde PowerShell se puede ejecutar con:

```powershell
python tools\patch_labeler\launcher.py `
  "outputs\patches\prueba_patches"
```

Tambien se puede ejecutar Streamlit directamente:

```powershell
streamlit run tools\patch_labeler\app.py -- `
  --dataset "Data parches test"
```

Las decisiones se guardan inmediatamente en `labels.csv`. Si se usa un
`patches.csv`, la interfaz muestra ademas el frame original con la ubicacion
de la ventana.

Criterios iniciales:

```text
fino:     particulas pequenas sin limites individuales claros
medio:    fragmentos distinguibles de tamano intermedio
grueso:   predominio de bloques grandes visibles
mixto:    no existe una clase dominante
invalido: fondo, tolva, desenfoque, polvo o iluminacion inutilizable
ambiguo:  no es posible decidir con suficiente confianza
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
modelos percepcion/carga_tolva_yolo/best.pt
haul_truck_segment_m_736_low_data_aug_07012025.pt
oversized_rocks_segment_20241205-004357.pt
```

`best.pt` es el modelo principal para obtener la mascara de `Carga`. Los pesos
de `models/` se conservan como modelos anteriores o complementarios.

Los archivos de pesos no se suben al repositorio por su tamano.

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
2. Generar el dataset definitivo de patches dentro de la mascara de material.
3. Etiquetar patches de alta confianza y revisar el balance por clase.
4. Extraer features de bordes, areas, perimetros y fragmentacion.
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
