# Granulometria visible en tolva

Proyecto experimental de vision computacional para estimar la granulometria
visual aparente del material visible en la tolva de camiones mineros.

El objetivo no es medir directamente una distribucion fisica en milimetros,
sino construir un indicador visual-operacional basado en segmentacion de la
carga, mapas de bordes, contornos y analisis local.

## Estado actual

Actualmente estan implementados cuatro componentes:

1. Inferencia YOLO-seg para detectar la region visible de `Carga`.
2. Inferencia DexiNed para generar mapas de bordes sobre imagenes o recortes.
3. Generacion trazable y etiquetado manual de patches.
4. Entrenamiento e inferencia de un baseline Random Forest para patches.

La segmentacion YOLO y la generacion de patches ya estan conectadas. El
baseline permite medir si las etiquetas contienen una senal aprendible, pero
todavia se necesita ampliar y balancear el dataset antes de usarlo como modelo
operacional.

## Estructura

```text
.
|-- predict_tolva.py
|-- modelos percepcion/
|   `-- carga_tolva_yolo/
|       |-- predict_carga.py
|       |-- preparar_patches.py
|       `-- README.md
|-- modelos granulometria/
|   `-- clasificador_patches/
|       |-- features.py
|       |-- train_random_forest.py
|       |-- predict_patches.py
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
modelo: models/haul_truck_segment_m_736_low_data_aug_07012025.pt
task: segment
imgsz: 736
class 0: caex
class 1: material
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

La opcion recomendada segmenta primero la clase `material` y genera ventanas
`128x128`, sin solapamiento y con al menos 80% de carga:

```powershell
python "modelos percepcion\carga_tolva_yolo\preparar_patches.py" `
  --source "Data parches test" `
  --run-name data_parches_test_carga `
  --open-labeler
```

Esto produce:

```text
outputs/yolo_carga/data_parches_test_carga/   mascaras y overlays
outputs/patches/data_parches_test_carga/      imagenes, patches.csv y labels.csv
```

Si las mascaras ya existen, tambien se puede ejecutar solamente el generador:

```powershell
python tools\patch_labeler\generate_patches.py `
  --source "Data parches test" `
  --mask-dir "outputs\yolo_carga\experimento\masks" `
  --run-name prueba_patches `
  --patch-size 128 `
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

El acceso directo carga el dataset generado mas reciente de `outputs/patches`,
abre el navegador y usa el entorno Conda `memoria-dexined`.

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

## Entrenamiento del clasificador

El baseline usa caracteristicas de bordes y textura calculadas con OpenCV y un
Random Forest. Solo toma etiquetas `fino`, `medio` y `grueso`; los patches
`mixto`, `invalido` y `ambiguo` quedan fuera del entrenamiento inicial.

```powershell
python "modelos granulometria\clasificador_patches\train_random_forest.py" `
  --dataset "outputs\patches\data_parches_test_carga_128"
```

El split se hace por frame original, no por patch. Cada corrida queda
versionada localmente en:

```text
modelos granulometria/clasificador_patches/runs/vNNN_random_forest/
```

La prueba v002 uso 181 patches utiles de 33 frames. Obtuvo F1 macro 0.402 en
validacion y 0.318 en test. Es una prueba funcional, no un resultado final: la
clase `grueso` tiene solo 24 ejemplos totales y ninguno de los cuatro ejemplos
`grueso` del test fue reconocido. Se requiere recolectar mas frames y balancear
las tres clases.

Para reentrenar conservando los mismos frames de validacion y test, usar el
`group_split.csv` de una corrida anterior mediante `--reuse-split`.

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
models/haul_truck_segment_m_736_low_data_aug_07012025.pt
models/oversized_rocks_segment_20241205-004357.pt
```

`haul_truck...pt`, usando la clase `material`, es el modelo principal para
obtener la mascara de carga. `oversized_rocks...pt` se reserva para detectar
rocas oversized y debe restringirse posteriormente a la mascara de carga para
reducir falsos positivos. `best.pt` se conserva como alternativa.

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

1. Generar patches de mas frames y secuencias independientes.
2. Alcanzar al menos 300-500 ejemplos claros por clase y balancear `grueso`.
3. Mantener fijo el conjunto test mediante `--reuse-split`.
4. Comparar el baseline OpenCV con features calculadas desde DexiNed.
5. Comparar Random Forest con SVM y una CNN liviana sobre edge maps.
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
