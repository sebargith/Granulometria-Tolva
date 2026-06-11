# AGENTS.md — Contexto operativo para Codex

## Rol del agente

Actúa como asistente de desarrollo para un proyecto de Memoria de Título en PSINET, orientado a visión computacional aplicada a minería. El foco de este repositorio es el subproyecto de **granulometría visible en la tolva de camiones mineros**.

Tu tarea principal es ayudar a implementar, ordenar y documentar un pipeline experimental que procese videos o frames de camiones con carga, detecte la región de carga en la tolva y estime un indicador granulométrico visual basado en bordes, contornos y análisis local.

El usuario trabaja en español. Mantén documentación y comentarios principales en español cuando sea razonable. Los nombres de scripts, variables y funciones pueden ir en inglés si mejora claridad técnica.

\---

## Resumen del proyecto

El proyecto busca estimar la **granulometría visual aparente** del material visible en la tolva de camiones mineros usando imágenes extraídas desde video.

Importante: el objetivo no es medir la granulometría física exacta de todo el volumen cargado. La cámara observa principalmente la superficie visible de la carga, por lo que la salida debe tratarse como un **indicador visual-operacional**.

La idea final es que el pipeline pueda funcionar sobre video o cámara en tiempo real:

```text
cámara/video
→ detección o segmentación de la carga en tolva
→ análisis de bordes/contornos dentro de la carga
→ estimación de proporción fina/media/gruesa
→ resultado por frame, secuencia o camión
```

\---

## Datos disponibles

Los datos provienen de videos almacenados en Nextcloud. La estructura local suele organizarse por carpetas de frames de la forma:

```text
Camera\_8\_YYYYMMDD\_HHMMSS/
```

Ejemplos:

```text
Camera\_8\_20241022\_134013/
Camera\_8\_20241023\_094005/
Camera\_8\_20241025\_175004/
```

Las imágenes son frames de una cámara fija, pero los camiones se mueven dentro del encuadre. Las tomas suelen ser posteriores o superiores/posteriores del camión, mostrando la tolva con carga.

Características visuales observadas:

* cámara fija;
* camión en movimiento;
* cambios de escala según distancia del camión;
* polvo;
* sombras;
* perspectiva variable;
* material superpuesto;
* textura compleja;
* zonas con mezcla de fino, medio y grueso;
* posible timestamp o texto sobre la imagen;
* fondo minero con rocas/camino que puede confundir si no se aísla la carga.

\---

## Decisión metodológica actual

Después de revisar la data, se decidió no partir con segmentación semántica completa fina/media/gruesa sobre toda la carga. Ese enfoque puede ser ambiguo porque las fronteras entre clases no son limpias y una misma zona puede contener material mixto.

El enfoque actual recomendado es:

```text
1. Detectar/segmentar la carga en la tolva.
2. Obtener una máscara de carga\_tolva.
3. Analizar la región de carga mediante DexiNed/OpenCV.
4. Probar análisis global y análisis local por grilla/patches.
5. Extraer features de bordes/contornos.
6. Clasificar o estimar nivel granulométrico visual.
7. Agregar resultados por frame y por camión.
```

El usuario indicó explícitamente que quiere experimentar con métodos basados en DexiNed y que, para el análisis granulométrico, se debe dejar de priorizar RGB como fuente principal. La señal principal debe venir de **bordes, contornos y estructura visual**.

\---

## Paper base relevante

El paper más relevante para este enfoque es:

**Li, H.; Asbjörnsson, G.; Lindqvist, M. “Image Process of Rock Size Distribution Using DexiNed-Based Neural Network”. Minerals, 2021.**

Ideas importantes del paper:

* usa preprocesamiento con escala de grises y CLAHE;
* usa DexiNed para obtener mapas de bordes de rocas;
* aplica binarización, operaciones morfológicas y OpenCV;
* clasifica regiones como finas, rocas conectadas y rocas individuales;
* calcula distribución de tamaño a partir de áreas de contornos;
* valida contra harneo de laboratorio;
* reconoce que la superposición de partículas introduce sesgo.

Diferencia con este proyecto:

* el paper trabaja con muestras de laboratorio más controladas;
* este proyecto trabaja con camiones reales, cámara fija, movimiento, polvo, sombras y material superpuesto;
* por eso no se debe copiar el método tal cual, sino adaptarlo.

\---

## Modelo disponible para detección/segmentación

Existe un modelo entregado por la empresa:

```text
haul\_truck\_segment\_m\_736\_low\_data\_aug\_07012025.pt
```

Probablemente es un modelo YOLO de segmentación entrenado con Ultralytics. Debe probarse para segmentar la región `carga\_tolva` o una región equivalente de la tolva/carga.

Parámetros iniciales sugeridos:

```text
imgsz = 736
conf = 0.25
```

Ajustar `conf` si detecta demasiado ruido o si no detecta nada.

Uso esperado:

```text
frame
→ modelo YOLO-seg
→ máscara carga\_tolva
→ análisis DexiNed/OpenCV dentro de la máscara
```

\---

## Pipeline general recomendado

### Etapa 1 — Organización de datos

Objetivo: ordenar los videos/frames y seleccionar datos útiles.

Tareas:

* leer carpetas `Camera\_8\_fecha\_hora`;
* seleccionar videos o secuencias útiles;
* extraer frames representativos;
* evitar frames redundantes;
* evitar frames muy lejanos, borrosos o con polvo extremo.

Primera iteración sugerida:

```text
60–120 frames base
5–8 frames buenos por carpeta/secuencia
12–15 carpetas si están disponibles
```

\---

### Etapa 2 — Split de datos

El split debe hacerse por carpeta, video o secuencia completa, no por frames aleatorios.

Proporción recomendada:

```text
70% train
15% val
15% test
```

Ejemplo con 15 carpetas:

```text
11 carpetas train
2 carpetas val
2 carpetas test
```

Motivo: evitar fuga de información. Frames cercanos del mismo video son muy similares y no deben repartirse entre train/val/test.

\---

### Etapa 3 — Segmentación de carga

Usar el modelo disponible `haul\_truck\_segment\_m\_736\_low\_data\_aug\_07012025.pt` para generar máscaras automáticas de carga/tolva.

También se pueden usar máscaras manuales de CVAT para validar o mejorar el modelo.

Salida esperada:

```text
outputs/masks/
  <nombre\_frame>\_mask.png
```

La máscara debe representar la región válida de análisis:

```text
carga\_tolva = superficie visible de material dentro de la tolva
```

Debe excluir:

* camino;
* cerros;
* otros camiones;
* bordes irrelevantes de la tolva;
* material externo al camión;
* timestamp si interfiere;
* zonas con polvo extremo u oclusión fuerte.

\---

### Etapa 4 — Experimentos DexiNed

Se deben comparar al menos dos líneas:

#### Método A — DexiNed global sobre carga completa

Flujo:

```text
frame
→ máscara carga\_tolva
→ recorte/enmascarado de carga
→ grayscale + CLAHE
→ DexiNed
→ binarización
→ morfología
→ findContours
→ features globales
→ indicador o clasificación global
```

Este método no requiere patches al inicio. Sirve como baseline rápido y es el más parecido al paper.

Features globales posibles:

* densidad total de bordes;
* número de contornos;
* área promedio de contornos;
* mediana de áreas;
* percentiles de área;
* proporción de contornos pequeños;
* proporción de contornos grandes;
* suma de perímetros;
* relación área/perímetro;
* número de componentes conectados;
* índice de fragmentación visual.

#### Método B — DexiNed local por grilla/patches

Flujo:

```text
frame
→ máscara carga\_tolva
→ dividir región en celdas/patches
→ DexiNed por celda
→ OpenCV features por celda
→ clasificador local
→ fina/media/gruesa/ambiguo
→ agregación por porcentajes
```

Este es el enfoque recomendado para robustez, porque la carga puede contener zonas heterogéneas.

Parámetros iniciales sugeridos:

```text
patch\_size = 128x128 px
stride = 96–128 px
cobertura mínima dentro de máscara = 70%
```

Las celdas o patches dudosos pueden etiquetarse como:

```text
ambiguo / ignorar
```

No usar los ambiguos para entrenar al inicio.

\---

### Etapa 5 — Etiquetado granulométrico

Para el enfoque local, los patches o celdas deben etiquetarse manualmente inicialmente.

Clases:

```text
fina
media
gruesa
ambiguo/ignorar
```

Criterios orientativos:

* `fina`: textura compacta, fragmentos pequeños, pocos bloques distinguibles, superficie más homogénea.
* `media`: fragmentos distinguibles pero sin predominio claro de bloques grandes; textura intermedia.
* `gruesa`: bloques grandes claros, contornos notorios, alta irregularidad, rocas grandes dominantes.
* `ambiguo/ignorar`: sombra fuerte, polvo, mezcla difícil, mala visibilidad, borde de tolva, timestamp, o patch sin predominio claro.

Primera iteración sugerida:

```text
300–500 patches por clase útil
1500–3000 patches útiles totales si es posible
```

No es necesario etiquetar todos los patches generados. Priorizar ejemplos claros y balanceados.

\---

### Etapa 6 — Modelos para clasificación local

Como el usuario quiere experimentar con DexiNed y no priorizar RGB, usar principalmente:

#### Variante 1 — Features de OpenCV + modelo clásico

Input:

```text
features extraídas desde edge maps de DexiNed
```

Modelos:

```text
Random Forest
SVM
Logistic Regression
XGBoost si está disponible
```

Ventaja: interpretable y alineado con el paper.

#### Variante 2 — Edge map + CNN liviana

Input:

```text
edge map DexiNed
```

Modelos:

```text
ResNet18
MobileNetV3
EfficientNet-B0
```

No usar RGB como entrada principal en esta variante; usar el mapa de bordes como imagen de entrada.

\---

### Etapa 7 — Agregación

Para cada frame:

```text
% fina
% media
% gruesa
clase predominante
índice granulométrico visual
```

Índice sugerido:

```text
IG = 1\*p\_fina + 2\*p\_media + 3\*p\_gruesa
```

Para cada camión/secuencia:

```text
promediar resultados de múltiples frames útiles
```

Salida esperada:

```text
Camión/Secuencia X:
- fina: 22%
- media: 47%
- gruesa: 31%
- predominante: media
- índice granulométrico: 2.09
```

\---

## Métricas

### Para modelo YOLO-seg de carga

Si hay ground truth:

```text
IoU de máscara
Dice
mAP de segmentación
revisión visual
```

Si solo se prueba el modelo entregado:

```text
revisión visual de máscaras
área detectada
confianza
fallos por frame/secuencia
```

### Para clasificación local fina/media/gruesa

```text
accuracy
F1 macro
precision por clase
recall por clase
matriz de confusión
```

La matriz de confusión es crítica. Es esperable que:

```text
fina ↔ media
media ↔ gruesa
```

se confundan más que:

```text
fina ↔ gruesa
```

### Para indicador final

```text
estabilidad temporal por secuencia
comparación contra evaluación visual humana
consistencia entre frames cercanos
```

\---

## Consideraciones para tiempo real

La meta final es que el sistema pueda funcionar en video/cámara:

```text
cámara
→ captura frame cada N ms o N frames
→ YOLO-seg detecta carga\_tolva
→ DexiNed/OpenCV analiza carga
→ clasificador estima fina/media/gruesa
→ agregación temporal
→ output en pantalla o archivo
```

Para tiempo real:

* no depender de máscaras manuales;
* usar modelo YOLO-seg para carga;
* usar procesamiento por grilla eficiente;
* limitar número de patches por frame;
* usar resolución controlada;
* ignorar frames sin carga suficiente;
* procesar solo una ROI amplia si es posible.

\---

## Estructura de repositorio sugerida

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── paths.yaml
│   ├── inference.yaml
│   └── experiments.yaml
├── models/
│   └── haul\_truck\_segment\_m\_736\_low\_data\_aug\_07012025.pt
├── data/
│   ├── raw\_videos/
│   ├── frames/
│   ├── splits/
│   ├── masks/
│   ├── patches/
│   └── labels/
├── scripts/
│   ├── predict\_yolo\_masks.py
│   ├── generate\_patches\_from\_masks.py
│   ├── run\_dexined\_global.py
│   ├── run\_dexined\_patches.py
│   ├── extract\_edge\_features.py
│   ├── label\_patches\_helper.py
│   ├── train\_feature\_classifier.py
│   ├── train\_edge\_cnn.py
│   ├── infer\_frame.py
│   └── infer\_video.py
├── src/
│   └── granulometria/
│       ├── io.py
│       ├── yolo\_masks.py
│       ├── patches.py
│       ├── dexined.py
│       ├── features.py
│       ├── classifiers.py
│       ├── aggregation.py
│       └── visualization.py
├── notebooks/
├── outputs/
│   ├── masks/
│   ├── edge\_maps/
│   ├── patches/
│   ├── features/
│   ├── predictions/
│   └── reports/
└── runs/
```

\---

## Convenciones de implementación

* No hacer cambios destructivos sin necesidad.
* Mantener scripts simples, ejecutables desde CLI.
* Usar `argparse` en scripts.
* Guardar outputs con estructura reproducible.
* No hardcodear rutas absolutas si se puede usar config.
* Registrar parámetros importantes en archivos `.json` o `.yaml`.
* Guardar visualizaciones de control:

  * imagen original;
  * máscara de carga;
  * edge map;
  * contornos;
  * grilla/patches;
  * predicción fina/media/gruesa.
* Siempre generar outputs verificables visualmente.

\---

## Primeras tareas recomendadas para Codex

1. Crear `requirements.txt`.
2. Crear script para probar el modelo YOLO `.pt` sobre una carpeta de frames.
3. Guardar máscaras binarias.
4. Crear visualizaciones overlay de máscara sobre imagen.
5. Crear script de generación de patches dentro de máscara.
6. Crear script para correr DexiNed sobre una imagen o carpeta.
7. Crear extractor de features desde edge maps.
8. Crear CSV de features por patch/frame.
9. Crear entrenamiento de Random Forest/SVM.
10. Crear pipeline de inferencia por frame.
11. Crear pipeline de inferencia por video.

\---

## No hacer por ahora

No priorizar por ahora:

* segmentación semántica completa fina/media/gruesa con U-Net/DeepLabV3+;
* segmentación roca por roca;
* estimación física exacta de PSD;
* cálculo de tamaño real en mm sin calibración geométrica;
* depender de RGB como señal principal;
* mezclar frames del mismo video entre train/val/test.

\---

## Resumen final para el agente

El mejor approach actual es:

```text
YOLO-seg para detectar carga\_tolva
+
DexiNed/OpenCV para analizar bordes y contornos
+
comparación global vs grilla/patches
+
clasificador local o features interpretables
+
agregación por porcentaje e índice granulométrico
+
salida por frame/secuencia/camión
```

El objetivo es construir un pipeline experimental robusto, trazable y eventualmente utilizable en tiempo real para estimar granulometría visual aparente en tolvas de camiones mineros.

