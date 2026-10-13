# Granulometría visual en tolva

[English](README.md) · [Evaluación](docs/EVALUATION.md) · [Instalación](docs/guides/QUICKSTART.md)

Pipeline experimental para estimar la proporción visual de material **fino, medio
y grueso** en la superficie visible de la carga de camiones mineros. Integra
segmentación YOLO, normalización, parches RGB, ResNet18 y agregación por frame y
video en una aplicación Streamlit. No mide tamaños en milímetros ni la
granulometría física de todo el volumen transportado.

## Resultados

| Evaluación | Unidad | F1 macro | Exactitud |
|---|---|---:|---:|
| OpenCV + Random Forest | 357 parches de test | 0,697 | 69,2 % |
| RGB + ResNet18 | Los mismos 357 parches | **0,757** | **75,4 %** |
| Pipeline completo | 90 frames de prueba técnica | 0,763 | 76,7 % |

Dataset revisado: 2.455 registros; 2.428 parches útiles de 90 frames tras filtrar
clase y confianza. Partición por frame. La prueba técnica tiene otros 90 frames
con referencia visual; no demuestra independencia por camión/video.

![Matriz de confusión de la prueba técnica](docs/public/technical_confusion.png)

RF se reajusta con train + validación tras seleccionar parámetros; la CNN conserva
el mejor checkpoint de validación. [Detalles y limitaciones](docs/EVALUATION.md).

## Flujo y aportes

Video → frames → máscara YOLO → ancho 896 px → parches RGB 96 × 96 → ResNet18 →
proporciones, dominante, índice y dispersión por frame/video.

El desarrollo abarca normalización, extracción trazable de parches, etiquetado
asistido y revisión humana, comparación RF/CNN, indicadores, agregación temporal
e interfaz. El segmentador fue proporcionado por PSINET y constituye la base
común con [Cono de Carga](https://github.com/sebargith/Cono-de-Carga).

Trabajo de memoria de título con PSINET y Universidad de Concepción, con datos de
Collahuasi. Estado: prototipo experimental local.

## Demo sin datos privados

```sh
python scripts/demo_portfolio.py --output outputs/portfolio_demo.html
```

Abra el HTML generado. Muestra métricas históricas y mapas sintéticos claramente
identificados; no ejecuta inferencia ni necesita pesos, GPU o servidor.

![Ejemplo sintético, no predicción del modelo](docs/public/synthetic_patches.png)

Inferencia real: [instalación](docs/guides/QUICKSTART.md),
[comandos operativos](docs/guides/USO_ACTUAL.md) y
[artefactos externos](docs/guides/ARTEFACTOS_EXTERNOS.md).

La clase media presenta mayor dificultad. La confianza softmax no está calibrada
y los frames correlacionados pueden sesgar la agregación. El próximo experimento
debe separar camiones/videos completos y evaluar condiciones nuevas.
[Ficha del modelo](docs/MODEL_CARD.md).

Los documentos internos e imágenes operacionales se conservan localmente.
La limpieza actual no elimina commits antiguos: [alcance de publicación](docs/PUBLICATION.md).
