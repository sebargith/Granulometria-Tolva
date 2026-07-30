# AGENTS.md - Contexto operativo de granulometria

## Objetivo

Este repositorio implementa un analitico de granulometria visual aparente para
la superficie de carga visible en tolvas de camiones CAEX. Los datos provienen
de la faena Collahuasi y el trabajo se desarrolla en el contexto de una Memoria
de Titulo junto a PSINET y la Facultad de Ingenieria de la Universidad de
Concepcion.

La salida es un indicador visual-operacional. No representa una medicion fisica
en milimetros ni la granulometria de todo el volumen cargado.

## Pipeline vigente

```text
video o frame
-> YOLO-seg de carga
-> mascara de material
-> normalizacion a 896 px de ancho
-> patches RGB de 96 x 96 px, stride 96
-> ResNet18 fino / medio / grueso
-> agregacion por frame y video
-> porcentajes, dominante, indice, dispersion, confianza y overlays
```

Configuracion principal:

- modelo YOLO: `modelos percepcion/carga_tolva_yolo/haul_truck_segment_m_736_low_data_aug_07012025.pt`;
- clasificador: `modelos granulometria/clasificador_patches/runs/resnet18_rgb_grid96_batch003_revisado_cuda_20260702/model.pt`;
- baseline: `modelos granulometria/clasificador_patches/runs/rf_grid96_batch003_revisado_20260702/model.joblib`;
- dataset final: `outputs/patches/grid96_batch003_data3_mixto_ai`;
- clases de entrenamiento: `fino`, `medio`, `grueso`;
- split: por frame o secuencia, nunca por patches aleatorios.

Las rutas compartidas se definen en `src/granulometria/paths.py`.

## Datos

```text
data/raw/evidencias_caex/
data/datasets/patch_frames/
data/evaluation/prueba_tecnica/
data/labeling/
data/legacy/
```

`data/legacy/` contiene material historico de DexiNed, datasets previos de
PSINET y entradas antiguas. No debe usarse como fuente por defecto para nuevos
entrenamientos.

## Decisiones metodologicas

- La normalizacion por ancho reduce el cambio de escala causado por el
  acercamiento del camion.
- La grilla de 96 px mantiene contexto local suficiente y fue la configuracion
  utilizada para el mejor modelo.
- ResNet18 RGB es el modelo principal actual.
- Random Forest se conserva como baseline interpretable.
- DexiNed fue explorado, pero sus bordes resultaron densos e incompletos bajo
  sombras, textura interna y superposicion. Se conserva como linea historica,
  no como entrada principal.
- `ambiguo` e `invalido` se usan para revision, pero se excluyen del
  entrenamiento de tres clases.
- La confianza de etiquetado sirve para auditoria y filtrado del dataset; no es
  una clase objetivo.

## Resultados de referencia

- ResNet18 Batch 003 revisado: F1 macro test 0,757; accuracy 0,754.
- Random Forest Batch 003 revisado: F1 macro test 0,697; accuracy 0,692.
- Prueba tecnica de 90 frames: accuracy 76,7%; F1 macro 0,763.

Las metricas internas deben interpretarse junto con una evaluacion externa por
camion. No comparar experimentos con splits diferentes como si fueran una sola
tabla de ranking.

## Indicadores

Por frame y video se reportan:

- proporciones de fino, medio y grueso;
- clase dominante;
- indice granulometrico visual:
  `IG = 1*p_fino + 2*p_medio + 3*p_grueso`;
- dispersion granulometrica como desviacion estandar ponderada respecto de IG;
- confianza media de los patches;
- cobertura, frames evaluables y tiempo de inferencia.

La agregacion de video combina los patches validos de sus frames. Se debe evitar
que muchos frames casi identicos dominen el resultado temporal.

## Componentes

- `tools/patch_labeler/app.py`: webapp Streamlit de etiquetado, revision y video.
- `modelos granulometria/inferencia_tolva/`: normalizacion e inferencia.
- `modelos granulometria/clasificador_patches/`: entrenamiento RF y ResNet18.
- `modelos percepcion/carga_tolva_yolo/`: inferencia YOLO de carga.
- `src/granulometria/`: metricas, rutas compartidas y DexiNed.
- `scripts/labeling/`: preparacion y compilacion del etiquetado IA.
- `scripts/reporting/`: generadores del informe y reportes.
- `tests/`: pruebas automatizadas.

## Documentacion

- informe vigente: `docs/informe_final/Capitulo_Granulometria_Informe_Final.docx`;
- figuras del informe: `docs/informe_final/assets/`;
- referencias del informe: `docs/informe_final/referencias/`;
- comandos actuales: `docs/guides/USO_ACTUAL.md`;
- presentaciones: `docs/presentations/`;
- referencias: `docs/references/`;
- batches y prompt: `docs/labeling/`.

Al cambiar una ruta o configuracion, actualizar:

1. `src/granulometria/paths.py`;
2. los README afectados;
3. los launchers;
4. los scripts de informe;
5. las rutas consignadas dentro del informe final.

## Convenciones

- Usar rutas relativas al repositorio cuando sea posible.
- Guardar parametros y metricas en JSON/CSV.
- Mantener outputs visuales auditables.
- No versionar datos, pesos, cachés, ZIP de intercambio ni corridas pesadas.
- No mezclar train, validation y test por patch.
- No borrar los modelos finales o el dataset revisado sin una version
  reemplazante verificada.
- Ejecutar `python -m unittest discover -s tests -v` y un smoke test de inferencia antes de cerrar cambios de
  pipeline.
