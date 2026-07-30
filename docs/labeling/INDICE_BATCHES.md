# Indice de batches de etiquetado

## Batch 001

```text
data/labeling/batches/batch001_grilla96_refs_v2_20260618_usado
```

- 50 frames.
- 1.282 etiquetas compiladas.
- Primera iteracion estable de carga normalizada y grilla de 96 px.
- Dataset procesado: `outputs/patches/grid96_batch001_ai_prompt_v2`.

## Batch 002

```text
data/labeling/batches/batch002_grilla96_data2_mixto_20260624
```

- 90 frames.
- 2.410 etiquetas compiladas.
- Paquetes mezclados por clase de origen.
- Dataset procesado: `outputs/patches/grid96_batch002_data2_mixto_ai`.

## Batch 003

```text
data/labeling/batches/batch003_grilla96_data3_mixto_20260702
```

- 90 frames.
- 2.455 etiquetas revisadas.
- Distribucion final: 710 fino, 917 medio, 823 grueso y 5 invalido.
- Dataset final: `outputs/patches/grid96_batch003_data3_mixto_ai`.
- Es el batch usado para los modelos finales Random Forest y ResNet18.

## Convencion para nuevos batches

```text
batchNNN_grilla96_<fuente>_YYYYMMDD
```

Dentro de cada batch se conservan:

```text
frames_a_etiquetar/
grillas/
respuestas_ia/
labels_revisadas/
frames_manifest.csv
```

Los ZIP de intercambio se pueden eliminar una vez compiladas y verificadas las
respuestas.
