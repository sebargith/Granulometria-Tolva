# Etiquetado por grilla normalizada

El flujo vigente presenta a la IA la carga completa normalizada con una grilla
numerada. Cada respuesta contiene una fila por celda y conserva frame, posicion,
clase, confianza y notas.

## Rutas

```text
data/labeling/references/                         referencias fino/medio/grueso
data/labeling/references_normalized_20260618_203838/
data/labeling/batches/                            batches terminados
scripts/labeling/prepare_grid_ai_packages.py
scripts/labeling/compile_grid_ai_labels.py
docs/labeling/PROMPT_ETIQUETADO_IA_GRILLA.md
```

Se eliminaron los ZIP de intercambio de batches ya compilados. Se conservaron
los frames normalizados, grillas, respuestas, etiquetas revisadas y manifiestos.

El metodo antiguo que enviaba patches individuales se retiro del espacio activo
porque fue reemplazado por el etiquetado con contexto de carga completa.
