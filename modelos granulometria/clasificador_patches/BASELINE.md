# Resultado baseline v002

Fecha: 2026-06-12

Dataset local:

```text
outputs/patches/data_parches_test_carga_128
```

Se usaron 181 patches etiquetados como `fino`, `medio` o `grueso`, provenientes
de 33 frames. La division fue por frame original:

| Split | Frames | Patches | Fino | Medio | Grueso |
|---|---:|---:|---:|---:|---:|
| Train | 23 | 127 | 62 | 49 | 16 |
| Validacion | 5 | 25 | 11 | 10 | 4 |
| Test | 5 | 29 | 15 | 10 | 4 |

Metricas del modelo seleccionado `rf_depth12_leaf2`:

```text
F1 macro validacion: 0.402
Accuracy validacion: 0.440
F1 macro test:       0.318
Accuracy test:       0.414
```

Matriz de confusion de test, filas reales y columnas predichas:

| Real | Fino | Medio | Grueso |
|---|---:|---:|---:|
| Fino | 5 | 7 | 3 |
| Medio | 2 | 7 | 1 |
| Grueso | 0 | 4 | 0 |

El pipeline de entrenamiento e inferencia queda validado tecnicamente, pero el
modelo no debe tratarse como operativo. La clase `grueso` esta subrepresentada
y el test contiene pocos frames. El siguiente experimento debe agregar mas
secuencias independientes, priorizar ejemplos claros `grueso` y mantener fijo
el `group_split.csv` de v002.

