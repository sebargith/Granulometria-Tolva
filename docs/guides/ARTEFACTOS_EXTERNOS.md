# Artefactos externos

Los datos, pesos y corridas pesadas no se almacenan en Git. Para ejecutar el
pipeline completo, deben restaurarse en las rutas indicadas debajo de la raiz
del repositorio.

## Modelos seleccionados

| Componente | Ruta esperada | Tamano | SHA-256 |
|---|---|---:|---|
| YOLO-seg de carga | `modelos percepcion/carga_tolva_yolo/haul_truck_segment_m_736_low_data_aug_07012025.pt` | 54.829.963 bytes | `BD7C9880D9A39B291DC03669DAD774DB9FD933D6026DF694EEE1E3A41CBF1DAB` |
| ResNet18 RGB | `modelos granulometria/clasificador_patches/runs/resnet18_rgb_grid96_batch003_revisado_cuda_20260702/model.pt` | 44.787.019 bytes | `9759D0FCC416985DD8F34DF40C5C4141B4172AD1AD0C5EFC0D39D77D2BE79549` |
| Random Forest | `modelos granulometria/clasificador_patches/runs/rf_grid96_batch003_revisado_20260702/model.joblib` | 18.496.894 bytes | `9EA6850F6542361BF91521546D0ABE88CEDB9A4091ECA022671997E2E434C561` |

El clasificador ResNet18 es el modelo principal. Random Forest se conserva como
baseline y el modelo YOLO se usa para aislar la carga visible.

## Dataset seleccionado

El dataset final revisado se espera en:

```text
outputs/patches/grid96_batch003_data3_mixto_ai/
```

Su archivo `labels.csv` contiene 2.455 registros revisados; 2.428 corresponden
a las clases de entrenamiento `fino`, `medio` y `grueso`. El SHA-256 de la
version local utilizada en los modelos finales es:

```text
FCA804A3C10890476F066412283421D5D0BB02F9216DA6679940BB16C70A41CB
```

El split se realiza por frame: 63 frames de entrenamiento, 13 de validacion y
14 de prueba. No debe reconstruirse mediante una division aleatoria por patch.

## Datos de evaluacion

La prueba tecnica externa se espera en:

```text
data/evaluation/prueba_tecnica/
```

Contiene 90 frames, con 30 referencias visuales para cada clase dominante:
`fino`, `medio` y `grueso`.

## Verificacion de archivos

En PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 "<ruta-del-archivo>"
```

Los hashes permiten confirmar que los pesos restaurados corresponden a los
modelos usados para las metricas documentadas.
