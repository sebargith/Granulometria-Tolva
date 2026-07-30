# Clasificadores de patches

Dataset actual:

```text
outputs/patches/grid96_batch003_data3_mixto_ai
```

Cada muestra es un patch RGB de 96 x 96 px con clase `fino`, `medio` o
`grueso`. El split se realiza por frame para impedir fuga de informacion.

## ResNet18

```powershell
python "modelos granulometria\clasificador_patches\train_resnet.py" `
  --dataset "outputs\patches\grid96_batch003_data3_mixto_ai" `
  --run-name "resnet18_rgb_grid96_nuevo" `
  --image-size 96 `
  --batch-size 64 `
  --freeze-epochs 8 `
  --finetune-epochs 8 `
  --device cuda
```

Modelo final actual:

```text
runs/resnet18_rgb_grid96_batch003_revisado_cuda_20260702/model.pt
```

## Random Forest

```powershell
python "modelos granulometria\clasificador_patches\train_random_forest.py" `
  --dataset "outputs\patches\grid96_batch003_data3_mixto_ai" `
  --run-name "rf_grid96_nuevo" `
  --n-jobs 1
```

Baseline final:

```text
runs/rf_grid96_batch003_revisado_20260702/model.joblib
```

Los runs historicos conservan metricas, matrices, parametros y splits. Sus pesos
se retiraron del espacio activo porque pueden reproducirse desde los datasets.
