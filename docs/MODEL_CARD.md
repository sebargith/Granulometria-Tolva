# ResNet18 RGB model card

**Task:** visual fine / medium / coarse classification. **Status:** experimental.

## Inputs

YOLO material class 1 -> mask -> crop -> width 896 px, preserved aspect ratio ->
black background outside mask -> RGB 96 x 96 patches, stride 96, minimum mask
coverage 0.80 -> ImageNet channel normalization. Terminal grid positions can
overlap at image boundaries; each visible pixel need not receive exactly one vote.

## Training

ImageNet ResNet18 with three-class head; eight frozen-backbone epochs (LR 1e-3),
eight fine-tuning epochs (LR 1e-5), AdamW, weight decay 1e-4. Weighted cross entropy,
training-only flips and color jitter. Best validation macro F1 checkpoint, not
last epoch. Reviewed Batch003: 2,428 eligible patches, source-frame split, seed 42.
See [evaluation](EVALUATION.md) for counts and protocol differences.

## Outputs

- Patch class and softmax scores; scores are not calibrated accuracy.
- Frame shares, dominant class and dominance margin.
- Index `IG = p_fine + 2*p_medium + 3*p_coarse`, range [1,3].
- Dispersion `sqrt(sum(p_class * (score_class - IG)^2))`, range [0,1].
- Video shares pool valid patch counts across extracted frames.

These describe visible surface appearance, not millimetres, mass or volume.
The same IG can describe different mixtures; inspect shares and dispersion together.

## Failure modes

Lighting, dust, occlusion, perspective and mask contamination affect predictions.
Width normalization assumes material spans the visible bed width. Upsampling
cannot restore distant detail. Medium is the weakest technical-test class.
Frame splits do not establish video/truck independence. A calibrated confidence
threshold and validated abstention policy have not been established.

## Provenance

The author trained the YOLO segmentation model as part of the thesis project
in collaboration with PSINet. The same trained model is shared with Cono de Carga,
with separate inference configurations and analytic branches. This repository
also develops the granulometry stage and integration. Data comes from
Collahuasi in a thesis collaboration with PSINET and Universidad de Concepción.
[Artifact hashes](guides/ARTEFACTOS_EXTERNOS.md) · [Publication boundary](PUBLICATION.md).
