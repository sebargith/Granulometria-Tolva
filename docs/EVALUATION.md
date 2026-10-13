# Evaluation and experimental evidence

## Dataset and labels

Reviewed Batch003: 2,455 annotations from 90 source frames. Five invalid records
and 22 low-confidence labels are excluded, leaving 2,428 RGB patches: 704 fine,
903 medium, 821 coarse. Annotation confidence is a filter, not a continuous loss
weight. Numbered full-load grids, visual references and AI assistance were followed
by human review. The labels describe visual appearance, not laboratory sizes.

| Split | Frames | Patches |
|---|---:|---:|
| Train | 63 | 1,706 |
| Validation | 13 | 365 |
| Test | 14 | 357 |

Grouping uses `source_image`; patches of a frame stay together. This does not
guarantee separation of videos, trucks or near-duplicate scenes. Dataset IDs do
not detect differently named copies across datasets.

## Model comparison

| Model | Validation macro F1 | Test macro F1 | Test accuracy |
|---|---:|---:|---:|
| OpenCV + RF | 0.670 | 0.697 | 0.692 |
| RGB + ResNet18 | 0.730 | 0.757 | 0.754 |

Both share held-out frames. RF selects a candidate on validation and refits on
train+validation. ResNet18 retains the best validation checkpoint (epoch 15),
without refitting on validation. This is the historical comparison, not an
identical-training-budget experiment.

Rows are reference labels; prediction columns are fine / medium / coarse:

| Reference | RF | CNN |
|---|---|---|
| Fine | 80 / 21 / 5 | 93 / 12 / 1 |
| Medium | 28 / 88 / 15 | 37 / 84 / 10 |
| Coarse | 3 / 38 / 79 | 1 / 27 / 92 |

## Full pipeline technical test

90 full frames, 30 per dominant visual class, all evaluable: 69/90 correct
(76.7%), macro F1 0.763. This is frame-level agreement, unlike the patch-level
tests. The two sets of 90 frames serve different roles.

![Technical confusion matrix](public/technical_confusion.png)

Fine recall 86.7%; medium 60.0%; coarse 83.3%. Twenty of 21 errors are between
adjacent classes. Medium shares visual cues with both extremes; these counts
alone do not establish why an individual case failed. Evaluability is not a mask
quality guarantee. No exact filename overlap was reported historically, which
does not establish video/truck independence.

## Public evidence

[`results.json`](public/results.json) contains aggregate counts and classifier
metrics, without source identifiers. Technical F1 is derived from the published
confusion counts. The arithmetic can be reproduced without private data;
reproducing training requires the private artifacts.

```sh
python scripts/reporting/build_portfolio_figures.py
python scripts/demo_portfolio.py
```

Figures use aggregate counts or synthetic labels, never operational imagery.

## Next evaluation

1. Collect video/truck/session identifiers; check near-duplicates before splitting.
2. Hold out whole independent groups; freeze models and thresholds before testing.
3. Compare RF/CNN on the same groups and document fitting budgets.
4. Report per-class metrics, coverage and performance by lighting, distance and dust.
5. Evaluate calibration, abstention and latency on named hardware. Compare pooled
   patch aggregation with uniform-frame/robust alternatives. Estimate uncertainty
   by truck/video, not by correlated patches.

Independent-video and production validation remain future work.
