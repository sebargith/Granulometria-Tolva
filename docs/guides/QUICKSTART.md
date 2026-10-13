# Quickstart and reproducibility

## 1. Clone and run the public demo

```sh
git clone https://github.com/sebargith/Granulometria-Tolva.git
cd Granulometria-Tolva
python scripts/demo_portfolio.py --output outputs/portfolio_demo.html
```

Python 3.10/3.11, no dependencies. Open the HTML in a browser. This is an offline
results viewer plus synthetic aggregation example, not model inference.

## 2. Full local environment

PowerShell, from the cloned repository root:

```powershell
conda create -n memoria-dexined python=3.10 -y
conda activate memoria-dexined
python -m pip install -r requirements.txt -c constraints-runtime.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The name `memoria-dexined` is retained for compatibility with the Windows launcher;
the selected classifier is ResNet18. An alternative environment can run the Python
scripts and `python -m streamlit` directly.

The [constraints snapshot](../../constraints-runtime.txt) records direct package
versions in the local Windows/Python 3.10.20 runtime. It is not a transitive lock
or a claim of testing every OS/GPU. The local torch build is `2.11.0+cu128`, with
torchvision `0.26.0+cu128`. Install matching PyTorch wheels for the chosen CPU/CUDA
platform and driver. The document-generation dependency is optional to inference
and not part of the measured ML runtime snapshot.

CPU inference: `--device cpu --patch-device cpu`.
GPU inference: YOLO `--device 0`, CNN `--patch-device cuda`.
RF is CPU-based. No performance target is asserted for CPU execution.

## 3. Restore authorized artifacts

Full inference requires the YOLO checkpoint and either CNN or RF weights at the
paths in [ARTEFACTOS_EXTERNOS.md](ARTEFACTOS_EXTERNOS.md). Obtain them through the
project's authorized handover; no public download is offered. Verify checksums
before using the documented results as a reference.

Training additionally requires `images/`, `patches.csv`, `labels.csv` for each
dataset. Keep source-frame names globally identifiable. Do not combine renamed
copies or reserve validation/test frames and then feed them back as training data.

## 4. Image inference

Place your authorized input frames in `input/my_frames`, then:

```powershell
python "modelos granulometria/inferencia_tolva/predict_granulometria_tolva.py" `
  --source "input/my_frames" `
  --run-name "cnn_my_frames" `
  --recursive `
  --device cpu `
  --patch-device cpu
```

Defaults select the final reviewed CNN, width 896 and 96 px patches. Results go
to `outputs/granulometria_tolva/`; an existing run name gets a unique suffix.
Use `--patch-model <path>` to test a candidate without changing project defaults.

## 5. Review labels or process a video

With a restored compiled patch dataset:

```powershell
python tools/patch_labeler/launcher.py "outputs/patches/grid96_batch003_data3_mixto_ai"
```

The launcher uses a free local port starting at 8501. In the sidebar choose
`Pipeline de video`, select your file and model paths, and execute. Video runs
are saved under `outputs/video_pipeline/`. Label corrections affect `labels.csv`
in the opened dataset; archive a snapshot before training.

## 6. Continue model development

Prepare new frames -> normalize/grids -> packages -> validate responses -> human
review -> freeze dataset/split -> train -> independent evaluation. Detailed
commands: [USO_ACTUAL.md](USO_ACTUAL.md). The compiler reads `labels_batch*.csv`
inside `respuestas_ia`; retain the expected IDs and column names.

Always pass `--dataset`: trainer defaults include historical experiments.
Both trainers accept multiple dataset paths. `--reuse-split group_split.csv`
preserves known assignments and puts unseen groups into **train**, not a new
holdout. This is useful for controlled expansion but does not evaluate new-domain
generalization. Reserve new independent videos separately for that purpose.

RF selects on validation, then refits train+validation. CNN selects the best
validation checkpoint and starts from ImageNet on each new run; there is no
resume-from-project-checkpoint option. Use `--dry-run` on the CNN trainer to
inspect the sample/split counts without training. Do not select hyperparameters
repeatedly against test results.

## Tests

Standard library tests (metrics, public evidence and demo):

```sh
python -m unittest discover -s tests -v
```

CPU integration tests (actual normalization, labels, grouped split and summaries):

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests/integration -v
```

Integration tests use synthetic images and CSVs, no model downloads or training.
They are not a substitute for the private real-model smoke test. CI runs both
levels and generates the offline report from a clean checkout.

## Common failures

- Missing weights/data: restore the manifest paths or use the public demo.
- RF argument error: the option is `--jobs`, not `--n-jobs`.
- GPU unavailable: confirm environment and wheel/driver compatibility, or use CPU.
- CUDA memory exhausted: reduce training batch size, keeping patch size unchanged.
- Missing classes in holdouts: add independent frames and inspect the split.
- Invalid AI responses: inspect `labels_revisadas/validation_report.json`.
- Package regeneration: `--overwrite` can erase prior response/review directories;
  create a new batch directory instead when preserving completed work.
