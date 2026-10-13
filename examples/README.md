# Public demo

```sh
python scripts/demo_portfolio.py --output outputs/portfolio_demo.html
```

Open the generated HTML locally. No server or dependencies are required. Paths
are resolved relative to the script, so execution also works outside the clone.

Two kinds of evidence are deliberately distinguished:

- **Recorded evaluation:** confusion counts and aggregate metrics transcribed
  from historical experiments in `docs/public/results.json`.
- **Synthetic example:** invented patch labels in `synthetic_frames.json`.
  The demo calls the actual index/dispersion function and demonstrates pooling
  valid patches across frames, including a frame with no valid patches.

The example does not segment imagery, classify rocks or reproduce model accuracy.
It contains no operational images, filenames, timestamps, weights or annotations.
For full inference use the [quickstart](../docs/guides/QUICKSTART.md).
