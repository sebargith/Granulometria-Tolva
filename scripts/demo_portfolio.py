"""Offline results report; no model weights or third-party packages required."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from granulometria.metrics import compute_granulometry_metrics

CLASSES = ("fino", "medio", "grueso")


def summarize(labels):
    if any(label not in CLASSES for label in labels):
        raise ValueError("Unknown synthetic label")
    counts = Counter(labels)
    shares = {label: counts[label] / len(labels) if labels else 0 for label in CLASSES}
    return {"count": len(labels), "shares": shares, **compute_granulometry_metrics(shares)}


def fmt(value):
    return "Not evaluable" if value is None else f"{value:.3f}"


def render_report(results, fixture):
    rows = []
    matrices = []
    for experiment in results["experiments"]:
        name = escape(experiment["name"])
        rows.append(f'<tr><th>{name}</th><td>{escape(experiment["unit"])}</td>'
                    f'<td>{experiment["test_f1_macro"]:.3f}</td>'
                    f'<td>{experiment["accuracy"]:.1%}</td></tr>')
        matrix_rows = "".join(
            '<tr><th>' + label + '</th>' + ''.join(f'<td>{n}</td>' for n in row) + '</tr>'
            for label, row in zip(CLASSES, experiment["matrix"])
        )
        matrices.append(f'<details><summary>{name}: confusion matrix</summary><div class="scroll">'
                        '<table><caption>Rows: reference. Columns: prediction.</caption>'
                        '<thead><tr><th>Class</th><th>fino</th><th>medio</th><th>grueso</th></tr></thead>'
                        f'<tbody>{matrix_rows}</tbody></table></div></details>')
    frame_html = []
    pooled = []
    for frame in fixture["frames"]:
        labels = frame["labels"]
        stats = summarize(labels)
        pooled.extend(labels)
        cells = ''.join(f'<div class="cell {label}"><span>{i+1:02d}</span>{label}</div>'
                        for i, label in enumerate(labels))
        frame_html.append(f'<article><h3>{escape(frame["id"])}</h3>'
                          f'<p>{stats["count"]} valid patches · IG {fmt(stats["indice_granulometrico"])}'
                          f' · Dispersion {fmt(stats["dispersion_granulometrica"])}</p>'
                          f'<div class="grid">{cells}</div></article>')
    aggregate = summarize(pooled)
    composition = ' · '.join(f'{label}: {share:.1%}' for label, share in aggregate["shares"].items())
    return '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visual granulometry | public results</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f7f8fa;color:#20252b;font:16px/1.55 system-ui,sans-serif}
main{max-width:1050px;margin:auto;padding:30px 22px}h1{font-size:30px;line-height:1.2}h2{font-size:22px}
header,section{padding:16px 0 24px;border-bottom:1px solid #ccd2d7}h3{font-size:17px;margin-bottom:0}
a{color:#0758a0}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #d7dce0}
th{font-weight:600}caption{text-align:left;padding:8px;color:#515a65}.scroll{overflow-x:auto}
details{margin-top:14px}summary{cursor:pointer;color:#0758a0}.note{border-left:4px solid #c17613;padding:10px 15px;background:#fff4df}
.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:5px;max-width:650px}
.cell{aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:14px;border-radius:3px}
.fino{background:#a9dbb2}.medio{background:#f6df80}.grueso{background:#eda396}article{padding:12px 0}
.numbers{font-weight:600;font-size:18px}footer{padding:20px 0;color:#515a65}
@media(max-width:480px){main{padding:16px}h1{font-size:25px}.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
</style><main><header><h1>Visual granulometry in haul trucks</h1>
<p>YOLO material mask → normalized RGB patches → ResNet18 → frame/video composition.</p>
<nav><a href="#results">Recorded results</a> · <a href="#example">Synthetic example</a> · <a href="#limits">Limits</a></nav>
</header><section id="results"><h2>Recorded experimental results</h2>
<p>July 2026 experiments. This offline report does not run inference.</p>
<div class="scroll"><table><thead><tr><th>Experiment</th><th>Unit</th><th>Macro F1</th><th>Accuracy</th></tr></thead><tbody>
''' + ''.join(rows) + '''</tbody></table></div><p>Classifier test: 357 patches from 14 held-out frames.
Technical test: 90 full frames, 30 per visual class. Different evaluation units.</p>
''' + ''.join(matrices) + '''</section><section id="example"><h2>Synthetic patch example</h2>
<p class="note">Invented labels, not model predictions. No operational imagery or data.</p>
''' + ''.join(frame_html) + f'''<h3>Pooled result</h3><p class="numbers">{composition}</p>
<p>{aggregate["count"]} valid patches · IG {fmt(aggregate["indice_granulometrico"])} · Dispersion {fmt(aggregate["dispersion_granulometrica"])}</p>
<p>Patch counts are pooled, so frames with more valid patches contribute more.
The empty frame contributes no labels. IG ranges from 1 (fine) to 3 (coarse).
Dispersion ranges from 0 (one class) to 1 (equal fine/coarse mixture).</p>
</section><section id="limits"><h2>Interpretation limits</h2>
<p>Surface appearance is not physical particle size. Frame grouping does not
establish independence by video or truck. RF is refit on train+validation; CNN
uses its best validation checkpoint. Scores are uncalibrated and temporal
redundancy can bias aggregation.</p></section><footer>Public portfolio example ·
Source code and full protocol: <a href="https://github.com/sebargith/Granulometria-Tolva">Granulometria-Tolva</a></footer></main></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/portfolio_demo.html")
    args = parser.parse_args()
    results = json.loads((ROOT / "docs/public/results.json").read_text(encoding="utf-8"))
    fixture = json.loads((ROOT / "examples/synthetic_frames.json").read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(results, fixture), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
