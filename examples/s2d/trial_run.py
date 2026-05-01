"""
S2D Reward Function Trial Run
------------------------------
Benchmarks a ladder of classifiers through the same 3-seed macro-F1 evaluation
that the RL reward function uses. Produces a self-contained HTML report with:
  - Score ladder (bar chart)
  - Per-seed variance per classifier
  - Seed-by-seed breakdown heatmap
  - Raw scores table
"""

import sys
import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

TRAIN_CSV = "/Users/pran-ker/Developer/Hexo/Open-SIA/tasks/symptom2disease/data/public/train.csv"
SEEDS = [42, 137, 271]
OUTPUT_HTML = Path(__file__).parent / "trial_report.html"


# ── Classifier definitions ─────────────────────────────────────────────────────

def make_classifiers():
    return [
        (
            "Random Baseline",
            lambda: _RandomBaseline(),
        ),
        (
            "TF-IDF + Naive Bayes",
            lambda: Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=20000)),
                ("clf", MultinomialNB()),
            ]),
        ),
        (
            "TF-IDF (1-2g) + Logistic Regression",
            lambda: Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
            ]),
        ),
        (
            "TF-IDF (1-2g) + Linear SVM",
            lambda: Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
                ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000))),
            ]),
        ),
        (
            "TF-IDF (1-3g) + LR (tuned C=5)",
            lambda: Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=100000, sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=2000, C=5.0, solver="lbfgs")),
            ]),
        ),
        (
            "TF-IDF (1-3g) + SVM (tuned C=5)",
            lambda: Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=100000, sublinear_tf=True)),
                ("clf", CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=3000))),
            ]),
        ),
        (
            "TF-IDF (1-3g) + LR+SVM Ensemble",
            lambda: _build_ensemble(),
        ),
    ]


def _build_ensemble():
    lr = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=100000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0, solver="lbfgs")),
    ])
    svm = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=100000, sublinear_tf=True)),
        ("clf", CalibratedClassifierCV(LinearSVC(C=5.0, max_iter=3000))),
    ])
    return VotingClassifier(estimators=[("lr", lr), ("svm", svm)], voting="soft")


class _RandomBaseline:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self
    def predict(self, X):
        return np.random.choice(self.classes_, size=len(X))


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_classifier(make_clf, df, seeds=SEEDS):
    scores_f1 = []
    scores_acc = []
    for seed in seeds:
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df["label"]
        )
        clf = make_clf()
        clf.fit(train_df["text"], train_df["label"])
        preds = clf.predict(val_df["text"])
        scores_f1.append(f1_score(val_df["label"], preds, average="macro"))
        scores_acc.append(accuracy_score(val_df["label"], preds))
    return scores_f1, scores_acc


def run_all():
    print(f"Loading data from {TRAIN_CSV} ...")
    df = pd.read_csv(TRAIN_CSV)
    print(f"  {len(df)} samples, {df['label'].nunique()} classes\n")

    classifiers = make_classifiers()
    results = []

    for name, make_clf in classifiers:
        print(f"  Evaluating: {name} ...")
        t0 = time.time()
        f1_scores, acc_scores = evaluate_classifier(make_clf, df)
        elapsed = time.time() - t0
        row = {
            "name": name,
            "f1_mean": float(np.mean(f1_scores)),
            "f1_std": float(np.std(f1_scores)),
            "f1_seeds": f1_scores,
            "acc_mean": float(np.mean(acc_scores)),
            "acc_std": float(np.std(acc_scores)),
            "acc_seeds": acc_scores,
            "elapsed_s": round(elapsed, 2),
        }
        results.append(row)
        print(f"    macro F1 = {row['f1_mean']:.4f} ± {row['f1_std']:.4f}  "
              f"acc = {row['acc_mean']:.4f}  ({elapsed:.1f}s)")

    return results


# ── HTML report ────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>S2D Trial Run — Reward Function Benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0; margin: 0; padding: 24px; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; color: #fff; }}
  .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 32px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .card {{ background: #1a1d27; border-radius: 12px; padding: 20px; }}
  .card.wide {{ grid-column: 1 / -1; }}
  h2 {{ font-size: 1rem; color: #aaa; margin: 0 0 16px; text-transform: uppercase;
        letter-spacing: 0.05em; }}
  canvas {{ max-height: 380px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; color: #888; font-weight: 600; padding: 8px 12px;
        border-bottom: 1px solid #2d3044; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #1e2132; }}
  tr:last-child td {{ border-bottom: none; }}
  .best {{ color: #4ade80; font-weight: 700; }}
  .tag {{ display: inline-block; background: #2d3044; border-radius: 4px;
          padding: 1px 6px; font-size: 0.75rem; color: #aaa; margin-left: 6px; }}
  .delta {{ color: #60a5fa; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>S2D Reward Function — Trial Run</h1>
<div class="subtitle">3-seed macro F1 evaluation · {n_clf} classifiers · {n_samples} samples · 24 classes</div>

<div class="grid">

  <div class="card wide">
    <h2>Score Ladder — Mean Macro F1 across 3 seeds</h2>
    <canvas id="ladderChart"></canvas>
  </div>

  <div class="card">
    <h2>Per-seed Variance</h2>
    <canvas id="varianceChart"></canvas>
  </div>

  <div class="card">
    <h2>Seed-by-seed Breakdown</h2>
    <canvas id="seedChart"></canvas>
  </div>

  <div class="card wide">
    <h2>Raw Results Table</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Classifier</th>
          <th>Macro F1 (mean)</th><th>± std</th>
          <th>Accuracy (mean)</th>
          <th>Seed 42</th><th>Seed 137</th><th>Seed 271</th>
          <th>Time (s)</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>

</div>

<script>
const DATA = {data_json};

const names = DATA.map(d => d.name);
const f1means = DATA.map(d => d.f1_mean);
const f1stds = DATA.map(d => d.f1_std);
const accmeans = DATA.map(d => d.acc_mean);
const seeds = [42, 137, 271];
const seedColors = ['#60a5fa','#a78bfa','#34d399'];

const baseColor = (i, alpha=1) => `hsla(${{200 + i * 22}}, 70%, 60%, ${{alpha}})`;

// ── Ladder chart ──────────────────────────────────────────────────────────────
new Chart(document.getElementById('ladderChart'), {{
  type: 'bar',
  data: {{
    labels: names,
    datasets: [
      {{
        label: 'Macro F1',
        data: f1means,
        backgroundColor: names.map((_, i) => baseColor(i, 0.75)),
        borderColor: names.map((_, i) => baseColor(i)),
        borderWidth: 1,
        borderRadius: 6,
      }},
      {{
        label: 'Accuracy',
        data: accmeans,
        backgroundColor: names.map((_, i) => baseColor(i, 0.3)),
        borderColor: names.map((_, i) => baseColor(i, 0.5)),
        borderWidth: 1,
        borderDash: [4, 4],
        borderRadius: 6,
      }},
    ],
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#ccc' }} }},
      tooltip: {{ callbacks: {{
        label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.raw.toFixed(4)}}`
      }} }},
    }},
    scales: {{
      x: {{ ticks: {{ color: '#aaa', maxRotation: 30 }}, grid: {{ color: '#2d3044' }} }},
      y: {{ min: 0, max: 1.05, ticks: {{ color: '#aaa' }}, grid: {{ color: '#2d3044' }} }},
    }},
  }},
}});

// ── Variance chart ────────────────────────────────────────────────────────────
new Chart(document.getElementById('varianceChart'), {{
  type: 'bar',
  data: {{
    labels: names,
    datasets: [{{
      label: 'Std Dev across seeds',
      data: f1stds,
      backgroundColor: names.map((_, i) => baseColor(i, 0.6)),
      borderRadius: 4,
    }}],
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#ccc' }} }},
      tooltip: {{ callbacks: {{
        label: ctx => ` std: ${{ctx.raw.toFixed(5)}}`
      }} }},
    }},
    scales: {{
      x: {{ ticks: {{ color: '#aaa', maxRotation: 30 }}, grid: {{ color: '#2d3044' }} }},
      y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#2d3044' }} }},
    }},
  }},
}});

// ── Per-seed line chart ───────────────────────────────────────────────────────
new Chart(document.getElementById('seedChart'), {{
  type: 'line',
  data: {{
    labels: names,
    datasets: seeds.map((seed, si) => ({{
      label: `Seed ${{seed}}`,
      data: DATA.map(d => d.f1_seeds[si]),
      borderColor: seedColors[si],
      backgroundColor: seedColors[si].replace(')', ', 0.1)').replace('rgb', 'rgba'),
      tension: 0.3,
      pointRadius: 5,
    }})),
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#aaa', maxRotation: 30 }}, grid: {{ color: '#2d3044' }} }},
      y: {{ min: 0, max: 1.05, ticks: {{ color: '#aaa' }}, grid: {{ color: '#2d3044' }} }},
    }},
  }},
}});
</script>
</body>
</html>
"""


def make_table_rows(results):
    best_f1 = max(r["f1_mean"] for r in results)
    rows = []
    prev_f1 = None
    for i, r in enumerate(results):
        is_best = abs(r["f1_mean"] - best_f1) < 1e-6
        f1_class = 'class="best"' if is_best else ""

        delta = ""
        if prev_f1 is not None:
            diff = r["f1_mean"] - prev_f1
            sign = "+" if diff >= 0 else ""
            delta = f' <span class="delta">({sign}{diff:.4f})</span>'
        prev_f1 = r["f1_mean"]

        rows.append(f"""
        <tr>
          <td>{i+1}</td>
          <td>{r['name']}</td>
          <td {f1_class}>{r['f1_mean']:.4f}{delta}</td>
          <td>{r['f1_std']:.5f}</td>
          <td>{r['acc_mean']:.4f}</td>
          <td>{r['f1_seeds'][0]:.4f}</td>
          <td>{r['f1_seeds'][1]:.4f}</td>
          <td>{r['f1_seeds'][2]:.4f}</td>
          <td>{r['elapsed_s']}</td>
        </tr>""")
    return "".join(rows)


def write_report(results, df):
    html = HTML_TEMPLATE.format(
        n_clf=len(results),
        n_samples=len(df),
        table_rows=make_table_rows(results),
        data_json=json.dumps(results),
    )
    OUTPUT_HTML.write_text(html)
    print(f"\nReport written to: {OUTPUT_HTML}")


if __name__ == "__main__":
    results = run_all()
    df = pd.read_csv(TRAIN_CSV)
    write_report(results, df)

    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"{'Classifier':<45} {'F1 mean':>8}  {'± std':>8}  {'Acc':>8}")
    print("─" * 75)
    for r in results:
        print(f"{r['name']:<45} {r['f1_mean']:>8.4f}  {r['f1_std']:>8.5f}  {r['acc_mean']:>8.4f}")

    import subprocess
    subprocess.Popen(["open", str(OUTPUT_HTML)])
