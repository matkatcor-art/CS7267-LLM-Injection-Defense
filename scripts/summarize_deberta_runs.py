"""Summarize the B2 DeBERTa runs in MLflow as mean ± std across seeds, per row.

Reads every finished deberta_<row> run in the phase3-detectors experiment (latest run per row
and seed, so a re-logged seed counts once), adds the tfidf and perplexity_b0 runs for
reference, and logs one deberta_seed_summary run with per-row metrics and summary.md. The same
summary is written to <out-dir>/b2_seed_summary.md and .csv so the repo keeps a copy.

Usage (repo root, with the MLflow server running):
    uv run python scripts/summarize_deberta_runs.py            # log the run and write the files
    uv run python scripts/summarize_deberta_runs.py --no-log   # only write the files
"""

import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--out-dir", default="report/baselines", help="Where to write the summary files")
parser.add_argument("--no-log", action="store_true", help="Write the files without logging a run")
args = parser.parse_args()

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
EXP = mlflow.get_experiment_by_name("phase3-detectors").experiment_id
METRICS = [
    "auroc/test/all", "tpr/test/all", "fpr/test/all", "tpr_at_matched_fpr/test/all",
    "auroc/mcptox/all", "tpr/mcptox/all", "length_binned_auroc/mcptox/all",
    "auroc/msb/all", "tpr/msb/all",
]

df = mlflow.search_runs(experiment_ids=[EXP])
df = df[df["tags.mlflow.runName"].str.startswith("deberta_", na=False)]
df = df[df["tags.summary"].isna()] if "tags.summary" in df.columns else df
df = df[df["status"] == "FINISHED"]
# Each (row, seed) may be logged more than once; keep only the latest so n_runs and std are per seed.
df = (df.sort_values("start_time")
        .drop_duplicates(["tags.mlflow.runName", "params.seed"], keep="last"))

per_run = df[["tags.mlflow.runName", "run_id", "params.seed", "params.selected_epoch",
              "params.pairwise_weight", "params.train_sources", "params.threshold"]
             + [f"metrics.{m}" for m in METRICS]].copy()
per_run.columns = [c.split(".", 1)[1] if c.startswith(("metrics.", "params.")) else c
                   for c in per_run.columns]
per_run = per_run.rename(columns={"tags.mlflow.runName": "row"})
per_run["row"] = per_run["row"].str.replace("deberta_", "", regex=False)

g = per_run.groupby("row")


def joined(column, sep=","):
    # Params absent from a run come back as NaN; show them as "-" instead of crashing the sort.
    return g[column].apply(lambda s: sep.join(sorted(set(s.fillna("-").astype(str)))))


summary = pd.DataFrame({
    "n_runs": g.size(),
    "seeds": joined("seed"),
    "epochs": joined("selected_epoch"),
    "pairwise_weight": joined("pairwise_weight"),
    "train_sources": joined("train_sources", " | "),
})
for m in METRICS:
    summary[f"{m} mean"] = g[m].mean()
    summary[f"{m} std"] = g[m].std(ddof=1)
    summary[f"{m} min"] = g[m].min()
    summary[f"{m} max"] = g[m].max()
summary = summary.reset_index()

# Compact mean ± std view for reading
compact = summary[["row", "n_runs", "seeds", "epochs", "pairwise_weight", "train_sources"]].copy()
for m in METRICS:
    compact[m] = summary.apply(
        lambda r: f"{r[f'{m} mean']:.3f} ± {r[f'{m} std']:.3f}" if pd.notna(r[f"{m} std"])
        else f"{r[f'{m} mean']:.3f}", axis=1)

# Baselines for reference
base = mlflow.search_runs(experiment_ids=[EXP])
base = base[base["tags.mlflow.runName"].isin(["tfidf", "perplexity_b0"])]
baselines = base[["tags.mlflow.runName", "run_id"] + [f"metrics.{m}" for m in METRICS
                                                      if f"metrics.{m}" in base.columns]]
baselines.columns = [c.replace("metrics.", "").replace("tags.mlflow.runName", "detector")
                     for c in baselines.columns]

md = ("# DeBERTa runs: mean ± std by row\n\n" + compact.to_markdown(index=False)
      + "\n\n# Baselines\n\n" + baselines.round(4).to_markdown(index=False)
      + "\n\n# Per run\n\n" + per_run.round(4).to_markdown(index=False) + "\n")

run_id = None
if not args.no_log:
    with mlflow.start_run(experiment_id=EXP, run_name="deberta_seed_summary") as run:
        mlflow.set_tags({"summary": "true", "source_runs": ",".join(per_run["run_id"])})
        mlflow.log_param("n_source_runs", len(per_run))
        mlflow.log_param("rows", ",".join(summary["row"]))
        for _, r in summary.iterrows():
            for m in METRICS:
                key = m.replace("/", "_")
                mlflow.log_metric(f"{r['row']}.{key}.mean", r[f"{m} mean"])
                if pd.notna(r[f"{m} std"]):
                    mlflow.log_metric(f"{r['row']}.{key}.std", r[f"{m} std"])
        mlflow.log_table(summary, "summary_by_row.json")
        mlflow.log_table(per_run, "per_run.json")
        mlflow.log_table(baselines, "baselines.json")
        mlflow.log_text(md, "summary.md")
        run_id = run.info.run_id
        print("RUN_ID", run_id)

# Keep a copy in the repo: the MLflow database is shared and runs in it can be deleted.
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
source = (f"MLflow run {run_id} (deberta_seed_summary, experiment phase3-detectors)" if run_id
          else "not logged to MLflow (--no-log)")
header = (f"<!-- Generated by scripts/summarize_deberta_runs.py; {source}. "
          "Latest finished run per row and seed. -->\n\n")
(out_dir / "b2_seed_summary.md").write_text(header + md)
summary.to_csv(out_dir / "b2_seed_summary.csv", index=False)
print(f"wrote {out_dir / 'b2_seed_summary.md'} and {out_dir / 'b2_seed_summary.csv'}")

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 30)
print(compact.T.to_string())
print()
print(per_run.drop(columns=["train_sources"]).round(3).to_string(index=False))
