"""Computes per-query NDCG@10 for each embedding model against ground truth KEs,
runs 1000 bootstrap subsampling iterations (90%) to estimate mean NDCG with 95% CI,
and saves per-model rowwise scores, bootstrap distributions, a summary CSV, and bar/box plots."""

import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score

BASE_DIR   = "../results/"
BASE_GROUND_TRUTH = BASE_DIR + "1a_data_preprocessing/"

DATASET_NAME = "ecotox"  

MODELS = [
    "bge", "biobert", "biolord", "e5", "gte",
    "instructor_xl", "jina", "medembed", "minilm",
    "nomic", "pubmedbert", "scibert", "tf_idf",
]

def get_result_path(model, dataset):
    return f"{BASE_DIR}2_{model}/{dataset}_k10_accepted.csv"
GROUND_TRUTH_FILE       = f"{BASE_GROUND_TRUTH}{DATASET_NAME}_processed.csv"
OUTPUT_DIR              = f"{BASE_DIR}5_error_analysis_{DATASET_NAME}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)     
COMBINED_SUMMARY_OUTPUT = OUTPUT_DIR + f"{DATASET_NAME}_all_models_bootstrap_summary.csv"
BARPLOT_OUTPUT          = OUTPUT_DIR + f"{DATASET_NAME}_all_models_ndcg_barplot.png"
BOXPLOT_OUTPUT          = OUTPUT_DIR + f"{DATASET_NAME}_all_models_ndcg_boxplot.png"

BOOTSTRAP_RUNS  = 1000
SAMPLE_FRACTION = 0.90
TOP_K           = 10
RANDOM_SEED     = 42

def compute_ndcg(true_ke, pred_list, top_k=10):
    y_true  = [0] * top_k
    y_score = list(range(top_k, 0, -1))
    if true_ke in pred_list:
        rank = pred_list.index(true_ke)
        y_true[rank] = 1
    return ndcg_score([y_true], [y_score], k=top_k)


print(f"Loading ground truth: {GROUND_TRUTH_FILE}")
truth_df = pd.read_csv(GROUND_TRUTH_FILE)[["query", "_true_ke"]]

all_summaries        = []
all_bootstrap_means  = {}   # list of 1000 bootstrap means

for model in MODELS:

    result_path = get_result_path(model, DATASET_NAME)
    print(f"\n{'='*55}")
    print(f"MODEL: {model.upper()}")
    print(f"  File: {result_path}")

    try:
        results_df = pd.read_csv(result_path)
    except FileNotFoundError:
        print(f"  [SKIP] File not found.")
        continue

    df = results_df.merge(truth_df, on="query", how="inner")
    print(f"  Merged rows: {len(df)}")
    if len(df) == 0:
        print(f"  [SKIP] No matching rows.")
        continue

    # row-wise NDCG
    ndcg_scores = []
    for _, row in df.iterrows():
        preds   = [str(x) for x in ast.literal_eval(row["preds"])]
        true_ke = str(ast.literal_eval(row["_true_ke"])[0])
        ndcg_scores.append(compute_ndcg(true_ke, preds, TOP_K))
    df["ndcg"] = ndcg_scores

    # bootstrap
    np.random.seed(RANDOM_SEED)
    sample_size     = int(len(df) * SAMPLE_FRACTION)
    bootstrap_means = []

    for run in range(BOOTSTRAP_RUNS):
        sampled_df = df.sample(n=sample_size, replace=False,
                               random_state=RANDOM_SEED + run)
        bootstrap_means.append(sampled_df["ndcg"].mean())
        if (run + 1) % 200 == 0:
            print(f"  Run {run+1:04d}/{BOOTSTRAP_RUNS} | "
                  f"Mean so far = {np.mean(bootstrap_means):.4f}")

    all_bootstrap_means[model] = bootstrap_means

    overall_mean = np.mean(bootstrap_means)
    overall_std  = np.std(bootstrap_means)
    ci_low  = overall_mean - 1.96 * (overall_std / np.sqrt(BOOTSTRAP_RUNS))
    ci_high = overall_mean + 1.96 * (overall_std / np.sqrt(BOOTSTRAP_RUNS))

    print(f"  Mean NDCG@{TOP_K} : {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"  95% CI           : [{ci_low:.4f}, {ci_high:.4f}]")

    # save per-model CSVs
    pd.DataFrame({"run": range(1, BOOTSTRAP_RUNS + 1),
                  "mean_ndcg": bootstrap_means}
                 ).to_csv(OUTPUT_DIR + f"{DATASET_NAME}_{model}_bootstrap.csv",
                          index=False)
    df.to_csv(OUTPUT_DIR + f"{DATASET_NAME}_{model}_rowwise_ndcg.csv", index=False)

    all_summaries.append({
        "model"               : model,
        "dataset"             : DATASET_NAME,
        "n_rows"              : len(df),
        "bootstrap_runs"      : BOOTSTRAP_RUNS,
        "sample_fraction"     : SAMPLE_FRACTION,
        "sample_size"         : sample_size,
        "mean_ndcg"           : round(overall_mean, 4),
        "std_ndcg"            : round(overall_std,  4),
        "mean_plus_minus_std" : f"{overall_mean:.4f} ± {overall_std:.4f}",
        "ci_lower"            : round(ci_low,  4),
        "ci_upper"            : round(ci_high, 4),
    })

summary_df = (pd.DataFrame(all_summaries)
                .sort_values("mean_ndcg", ascending=False)
                .reset_index(drop=True))
summary_df.insert(0, "rank", range(1, len(summary_df) + 1))
summary_df.to_csv(COMBINED_SUMMARY_OUTPUT, index=False)

print("\n\n" + "="*55)
print("FINAL RANKING")
print("="*55)
print(summary_df[["rank", "model", "mean_ndcg", "std_ndcg",
                   "ci_lower", "ci_upper"]].to_string(index=False))

models_sorted = summary_df["model"].tolist()
means_sorted  = summary_df["mean_ndcg"].tolist()
stds_sorted   = summary_df["std_ndcg"].tolist()

n      = len(models_sorted)
cmap   = plt.cm.Blues_r
colors = [cmap(0.2 + 0.55 * i / max(n - 1, 1)) for i in range(n)]

print("\nGenerating bar plot...")

fig, ax = plt.subplots(figsize=(14, 6))

bars = ax.bar(
    models_sorted, means_sorted,
    yerr=stds_sorted,
    color=colors, edgecolor="black", linewidth=0.8,
    capsize=6,
    error_kw=dict(elinewidth=2.0, ecolor="black", capthick=2.0),
    width=0.6, zorder=3,
)

for bar, m, s in zip(bars, means_sorted, stds_sorted):
    ax.text(bar.get_x() + bar.get_width() / 2,
            m + s + 0.001,
            f"{m:.4f}\n±{s:.4f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

all_means = np.array(means_sorted)
all_stds  = np.array(stds_sorted)
y_margin  = max(all_stds) * 10
y_min     = max(0.0, all_means.min() - y_margin * 2)
y_max     = all_means.max() + all_stds.max() * 12 + 0.01

ax.set_ylim(y_min, y_max)
ax.set_ylabel(f"Mean NDCG@{TOP_K}", fontsize=12)
ax.set_xlabel("Embedding Model", fontsize=12)
ax.set_title(
    f"{DATASET_NAME.replace('_', ' ').title()} — NDCG@{TOP_K} across all models\n"
    f"({BOOTSTRAP_RUNS} bootstrap runs, {int(SAMPLE_FRACTION*100)}% subsampling) "
    f"— sorted best → worst",
    fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=30, labelsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.annotate(f"* Y-axis starts at {y_min:.3f}",
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(BARPLOT_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved bar plot : {BARPLOT_OUTPUT}")

print("Generating box plot...")

# collect bootstrap data in sorted order
box_data = [all_bootstrap_means[m] for m in models_sorted]

fig, ax = plt.subplots(figsize=(14, 6))

bp = ax.boxplot(
    box_data,
    patch_artist=True,          # filled boxes
    notch=False,                # standard rectangular box
    vert=True,
    widths=0.55,
    medianprops=dict(color="black", linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker="o", markersize=3,
                    markerfacecolor="gray", alpha=0.5),
    boxprops=dict(linewidth=1.2),
)

# apply colour gradient (best -> worst)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

# overlay individual mean dots
for i, (m, data) in enumerate(zip(models_sorted, box_data), start=1):
    ax.scatter(i, np.mean(data), color="red",
               zorder=5, s=40, label="Mean" if i == 1 else "")

# annotate median above each box
for i, data in enumerate(box_data, start=1):
    med = np.median(data)
    ax.text(i, med + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.005,
            f"{med:.4f}", ha="center", va="bottom",
            fontsize=7, fontweight="bold", color="black")

ax.set_xticks(range(1, n + 1))
ax.set_xticklabels(models_sorted, rotation=30, ha="right", fontsize=10)
ax.set_ylabel(f"Bootstrap Mean NDCG@{TOP_K}", fontsize=12)
ax.set_xlabel("Embedding Model", fontsize=12)
ax.set_title(
    f"{DATASET_NAME.replace('_', ' ').title()} — NDCG@{TOP_K} Bootstrap Distribution\n"
    f"({BOOTSTRAP_RUNS} bootstrap runs, {int(SAMPLE_FRACTION*100)}% subsampling) "
    f"— sorted best → worst",
    fontsize=12, fontweight="bold")

ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(BOXPLOT_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved box plot : {BOXPLOT_OUTPUT}")

print("\nDone!")