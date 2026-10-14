"""Loads Group A models from Friedman-based grouping results, then takes the union of their
top-10 KE predictions across all queries to build per-query KE lists and model-agreement counts,
and saves union CSVs, JSONs, global KE frequency stats, and binned distribution plots."""


import ast
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATASET = "ctd_phenotype"   

BASE_DIR = Path("../results/")
STAT_DIR      = BASE_DIR / "6_friedman_test_ctd_phenotype"
OUTPUT_DIR    = STAT_DIR / f"{DATASET}_groupA_union"

STAT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CI groups CSV produced by the previous script
CI_GROUPS_CSV = STAT_DIR / f"{DATASET}_friedman_groups.csv"

TOP_K = 10

print(f"\n{'='*60}")
print(f"Dataset : {DATASET}")
print(f"{'='*60}\n")

if not CI_GROUPS_CSV.exists():
    raise FileNotFoundError(
        f"CI groups file not found: {CI_GROUPS_CSV}\n"
        f"Run the bootstrap CI grouping script first."
    )

ci_groups_df = pd.read_csv(CI_GROUPS_CSV)

# Group A = group_label == "A"
group_a_df      = ci_groups_df[ci_groups_df["group_label"] == "A"]
GROUP_A_MODELS  = group_a_df["model"].tolist()
all_groups      = ci_groups_df.groupby("group_label")["model"].apply(list).to_dict()

print("Groups loaded from CI groups CSV:")
for label, members in sorted(all_groups.items()):
    marker = "  <- using this" if label == "A" else ""
    print(f"  Group {label}: {', '.join(sorted(members))}{marker}")

print(f"\nGroup A models ({len(GROUP_A_MODELS)}): {GROUP_A_MODELS}")

if not GROUP_A_MODELS:
    raise ValueError("No Group A models found in CI groups CSV.")

print(f"\n{'-'*60}")
print("Loading predictions...")
print(f"{'-'*60}\n")

query_union   = {}   # query -> set of KE strings
query_sources = {}   # query -> {ke: [model1, model2, ...]}
loaded_models = []

for model in GROUP_A_MODELS:
    result_path = BASE_DIR / f"2_{model}" / f"{DATASET}_k10_accepted.csv"
    if not result_path.exists():
        print(f"  [SKIP] {model:<20} not found: {result_path}")
        continue

    df = pd.read_csv(result_path)
    loaded_models.append(model)
    print(f"  {model:<20} rows={len(df)}")

    for _, row in df.iterrows():
        query = row["query"]
        try:
            preds = ast.literal_eval(str(row["preds"]))
        except Exception:
            preds = [str(row["preds"])]
        preds = [str(p).strip() for p in preds]

        if query not in query_union:
            query_union[query]   = set()
            query_sources[query] = {}

        for ke in preds:
            query_union[query].add(ke)
            query_sources[query].setdefault(ke, []).append(model)

N_MODELS = len(loaded_models)
print(f"\nLoaded {N_MODELS} models | {len(query_union)} queries")

# LIST: query -> sorted list of all union KEs
list_rows = []
for query, ke_set in sorted(query_union.items()):
    ke_list = sorted(ke_set)
    list_rows.append({
        "query"         : query,
        "union_ke_list" : ke_list,
        "union_ke_count": len(ke_list),
    })
list_df = pd.DataFrame(list_rows)

# DICT: query -> {ke: count_of_models_that_predicted_it}
dict_rows  = []
json_dict  = {}   # used for downstream steps

for query in sorted(query_sources.keys()):
    ke_counts = {
        ke: len(models)
        for ke, models in query_sources[query].items()
    }
    ke_counts_sorted = dict(
        sorted(ke_counts.items(), key=lambda x: x[1], reverse=True)
    )
    json_dict[query] = ke_counts_sorted

    most_common_ke    = max(ke_counts_sorted, key=ke_counts_sorted.get)
    most_common_count = ke_counts_sorted[most_common_ke]

    dict_rows.append({
        "query"          : query,
        "ke_count_dict"  : ke_counts_sorted,
        "total_unique_ke": len(ke_counts_sorted),
        "most_common_ke" : most_common_ke,
        "max_count"      : most_common_count,
        "high_agreement" : sum(
            1 for c in ke_counts_sorted.values()
            if c >= max(2, N_MODELS // 2)
        ),
    })
dict_df = pd.DataFrame(dict_rows)

# list CSV
list_df_csv = list_df.copy()
list_df_csv["union_ke_list"] = list_df_csv["union_ke_list"].apply(str)
list_path = OUTPUT_DIR / f"{DATASET}_groupA_union_list.csv"
list_df_csv.to_csv(list_path, index=False)

# dict CSV
dict_df_csv = dict_df.copy()
dict_df_csv["ke_count_dict"] = dict_df_csv["ke_count_dict"].apply(str)
dict_path = OUTPUT_DIR / f"{DATASET}_groupA_union_dict.csv"
dict_df_csv.to_csv(dict_path, index=False)

# JSON list
json_list_out = {row["query"]: row["union_ke_list"] for _, row in list_df.iterrows()}
list_json_path = OUTPUT_DIR / f"{DATASET}_groupA_union_list.json"
with open(list_json_path, "w") as f:
    json.dump(json_list_out, f, indent=2)

# JSON dict
dict_json_path = OUTPUT_DIR / f"{DATASET}_groupA_union_dict.json"
with open(dict_json_path, "w") as f:
    json.dump(json_dict, f, indent=2)

print(f"\nSaved: {list_path.name}")
print(f"Saved: {dict_path.name}")
print(f"Saved: {list_json_path.name}")
print(f"Saved: {dict_json_path.name}")

global_ke_counter = Counter()
for q in json_dict:
    for ke, count in json_dict[q].items():
        global_ke_counter[ke] += count

global_df = pd.DataFrame(
    global_ke_counter.most_common(),
    columns=["ke", "total_model_votes"]
)
global_df["queries_appeared_in"] = global_df["ke"].apply(
    lambda ke: sum(1 for q in json_dict if ke in json_dict[q])
)
global_df["avg_models_per_query"] = (
    global_df["total_model_votes"] / global_df["queries_appeared_in"]
).round(2)

global_path = OUTPUT_DIR / f"{DATASET}_groupA_ke_global_frequency.csv"
global_df.to_csv(global_path, index=False)
print(f"Saved: {global_path.name}")


freq_values = global_df["queries_appeared_in"].values
max_freq    = int(freq_values.max())
interval    = 10

bin_starts = list(range(1, max_freq + interval + 1, interval))  
bin_edges  = [b - 0.5 for b in bin_starts] + [bin_starts[-1] + interval - 0.5]
bin_labels = [f"{bin_starts[i]}-{bin_starts[i] + interval - 1}"
              for i in range(len(bin_starts))]

counts, _ = np.histogram(freq_values, bins=bin_edges)

# remove trailing empty bins
last_nonzero = np.max(np.nonzero(counts)) + 1
counts       = counts[:last_nonzero]
bin_labels   = bin_labels[:last_nonzero]

n_bins  = len(counts)
palette = []
for i in range(n_bins):
    frac = i / max(n_bins - 1, 1)
    if frac < 0.33:
        palette.append("#2196F3")
    elif frac < 0.66:
        palette.append("#FF9800")
    else:
        palette.append("#E53935")

fig, ax = plt.subplots(figsize=(max(12, n_bins * 0.9), 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x_pos = np.arange(n_bins)
bars  = ax.bar(x_pos, counts, color=palette,
               edgecolor="none", width=0.7, zorder=3)

for xi, cnt in enumerate(counts):
    if cnt > 0:
        ax.text(xi, cnt + max(counts) * 0.01,
                str(cnt),
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#222222")

ax.set_xticks(x_pos)
ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
ax.set_xlabel("Number of queries KE appears in (frequency interval)", fontsize=11)
ax.set_ylabel("Number of unique KEs", fontsize=11)
ax.set_title(
    f"{DATASET.replace('_', ' ').title()} — KE frequency distribution\n"
    f"Group A union predictions  ({N_MODELS} models, {len(query_union)} queries, "
    f"{len(global_df)} unique KEs)",
    fontsize=12, fontweight="bold"
)
ax.grid(axis="y", linestyle="--", color="#cccccc", alpha=0.7, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_patches = [
    mpatches.Patch(color="#2196F3", label="Rare KEs (low query coverage)"),
    mpatches.Patch(color="#FF9800", label="Medium frequency KEs"),
    mpatches.Patch(color="#E53935", label="Core KEs (high query coverage)"),
]
ax.legend(handles=legend_patches, fontsize=9,
          loc="upper right", framealpha=0.9)

long_tail = int(counts[0]) if counts[0] > 0 else 0
pct_tail  = long_tail / len(global_df) * 100
ax.text(0.01, 0.97,
        f"Long tail: {long_tail} KEs ({pct_tail:.1f}%) appear in only 1-10 queries",
        transform=ax.transAxes, fontsize=8.5,
        color="#555555", va="top")

plt.tight_layout()
plot_path = OUTPUT_DIR / f"{DATASET}_groupA_ke_frequency_intervals.png"
plt.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {plot_path.name}")

all_counts   = [c for q in json_dict for c in json_dict[q].values()]
count_series = pd.Series(all_counts)

print(f"\n{'='*60}")
print("AGREEMENT DISTRIBUTION")
print(f"(how many Group A models agree on each KE per query)")
print(f"{'='*60}")
print(f"\n  Total query-KE pairs : {len(all_counts)}")
print(f"\n  {'Count':>6} | {'Meaning':<32} | Frequency")
print(f"  {'-'*60}")
for c in range(1, N_MODELS + 1):
    freq    = int((count_series == c).sum())
    pct     = freq / len(all_counts) * 100
    meaning = f"predicted by {c}/{N_MODELS} models"
    bar     = "█" * int(pct / 2)
    print(f"  {c:>6} | {meaning:<32} | {freq:>6} ({pct:5.1f}%)  {bar}")

print(f"\n{'='*60}")
print("OUTPUT FILES")
print(f"{'='*60}")
for f in sorted(OUTPUT_DIR.glob("*")):
    print(f"  {f.name:<55} {f.stat().st_size/1024:>8.1f} KB")

print(f"\n{'='*60}")
print("SPREAD SUMMARY")
print(f"{'='*60}")
print(f"  Unique KEs total          : {len(global_df)}")
print(f"  KEs in only 1 query       : "
      f"{(global_df['queries_appeared_in'] == 1).sum()}")
print(f"  KEs in 10+ queries        : "
      f"{(global_df['queries_appeared_in'] >= 10).sum()}")
print(f"  KEs in 50+ queries        : "
      f"{(global_df['queries_appeared_in'] >= 50).sum()}")
print(f"  Avg queries per KE        : "
      f"{global_df['queries_appeared_in'].mean():.2f}")
print(f"  Max queries for one KE    : "
      f"{global_df['queries_appeared_in'].max()}")
print(f"\nDone.\n")

ke_counts    = list_df["union_ke_count"].values
max_ke_count = int(ke_counts.max())
min_ke_count = int(ke_counts.min())

# Bin size 
bin_size   = TOP_K
bin_starts = list(range(1, max_ke_count + bin_size + 1, bin_size))  
bin_edges  = [b - 0.5 for b in bin_starts] + [bin_starts[-1] + bin_size - 0.5]
bin_labels = [f"{bin_starts[i]}-{bin_starts[i] + bin_size - 1}"
              for i in range(len(bin_starts))]

counts_ke, _ = np.histogram(ke_counts, bins=bin_edges)

# trim trailing empty bins
last_nz    = np.max(np.nonzero(counts_ke)) + 1
counts_ke  = counts_ke[:last_nz]
bin_labels = bin_labels[:last_nz]
n_bins_ke  = len(counts_ke)

palette_ke = []
for i in range(n_bins_ke):
    frac = i / max(n_bins_ke - 1, 1)
    if frac < 0.33:
        palette_ke.append("#2196F3")
    elif frac < 0.66:
        palette_ke.append("#FF9800")
    else:
        palette_ke.append("#E53935")

fig2, ax2 = plt.subplots(figsize=(max(8, n_bins_ke * 1.5), 6))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")

x2 = np.arange(n_bins_ke)
ax2.bar(x2, counts_ke, color=palette_ke, edgecolor="none", width=0.5, zorder=3)

for xi, cnt in enumerate(counts_ke):
    if cnt > 0:
        ax2.text(xi, cnt + max(counts_ke) * 0.01, str(cnt),
                 ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color="#222222")

ax2.set_xticks(x2)
ax2.set_xticklabels(bin_labels, rotation=0, ha="center", fontsize=11)
ax2.set_xlabel("Number of unique KEs in union (per query)", fontsize=11)
ax2.set_ylabel("Number of queries", fontsize=11)
ax2.set_title(
    f"{DATASET.replace('_', ' ').title()} — Union KE count distribution\n"
    f"Group A union  ({N_MODELS} models, {len(query_union)} queries)",
    fontsize=12, fontweight="bold"
)
ax2.grid(axis="y", linestyle="--", color="#cccccc", alpha=0.7, zorder=0)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2.text(0.01, 0.97,
         f"Median: {int(np.median(ke_counts))} KEs/query  |  "
         f"Mean: {np.mean(ke_counts):.1f}  |  "
         f"Range: {min_ke_count}-{max_ke_count}",
         transform=ax2.transAxes, fontsize=8.5, color="#555555", va="top")

plt.tight_layout()
ke_plot_path = OUTPUT_DIR / f"{DATASET}_groupA_union_ke_count_distribution.png"
plt.savefig(ke_plot_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {ke_plot_path.name}")