"""Runs a Friedman test across all models using 1000 bootstrap mean NDCG distributions,
followed by Nemenyi post-hoc pairwise comparisons to identify statistically similar groups,
and saves group assignments, p-value matrices, a summary CSV, and a colour-coded bar plot with group brackets."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

RESULTS_DIR = Path("../results/5_error_analysis_ctd_phenotype")
OUTPUT_DIR  = Path("../results/6_friedman_test_ctd_phenotype")
DATASET     = "ctd_phenotype"
P_THRESHOLD = 0.05          # models with p > this are in the same group
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RESULTS_DIR.glob(f"{DATASET}_*_bootstrap.csv"))
files = [f for f in files if "all_models" not in f.name and "summary" not in f.name]

if not files:
    raise FileNotFoundError(f"No bootstrap files found in {RESULTS_DIR}")

print(f"Found {len(files)} model bootstrap files\n")

model_data = {}
for fpath in files:
    model_name = fpath.stem.replace(f"{DATASET}_", "", 1).replace("_bootstrap", "")
    df   = pd.read_csv(fpath)
    col  = "mean_ndcg" if "mean_ndcg" in df.columns else df.columns[-1]
    vals = df[col].dropna().values
    model_data[model_name] = vals
    print(f"  {model_name:20s} -> {len(vals)} runs, mean={vals.mean():.4f}")

# Align all models to same number of runs (min across models)
min_runs = min(len(v) for v in model_data.values())
print(f"\nAligning to {min_runs} runs per model")

matrix = pd.DataFrame({
    model: vals[:min_runs]
    for model, vals in model_data.items()
})
# rows = bootstrap runs, columns = models
print(f"Matrix shape: {matrix.shape}  (runs × models)\n")

stat, p_friedman = friedmanchisquare(*[matrix[m].values for m in matrix.columns])

print("=" * 55)
print("FRIEDMAN TEST")
print("=" * 55)
print(f"  Chi-square statistic : {stat:.4f}")
print(f"  p-value              : {p_friedman:.6e}")

if p_friedman < P_THRESHOLD:
    print(f" p < {P_THRESHOLD} -> significant differences exist -> running Nemenyi post-hoc\n")
else:
    print(f" p >= {P_THRESHOLD} -> no significant differences detected")
    print("  All models treated as one group.\n")

print("Running Nemenyi post-hoc test...")
nemenyi_pvals = sp.posthoc_nemenyi_friedman(matrix.values)
nemenyi_pvals.index   = matrix.columns.tolist()
nemenyi_pvals.columns = matrix.columns.tolist()

nemenyi_pvals.to_csv(OUTPUT_DIR / f"{DATASET}_nemenyi_pvalues.csv")
print(f"Saved p-value matrix -> {DATASET}_nemenyi_pvalues.csv\n")

model_list = matrix.columns.tolist()
n          = len(model_list)

# mean NDCG for sorting
mean_ndcg = {m: matrix[m].mean() for m in model_list}

# build adjacency; edge exists if NOT significantly different
adj = {m: set() for m in model_list}
for i in range(n):
    for j in range(i + 1, n):
        m1, m2 = model_list[i], model_list[j]
        if nemenyi_pvals.loc[m1, m2] > P_THRESHOLD:
            adj[m1].add(m2)
            adj[m2].add(m1)

# Union-Find
parent = {m: m for m in model_list}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    parent[find(x)] = find(y)

for m in model_list:
    for neighbor in adj[m]:
        union(m, neighbor)

raw_groups = {}
for m in model_list:
    raw_groups.setdefault(find(m), []).append(m)

# sort groups by best mean NDCG descending
groups = sorted(
    raw_groups.values(),
    key=lambda grp: max(mean_ndcg[m] for m in grp),
    reverse=True
)

GROUP_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H","I","J","K","L","M"]
PALETTE = [
    "#1b7837",  
    "#2166ac",  
    "#d6604d",  
    "#8073ac",  
    "#b35806",  
    "#878787",  
    "#4dac26",  
    "#d01c8b",  
    "#01665e",  
    "#f4a582",  
    "#762a83",  
    "#c7eae5",  
    "#e08214",  
]

def gcolor(gid):
    return PALETTE[min(gid, len(PALETTE) - 1)]

model_to_gid = {}
group_rows   = []

print("=" * 55)
print("GROUP ASSIGNMENTS")
print("=" * 55)

for gid, grp in enumerate(groups):
    label   = GROUP_LABELS[gid] if gid < len(GROUP_LABELS) else str(gid + 1)
    members = sorted(grp, key=lambda m: mean_ndcg[m], reverse=True)
    print(f"\n  Group {label}: {members}")
    for m in members:
        model_to_gid[m] = gid
        group_rows.append({
            "model"      : m,
            "group_label": label,
            "group_id"   : gid + 1,
            "mean_ndcg"  : round(mean_ndcg[m], 4),
            "color"      : gcolor(gid),
        })

groups_df = pd.DataFrame(group_rows).sort_values("mean_ndcg", ascending=False)
groups_df.to_csv(OUTPUT_DIR / f"{DATASET}_friedman_groups.csv", index=False)
print(f"\nSaved groups -> {DATASET}_friedman_groups.csv")

# pairwise significance matrix (YES/no)
sig_matrix = pd.DataFrame(index=model_list, columns=model_list, dtype=str)
for m1 in model_list:
    for m2 in model_list:
        if m1 == m2:
            sig_matrix.loc[m1, m2] = "-"
        else:
            sig_matrix.loc[m1, m2] = "no" if nemenyi_pvals.loc[m1, m2] > P_THRESHOLD else "SIG"
sig_matrix.to_csv(OUTPUT_DIR / f"{DATASET}_nemenyi_sig_matrix.csv")

# Friedman summary
friedman_summary = pd.DataFrame([{
    "dataset"         : DATASET,
    "n_models"        : n,
    "n_bootstrap_runs": min_runs,
    "friedman_stat"   : round(stat, 4),
    "friedman_pvalue" : f"{p_friedman:.6e}",
    "significant"     : p_friedman < P_THRESHOLD,
    "p_threshold"     : P_THRESHOLD,
    "n_groups"        : len(groups),
}])
friedman_summary.to_csv(OUTPUT_DIR / f"{DATASET}_friedman_summary.csv", index=False)

# sort models best -> worst
ordered   = sorted(model_list, key=lambda m: mean_ndcg[m], reverse=True)
means_plt = [mean_ndcg[m] for m in ordered]
colors    = [gcolor(model_to_gid[m]) for m in ordered]
x_pos     = np.arange(len(ordered))

# bootstrap std for error bars
std_ndcg  = {m: matrix[m].std() for m in model_list}
errs      = [std_ndcg[m] for m in ordered]

fig, ax = plt.subplots(figsize=(max(16, len(ordered) * 1.3), 7))
fig.subplots_adjust(right=0.72)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.grid(axis="y", linestyle="--", color="#cccccc", alpha=0.7, zorder=0)

# bars
ax.bar(x_pos, means_plt, width=0.62, color=colors, edgecolor="none", zorder=3)

# error bars (std of bootstrap)
ax.errorbar(x_pos, means_plt,
            yerr=errs,
            fmt="none", ecolor="black",
            elinewidth=1.8, capsize=5, capthick=1.8, zorder=5)

# value labels
label_gap = max(errs) + 0.004
for xi, (m, val, e) in enumerate(zip(ordered, means_plt, errs)):
    ax.text(xi, val + e + 0.002,
            f"{val:.4f}\n±{e:.4f}",
            ha="center", va="bottom",
            fontsize=6.5, fontweight="bold",
            color="#222222", linespacing=1.3)

# group brackets
bracket_start = max(m + e for m, e in zip(means_plt, errs)) + label_gap + 0.025
bracket_step  = 0.022

for gid, grp in enumerate(groups):
    xs = [x_pos[ordered.index(m)] for m in grp if m in ordered]
    if not xs:
        continue
    x1, x2 = min(xs) - 0.28, max(xs) + 0.28
    y     = bracket_start + gid * bracket_step
    label = GROUP_LABELS[gid] if gid < len(GROUP_LABELS) else str(gid + 1)
    col   = gcolor(gid)

    ax.plot([x1, x2], [y, y], color=col, lw=3.0, solid_capstyle="round", zorder=7)
    for xend in [x1, x2]:
        ax.plot([xend, xend], [y - 0.003, y + 0.003], color=col, lw=1.8, zorder=7)
    ax.text((x1 + x2) / 2, y + 0.005,
            f"Group {label}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=col)

# colour strip at bar base
strip_h = max(means_plt) * 0.013
for xi, m in enumerate(ordered):
    ax.add_patch(plt.Rectangle(
        (xi - 0.31, 0), 0.62, strip_h,
        color=gcolor(model_to_gid[m]), zorder=6, alpha=0.9))

ax.set_xticks(x_pos)
ax.set_xticklabels(ordered, rotation=35, ha="right", fontsize=10.5)
ax.set_ylabel("Mean NDCG@10", fontsize=11)
ax.set_xlabel("Embedding Model", fontsize=11, labelpad=8)

y_top = bracket_start + len(groups) * bracket_step + 0.05
ax.set_ylim(0, y_top)
ax.set_xlim(-0.65, len(ordered) - 0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#aaaaaa")
ax.spines["bottom"].set_color("#aaaaaa")

ax.set_title(
    f"{DATASET.replace('_', ' ').title()} - NDCG@10 across all models\n"
    f"Friedman χ²={stat:.2f}, p={p_friedman:.2e}  |  "
    f"Nemenyi post-hoc (p>{P_THRESHOLD} = same group)  ·  same colour = same group",
    fontsize=11, fontweight="bold", pad=14
)

# legend outside
legend_patches = [
    mpatches.Patch(
        color=gcolor(gid), alpha=0.9,
        label=(f"Group {GROUP_LABELS[gid] if gid < len(GROUP_LABELS) else gid+1}: "
               f"{', '.join(sorted(grp, key=lambda m: mean_ndcg[m], reverse=True))}")
    )
    for gid, grp in enumerate(groups)
]
ax.legend(
    handles=legend_patches,
    title=f"Nemenyi groups (p>{P_THRESHOLD})\nsame group = not significantly different",
    title_fontsize=9, fontsize=8.5,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    framealpha=0.95, edgecolor="#cccccc"
)

ax.text(0.01, 0.01,
        f"* Error bars = bootstrap std  "
        f"|  groups = Nemenyi post-hoc connected components  "
        f"|  Friedman p={p_friedman:.2e}",
        transform=ax.transAxes, fontsize=7.5, color="#888888")

plot_path = OUTPUT_DIR / f"{DATASET}_friedman_groups_barplot.png"
plt.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved bar plot -> {plot_path}")
print("\nDone!")

import networkx as nx
from itertools import combinations

# CLIQUE-BASED GROUPING
print("\n" + "=" * 55)
print("CLIQUE-BASED GROUP ASSIGNMENTS")
print("=" * 55)

# Build adjacency graph: edge = NOT significantly different
G = nx.Graph()
G.add_nodes_from(model_list)
for m1, m2 in combinations(model_list, 2):
    if nemenyi_pvals.loc[m1, m2] > P_THRESHOLD:
        G.add_edge(m1, m2)

# Find all maximal cliques
cliques = list(nx.find_cliques(G))

# Sort cliques by best mean NDCG descending
cliques = sorted(cliques, key=lambda grp: max(mean_ndcg[m] for m in grp), reverse=True)

# Assign group labels
clique_rows = []
model_to_clique_labels = {m: [] for m in model_list}  # a model can appear in multiple cliques

print()
for gid, clique in enumerate(cliques):
    label   = GROUP_LABELS[gid] if gid < len(GROUP_LABELS) else str(gid + 1)
    members = sorted(clique, key=lambda m: mean_ndcg[m], reverse=True)
    print(f"  Clique {label}: {members}")
    for m in members:
        model_to_clique_labels[m].append(label)
        clique_rows.append({
            "model"       : m,
            "clique_label": label,
            "clique_id"   : gid + 1,
            "mean_ndcg"   : round(mean_ndcg[m], 4),
            "color"       : gcolor(gid),
        })

# Show models that belong to multiple cliques
print("\n  Models appearing in multiple cliques (ambiguous):")
any_multi = False
for m, labels in model_to_clique_labels.items():
    if len(labels) > 1:
        print(f"    {m}: {labels}")
        any_multi = True
if not any_multi:
    print("    None - cliques are non-overlapping (same result as Union-Find)")

# Save clique assignments CSV
cliques_df = pd.DataFrame(clique_rows).sort_values(["clique_id", "mean_ndcg"], ascending=[True, False])
cliques_df.to_csv(OUTPUT_DIR / f"{DATASET}_clique_groups.csv", index=False)
print(f"\nSaved clique groups -> {DATASET}_clique_groups.csv")

# Save a compact summary: one row per model, all clique labels it belongs to
clique_summary_rows = []
for m in sorted(model_list, key=lambda m: mean_ndcg[m], reverse=True):
    labels = model_to_clique_labels[m]
    clique_summary_rows.append({
        "model"         : m,
        "mean_ndcg"     : round(mean_ndcg[m], 4),
        "clique_labels" : ",".join(labels),
        "n_cliques"     : len(labels),
        "ambiguous"     : len(labels) > 1,
    })
clique_summary_df = pd.DataFrame(clique_summary_rows)
clique_summary_df.to_csv(OUTPUT_DIR / f"{DATASET}_clique_summary.csv", index=False)
print(f"Saved clique summary  -> {DATASET}_clique_summary.csv")

# CLIQUE BAR PLOT
ordered_c   = sorted(model_list, key=lambda m: mean_ndcg[m], reverse=True)
means_c     = [mean_ndcg[m] for m in ordered_c]
errs_c      = [std_ndcg[m]  for m in ordered_c]

# Bar color = color of the FIRST (best) clique the model belongs to
bar_colors_c = [gcolor(cliques.index(
    next(cl for cl in cliques if m in cl)
)) for m in ordered_c]

x_pos_c = np.arange(len(ordered_c))

fig2, ax2 = plt.subplots(figsize=(max(16, len(ordered_c) * 1.3), 7))
fig2.subplots_adjust(right=0.72)
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")
ax2.grid(axis="y", linestyle="--", color="#cccccc", alpha=0.7, zorder=0)

ax2.bar(x_pos_c, means_c, width=0.62, color=bar_colors_c, edgecolor="none", zorder=3)
ax2.errorbar(x_pos_c, means_c,
             yerr=errs_c,
             fmt="none", ecolor="black",
             elinewidth=1.8, capsize=5, capthick=1.8, zorder=5)

# Value labels
for xi, (val, e) in enumerate(zip(means_c, errs_c)):
    ax2.text(xi, val + e + 0.002,
             f"{val:.4f}\n±{e:.4f}",
             ha="center", va="bottom",
             fontsize=6.5, fontweight="bold",
             color="#222222", linespacing=1.3)

# Clique brackets - one bracket per clique
bracket_start_c = max(m + e for m, e in zip(means_c, errs_c)) + max(errs_c) + 0.030
bracket_step_c  = 0.022

for gid, clique in enumerate(cliques):
    xs = [x_pos_c[ordered_c.index(m)] for m in clique if m in ordered_c]
    if not xs:
        continue
    x1, x2 = min(xs) - 0.28, max(xs) + 0.28
    y       = bracket_start_c + gid * bracket_step_c
    label   = GROUP_LABELS[gid] if gid < len(GROUP_LABELS) else str(gid + 1)
    col     = gcolor(gid)

    ax2.plot([x1, x2], [y, y], color=col, lw=3.0, solid_capstyle="round", zorder=7)
    for xend in [x1, x2]:
        ax2.plot([xend, xend], [y - 0.003, y + 0.003], color=col, lw=1.8, zorder=7)
    ax2.text((x1 + x2) / 2, y + 0.005,
             f"Clique {label}",
             ha="center", va="bottom",
             fontsize=8.5, fontweight="bold", color=col)

# Color strip at bar base
strip_h_c = max(means_c) * 0.013
for xi, m in enumerate(ordered_c):
    ax2.add_patch(plt.Rectangle(
        (xi - 0.31, 0), 0.62, strip_h_c,
        color=bar_colors_c[xi], zorder=6, alpha=0.9))

ax2.set_xticks(x_pos_c)
ax2.set_xticklabels(ordered_c, rotation=35, ha="right", fontsize=10.5)
ax2.set_ylabel("Mean NDCG@10", fontsize=11)
ax2.set_xlabel("Embedding Model", fontsize=11, labelpad=8)

y_top_c = bracket_start_c + len(cliques) * bracket_step_c + 0.05
ax2.set_ylim(0, y_top_c)
ax2.set_xlim(-0.65, len(ordered_c) - 0.35)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color("#aaaaaa")
ax2.spines["bottom"].set_color("#aaaaaa")

ax2.set_title(
    f"{DATASET.replace('_', ' ').title()} - NDCG@10 across all models\n"
    f"Friedman χ²={stat:.2f}, p={p_friedman:.2e}  |  "
    f"Nemenyi post-hoc (p>{P_THRESHOLD} = same clique)  ·  same colour = same clique\n"
    f"Note: a model may appear in multiple cliques (shown in summary CSV)",
    fontsize=11, fontweight="bold", pad=14
)

# Legend - one entry per clique
legend_patches_c = [
    mpatches.Patch(
        color=gcolor(gid), alpha=0.9,
        label=(f"Clique {GROUP_LABELS[gid] if gid < len(GROUP_LABELS) else gid+1}: "
               f"{', '.join(sorted(cl, key=lambda m: mean_ndcg[m], reverse=True))}")
    )
    for gid, cl in enumerate(cliques)
]
ax2.legend(
    handles=legend_patches_c,
    title=f"Nemenyi cliques (p>{P_THRESHOLD})\nevery pair within clique is non-significant",
    title_fontsize=9, fontsize=8.5,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    framealpha=0.95, edgecolor="#cccccc"
)

ax2.text(0.01, 0.01,
         f"* Error bars = bootstrap std  "
         f"|  brackets = maximal cliques (stricter than connected components)  "
         f"|  Friedman p={p_friedman:.2e}",
         transform=ax2.transAxes, fontsize=7.5, color="#888888")

clique_plot_path = OUTPUT_DIR / f"{DATASET}_clique_groups_barplot.png"
plt.savefig(clique_plot_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved clique bar plot -> {clique_plot_path}")
print("\nDone (clique analysis complete)!")