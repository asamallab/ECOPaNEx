import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "../results/analyse_folder/1_data_processing/ecotox_processed.csv"
total_rows = len(pd.read_csv(csv_path))
print(f"Total rows: {total_rows}")

results_base = "../results/analyse_folder"

models = [
    "2_bge", "2_biobert", "2_biolord", "2_e5", "2_gte",
    "2_instructor_xl", "2_jina", "2_medembed", "2_minilm",
    "2_nomic", "2_pubmedbert", "2_scibert", "2_tf_idf"
]

accepted_counts = {}
for model in models:
    csv_path = os.path.join(results_base, model, "ecotox_k10_accepted.csv")
    if os.path.exists(csv_path):
        accepted_counts[model] = len(pd.read_csv(csv_path))
    else:
        print(f"WARNING: File not found for {model}")
        accepted_counts[model] = 0

# Sort by accepted count descending
sorted_models = sorted(models, key=lambda m: accepted_counts[m], reverse=True)

labels = [m.replace("2_", "") for m in sorted_models]
accepted = [accepted_counts[m] for m in sorted_models]
rejected = [total_rows - a for a in accepted]

x = range(len(labels))
bar_width = 0.5

fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(x, accepted, bar_width, label="Accepted", color="#4C9BE8")
ax.bar(x, rejected, bar_width, bottom=accepted, label="Rejected", color="#E85C5C")

# Annotate accepted count and percentage
for i, (a, r) in enumerate(zip(accepted, rejected)):
    ax.text(i, a / 2, str(a), ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    pct = f"{100 * a / total_rows:.1f}%"
    ax.text(i, total_rows + total_rows * 0.01, pct, ha="center", va="bottom", fontsize=8, color="#333")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
ax.set_ylabel("Number of Rows")
ax.set_title(f"Accepted vs Rejected per Model  (Total = {total_rows})")
ax.axhline(total_rows, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend()
ax.set_ylim(0, total_rows * 1.12)

plt.tight_layout()
output_dir = "../results/analyse_folder/6_accepted_plot"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "model_acceptance_ecotox_barplot.png"), dpi=150)
plt.show()
print(f"Saved: {output_dir}/model_acceptance_barplot.png")