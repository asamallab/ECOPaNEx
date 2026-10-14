"""Evaluates retrieval metrics (MRR, NDCG, Hit, Precision, Recall) for CTD Phenotype KE predictions.
Compares real model performance against label-randomized baseline to validate model signal.

example usage:
python3 4_ctd_phenotype_random_true_labels.py ../results/ ../results/1a_data_preprocessing/ctd_phenotype_processed.csv ../results/4_ctd_phenotype_label_randomization_nomic ctd_phenotype_k10_accepted.csv 2_nomic
"""

import sys
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 6:
    print("Usage: python script.py <base_dir> <ctd_phenotype_ke_mapping_tsv> <output_dir> <ctd_phenotype_k_10_accepted_file> <2_best_model_name>")
    sys.exit(1)

BASE_DIR      = sys.argv[1]
GT_TSV        = sys.argv[2]
OUTPUT_DIR    = sys.argv[3]
ACCEPTED_FILE = sys.argv[4]
MODEL_RUN     = sys.argv[5]

K=10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_list(val):
    try:
        return ast.literal_eval(str(val))
    except:
        return []

df = pd.read_csv(os.path.join(BASE_DIR, MODEL_RUN, ACCEPTED_FILE))
df["preds"] = df["preds"].apply(parse_list)
df["preds"] = df["preds"].apply(lambda x: [str(i).strip() for i in x])

gt_df = pd.read_csv(GT_TSV)

gt_dict = {}
for _, row in gt_df.iterrows():
    key = str(row["phenotypename"]).strip()
    try:
        ke_list = ast.literal_eval(str(row["Mapped_KE"]))
    except:
        ke_list = [str(row["Mapped_KE"])]
    ke_list = [str(k).strip() for k in ke_list]
    if key not in gt_dict:
        gt_dict[key] = set()
    gt_dict[key].update(ke_list)

df["ground_truth"] = df["query"].apply(lambda q: gt_dict.get(str(q).strip(), set()))

def compute_metrics(df, k=10):
    mrr, ndcg, hit, prec, rec = [], [], [], [], []
    for _, row in df.iterrows():
        preds = row["preds"][:k]
        gt = row["ground_truth"]
        if not preds or not gt:
            mrr.append(0); ndcg.append(0); hit.append(0); prec.append(0); rec.append(0)
            continue
        rr = 0
        for i, p in enumerate(preds):
            if p in gt:
                rr = 1 / (i + 1)
                break
        mrr.append(rr)
        hit.append(int(any(p in gt for p in preds)))
        correct = sum(p in gt for p in preds)
        prec.append(correct / k)
        rec.append(correct / len(gt))
        dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(preds) if p in gt)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt), k)))
        ndcg.append(dcg / idcg if idcg > 0 else 0)
    return {
        "MRR": np.mean(mrr),
        "NDCG@K": np.mean(ndcg),
        "Hit@K": np.mean(hit),
        "Precision@K": np.mean(prec),
        "Recall@K": np.mean(rec),
    }

def label_randomization(df):
    shuffled = df.copy()
    shuffled["ground_truth"] = np.random.permutation(shuffled["ground_truth"].values)
    return shuffled

real_metrics = compute_metrics(df, K)
rand_df = label_randomization(df)
rand_metrics = compute_metrics(rand_df, K)

keys = ["MRR", "NDCG@K", "Hit@K", "Precision@K", "Recall@K"]
real_vals = [real_metrics[k] for k in keys]
rand_vals = [rand_metrics[k] for k in keys]

# --- save table ---
results_df = pd.DataFrame({
    "Metric": keys,
    "Real": real_vals,
    "Label_Randomized": rand_vals,
})
results_df["Difference"] = results_df["Real"] - results_df["Label_Randomized"]
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "ctd_phenotype_real_vs_randomized_metrics.csv"),
    index=False,
)
print(results_df.round(4).to_string(index=False))

# ---- plotting ----
x = np.arange(len(keys))
w = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - w/2, real_vals, width=w, label="Real")
plt.bar(x + w/2, rand_vals, width=w, label="Label Randomized")
plt.xticks(x, keys)
plt.ylabel("Score")
plt.title("ECOTOX CTD Phenotype - Real vs Label Randomization")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ctd_phenotype.png"), dpi=150)
plt.close()