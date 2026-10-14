"""Evaluates retrieval metrics (MRR, NDCG, Hit, Precision, Recall) for Ecotox KE predictions.
Compares real model performance against label-randomized baseline to validate model signal.

example usage:
python3 4_ecotox_random_true_labels.py ../results/ ..
/results/1a_data_preprocessing/ecotox_processed.csv ../results/4_ecotox_label_randomization_e5 ecotox_k10_accepted.c
sv 2_e5
"""

import sys

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 6:
    print("Usage: python script.py <base_dir> <ecotox_ke_mapping_tsv> <output_dir> <ecotox_k_10_accepted_file> <2_best_model_name>")
    sys.exit(1)

BASE_DIR      = sys.argv[1]
GT_TSV        = sys.argv[2]
OUTPUT_DIR    = sys.argv[3]
ACCEPTED_FILE = sys.argv[4]
MODEL_RUN     = sys.argv[5]
K             = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_list(val):
    try:
        return ast.literal_eval(str(val))
    except:
        return []

def parse_ke_ids(val):
    s = str(val).strip()
    if s in ("-", "nan", ""):
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(k).strip() for k in parsed]
        return [str(parsed).strip()]
    except:
        return [s]

df = pd.read_csv(os.path.join(BASE_DIR, MODEL_RUN, ACCEPTED_FILE))
df["preds"] = df["preds"].apply(parse_list)
df["preds"] = df["preds"].apply(lambda x: [str(i).strip() for i in x])

gt_df = pd.read_csv(GT_TSV, dtype=str)
gt_df = gt_df[gt_df["Info"].str.strip().str.lower() == "mapped"].copy()
print(f"GT rows with Already_mapped=Yes: {len(gt_df)}")

gt_dict = {}
for _, row in gt_df.iterrows():
    meas  = str(row["measurement_description"]).strip()
    trend = str(row["trend_description"]).strip()
    ke_ids = parse_ke_ids(row["Mapped_KE_ID"])
    if not ke_ids:
        continue
    key = (meas, trend)
    if key not in gt_dict:
        gt_dict[key] = set()
    gt_dict[key].update(ke_ids)


def lookup_gt(query_text):
    q = str(query_text).strip()
    last_word = q.split()[-1] if q.split() else ""
    for (meas, trend), ke_set in gt_dict.items():
        if q.startswith(meas) and last_word == trend:
            return ke_set
    return set()

df["ground_truth"] = df["query"].apply(lookup_gt)

matched = df["ground_truth"].apply(lambda x: len(x) > 0).sum()
print(f"Rows with ground truth matched: {matched} / {len(df)}")

sample = df[df["ground_truth"].apply(lambda x: len(x) > 0)].head(2)
for _, row in sample.iterrows():
    print("Preds:", row["preds"][:5])
    print("GT KEs:", row["ground_truth"])
    print()

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
    os.path.join(OUTPUT_DIR, "ecotox_real_vs_randomized_metrics.csv"),
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
plt.title("ECOTOX Ecotox - Real vs Label Randomization")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ecotox.png"), dpi=150)
plt.close()