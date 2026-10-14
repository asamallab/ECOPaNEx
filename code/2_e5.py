""" 
Step 2: Data Retrieval
----------------------

Model - E5
 The script performs semantic retrieval of Key Events (KEs) using a E5 SentenceTransformer model.
 It computes similarity between queries and KE texts, applies threshold filtering, 
 and evaluates performance using ranking metrics (MRR, NDCG, Hit@K, Precision, Recall, Coverage).
 """

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import ast
import sys
from pathlib import Path
import sys
from pathlib import Path

# ARGUMENTS 
if len(sys.argv) != 4:
    print(
        "Usage: python3 2_e5.py "
        "<base_dir> <input_dir> <output_dir>"
    )
    sys.exit(1)

base_dir        = sys.argv[1]
input_dir_name  = sys.argv[2]
output_dir_name = sys.argv[3]

# PATHS 
BASE_DIR = Path(base_dir)
INPUT_DIR = BASE_DIR / input_dir_name
OUTPUT_DIR = BASE_DIR / output_dir_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
log_file = OUTPUT_DIR / '2_e5.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# FILES 
AOP_FILE = INPUT_DIR / "aop_ke_cleaned.csv"

FILES = {
    "ecotox": INPUT_DIR / "ecotox_processed.csv",
    "ctd_disease": INPUT_DIR / "ctd_disease_processed.csv",
    "ctd_phenotype": INPUT_DIR / "ctd_phenotype_processed.csv"
}

# K VALUES
K_VALUES = {
    "ecotox": [1,3, 5, 7, 10],
    "ctd_disease": [1,3, 5, 7, 10],
    "ctd_phenotype": [1,3, 5, 7, 10]
}

#THRESHOLD
THRESHOLD = 0.5

ke_df = pd.read_csv(AOP_FILE)
ke_ids = ke_df['KE_iden'].astype(str).tolist()
ke_texts = ["passage: " + t for t in ke_df['ke_text'].fillna('').tolist()]

# MODEL
model = SentenceTransformer("intfloat/e5-large-v2")
ke_vectors = model.encode(
    ke_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

def parse_true_ke(val):
    try:
        parsed = ast.literal_eval(str(val))
        return {str(v) for v in parsed}
    except:
        return {str(val)}

def retrieve(df):
    queries = ["query: " + q for q in df['query'].fillna('').tolist()]
    query_vectors = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    sim = np.dot(query_vectors, ke_vectors.T)

    preds_list, scores_list = [], []

    for i in range(len(queries)):
        scores = sim[i]
        indices = np.argsort(scores)[::-1]
        preds_list.append([ke_ids[idx] for idx in indices])
        scores_list.append([float(scores[idx]) for idx in indices])

    df['preds'] = preds_list
    df['scores'] = scores_list
    return df

# EVALUATION
def compute_metrics_and_save(df, dataset_name):
    ks = K_VALUES[dataset_name]
    results = []

    for k in ks:
        mrr_list, ndcg_list, hit_list = [], [], []
        prec_list, recall_list = [], []

        accepted_rows = []
        rejected_rows = []

        for i, row in df.iterrows():
            true_kes = parse_true_ke(row['_true_ke'])

            preds = row['preds'][:k]
            scores = row['scores'][:k]

            filtered = [(p, s) for p, s in zip(preds, scores) if s > THRESHOLD]

            if len(filtered) == 0:
                rejected_rows.append({"row_id": i, "query": row['query']})
                mrr_list.append(0)
                ndcg_list.append(0)
                hit_list.append(0)
                prec_list.append(0)
                recall_list.append(0)
                continue

            filtered_preds = [p for p, _ in filtered]

            accepted_rows.append({
                "row_id": i,
                "query": row['query'],
                "preds": filtered_preds,
                "scores": [s for _, s in filtered]
            })

            hit = int(any(p in true_kes for p in filtered_preds))
            hit_list.append(hit)

            correct = sum(p in true_kes for p in filtered_preds)
            prec = correct / len(filtered_preds)
            prec_list.append(prec)

            recall = correct / len(true_kes)
            recall_list.append(recall)

            rr = 0
            for rank, p in enumerate(filtered_preds):
                if p in true_kes:
                    rr = 1 / (rank + 1)
                    break
            mrr_list.append(rr)

            dcg = 0
            for idx, p in enumerate(filtered_preds):
                if p in true_kes:
                    dcg += 1 / np.log2(idx + 2)

            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_kes), len(filtered_preds)))])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_list.append(ndcg)

        coverage = np.mean([
            any(s > THRESHOLD for s in row['scores'][:k])
            for _, row in df.iterrows()
        ])

        pd.DataFrame(accepted_rows).to_csv(
            OUTPUT_DIR / f"{dataset_name}_k{k}_accepted.csv", index=False
        )

        pd.DataFrame(rejected_rows).to_csv(
            OUTPUT_DIR / f"{dataset_name}_k{k}_rejected.csv", index=False
        )

        results.append({
            "dataset": dataset_name,
            "k": k,
            "MRR": np.mean(mrr_list),
            "NDCG@K": np.mean(ndcg_list),
            "Hit@K": np.mean(hit_list),
            "Precision@K": np.mean(prec_list),
            "Recall@K": np.mean(recall_list),
            "Coverage": coverage
        })

    return pd.DataFrame(results)

all_metrics = []

for name, file_path in FILES.items():
    print(f"\nProcessing {name}...")
    df = pd.read_csv(file_path)
    df = retrieve(df)

    metrics_df = compute_metrics_and_save(df, name)
    metrics_df.to_csv(OUTPUT_DIR / f"{name}_metrics.csv", index=False)

    all_metrics.append(metrics_df)

final_metrics = pd.concat(all_metrics, ignore_index=True)
final_metrics.to_csv(OUTPUT_DIR / "all_metrics.csv", index=False)