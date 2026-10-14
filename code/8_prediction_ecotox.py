import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_FILE  = "../input_data/ECOTOX_KE_info.tsv"
AOP_FILE    = "../results/1a_data_preprocessing/aop_ke_cleaned.csv"
OUTPUT_FILE = "../results/8_prediction/ecotox_predictions.csv"

Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

TOP_K = 10

df = pd.read_csv(INPUT_FILE, sep="\t")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df_pred = df[
    df["Info"].astype(str).str.strip().str.lower() == "to be predicted"
].copy()
print("Rows needing prediction:", len(df_pred))

ke_df    = pd.read_csv(AOP_FILE)
ke_ids   = ke_df["KE_iden"].astype(str).tolist()
ke_texts = ["passage: " + t for t in ke_df["ke_text"].fillna("").tolist()]

print("Loading model...")
model = SentenceTransformer("intfloat/e5-large-v2")

print("Encoding KE texts...")
ke_vectors = model.encode(
    ke_texts,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True,
)

# Combine three columns into one query string
query_cols = ["measurement_description", "measurement_definition", "trend_description"]

def combine_query(row):
    parts = [str(row[col]).strip() for col in query_cols if pd.notna(row[col]) and str(row[col]).strip() not in ("", "nan")]
    return " | ".join(parts)

df_pred["combined_query"] = df_pred.apply(combine_query, axis=1)
queries = ["query: " + q for q in df_pred["combined_query"].tolist()]

print("Encoding queries...")
query_vectors = model.encode(
    queries,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True,
)

similarity = np.dot(query_vectors, ke_vectors.T)

top10_kes, top10_scores = [], []
for i in range(len(queries)):
    scores  = similarity[i]
    indices = np.argsort(scores)[::-1][:TOP_K]
    top10_kes.append([ke_ids[idx] for idx in indices])
    top10_scores.append([float(scores[idx]) for idx in indices])

df_pred["Top10_KE_ID"]  = top10_kes
df_pred["Top10_Scores"] = top10_scores

df_pred.to_csv(OUTPUT_FILE, index=False)
print("\nTop-10 KE prediction saved:")
print(OUTPUT_FILE)

# EXPLODE Top10 into one row per KE
rows = []
for _, row in df_pred.iterrows():
    for ke_id, score in zip(row["Top10_KE_ID"], row["Top10_Scores"]):
        new_row = row.drop(["Top10_KE_ID", "Top10_Scores"]).to_dict()
        new_row["KE_ID"]    = ke_id
        new_row["KE_Score"] = score
        rows.append(new_row)

exploded_df = pd.DataFrame(rows)

# Merge KE details from ke_df
ke_df["KE_iden"] = ke_df["KE_iden"].astype(str)
exploded_df["KE_ID"] = exploded_df["KE_ID"].astype(str)

exploded_df = exploded_df.merge(ke_df, left_on="KE_ID", right_on="KE_iden", how="left")

# Save for manual checking
EXPLODED_OUTPUT = "../results/8_prediction/ecotox_predictions_exploded.csv"
exploded_df.to_csv(EXPLODED_OUTPUT, index=False)
print("\nExploded predictions saved to:", EXPLODED_OUTPUT)

# Filter by threshold and save
THRESHOLD = 0.5
filtered_df = exploded_df[exploded_df["KE_Score"] > THRESHOLD].copy()

FILTERED_OUTPUT = "../results/8_prediction/ecotox_predictions_filtered.csv"
filtered_df.to_csv(FILTERED_OUTPUT, index=False)
print("\nFiltered predictions (score > 0.5) saved to:", FILTERED_OUTPUT)