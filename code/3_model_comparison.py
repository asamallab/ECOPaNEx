"""
Step 3 - Model Comparison
-------------------------

This code plots the different metrics as bar plots to visualize the differences across each model.
For each K value, this code generates a plot for the input datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BASE_DIR = Path('../results/')
output_dir = BASE_DIR / "3_model_comparison"
output_dir.mkdir(parents=True, exist_ok=True)

MODEL_DIRS = {
    "bge": BASE_DIR / "2_bge",
    "biobert": BASE_DIR / "2_biobert",
    "pubmedbert": BASE_DIR / "2_pubmedbert",
    "biolord": BASE_DIR / "2_biolord",
    "tfidf": BASE_DIR / "2_tf_idf",
    "scibert": BASE_DIR / "2_scibert",
    "gte": BASE_DIR / "2_gte",
    "medembed": BASE_DIR / "2_medembed",
    "instructor_xl": BASE_DIR / "2_instructor_xl",
    "nomic": BASE_DIR / "2_nomic",
    "e5": BASE_DIR / "2_e5",
    "jina": BASE_DIR / "2_jina"
}

BIOMEDICAL_MODELS = {
    "biobert",
    "pubmedbert",
    "biolord",
    "scibert",
    "medembed"
}

MODEL_DISPLAY_NAMES = {
    "biobert": "BioBERT",
    "pubmedbert": "PubMedBERT",
    "biolord": "BioLORD",
    "scibert": "SciBERT",
    "medembed": "MedEmbed",
    "bge": "BGE",
    "instructor_xl": "Instructor-XL",
    "e5": "E5",
    "nomic": "Nomic",
    "gte": "GTE",
    "jina": "Jina",
    "tfidf": "TF-IDF"
}

DATASETS = ["ecotox", "ctd_disease", "ctd_phenotype"]
metrics = ["MRR", "NDCG@K", "Hit@K", "Precision@K", "Recall@K"]

all_data = []

for model_name, model_path in MODEL_DIRS.items():
    for dataset in DATASETS:
        file_path = model_path / f"{dataset}_metrics.csv"
        df = pd.read_csv(file_path)
        df["model"] = model_name
        df["dataset"] = dataset
        all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv(output_dir / "all_models_combined_metrics.csv", index=False)

def plot_metric(dataset, k, metric):
    df = final_df[(final_df["dataset"] == dataset) & (final_df["k"] == k)].copy()
    df = df.sort_values(by=metric, ascending=False)

    models = df["model"].map(MODEL_DISPLAY_NAMES).tolist()
    values = df[metric].tolist()
    colors = ["blue" if m in BIOMEDICAL_MODELS else "orange" for m in df["model"]]

    plt.figure(figsize=(12, 6))
    plt.bar(models, values, color=colors)

    plt.title(f"{dataset} | K={k} | {metric}")
    plt.xlabel("Models")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    for i, v in enumerate(values):
        plt.text(i, v, f"{round(v,3)}", ha='center', va='bottom', fontsize=8)

    legend_elements = [
        Patch(facecolor='blue', label='Biomedical'),
        Patch(facecolor='orange', label='Non-Biomedical')
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset}_k{k}_{metric}.png")
    plt.close()

def plot_line_metric(dataset, metric):
    df = final_df[final_df["dataset"] == dataset].copy()
    
    # Sort by K to ensure the line plots connect chronologically
    df = df.sort_values(by="k")
    
    plt.figure(figsize=(12, 7))
    
    # Use a diverse color palette to separate the 12 models easily
    colors = plt.cm.tab20.colors 
    
    for idx, model_name in enumerate(MODEL_DIRS.keys()):
        model_df = df[df["model"] == model_name]
        if model_df.empty:
            continue
            
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        # Solid lines for Biomedical, dashed lines for Non-Biomedical
        if model_name in BIOMEDICAL_MODELS:
            linestyle = "-"
            marker = "o"
        else:
            linestyle = "--"
            marker = "s"
            
        plt.plot(
            model_df["k"].astype(str), # Convert to string to treat K values as categorical steps
            model_df[metric], 
            label=display_name, 
            color=colors[idx % len(colors)],
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            markersize=6
        )
        
    plt.title(f"{dataset} | {metric} Trend Across K Values")
    plt.xlabel("K Values")
    plt.ylabel(metric)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Place the legend on the side so it does not overlap with data points
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset}_{metric}_trend_across_k.png")
    plt.close()

# Execution loop for both types of plots
for dataset in DATASETS:
    # Generate original bar charts
    ks = final_df[final_df["dataset"] == dataset]["k"].unique()
    for k in ks:
        for metric in metrics:
            plot_metric(dataset, k, metric)
            
    # Generate new line charts across all K values
    for metric in metrics:
        plot_line_metric(dataset, metric)