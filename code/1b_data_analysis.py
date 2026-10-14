"""
Step 1b: Analyze the processed data
-----------------------------------
This code evaluates data distribution across various endpoints and quantifies the associated true Key Events (KEs).
"""

import logging
import pandas as pd
import ast
import sys
from pathlib import Path

if len(sys.argv) != 4:
    print(
        "Usage: python3 1b_data_analysis.py "
        "<base_dir> <input_dir> <output_dir>"
    )
    sys.exit(1)

base_dir        = sys.argv[1]
input_dir_name  = sys.argv[2]
output_dir_name = sys.argv[3]



BASE_DIR   = Path(base_dir)
INPUT_DIR  = BASE_DIR / input_dir_name
OUTPUT_DIR = BASE_DIR / output_dir_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
log_file = OUTPUT_DIR / '1b_data_analysis.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

INPUT_FILES = {
    "ctd_disease"  : INPUT_DIR / "ctd_disease_processed.csv",
    "ctd_phenotype": INPUT_DIR / "ctd_phenotype_processed.csv",
    "ecotox"       : INPUT_DIR / "ecotox_processed.csv",
}

def count_ke(value):
    if pd.isna(value) or str(value).strip() == "":
        return 0
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return len(parsed)
        else:
            return 1  
    except:
        return 1 if str(value).strip() else 0


for name, file_path in INPUT_FILES.items():

    df = pd.read_csv(file_path, sep=",")
    # Right after loading the df, normalise the column name
    df = df.rename(columns={"Mapped_KE_ID": "Mapped_KE"})

    logging.info(" Columns detected:")
    logging.info(df.columns.tolist())

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    logging.info(f"\n'Info' unique values: {df['Info'].unique().tolist()}")
    logging.info(f"'Info' null count: {df['Info'].isna().sum()}")

    df_yes = df[df["Info"].str.strip() == "Mapped"].copy()
    logging.info(f"\n Info rows: {len(df_yes)}")

    logging.info(f"\n Sample Mapped_KE values (first 5):")
    logging.info(df_yes["Mapped_KE"].head(5).tolist())
    logging.info(f"Mapped_KE dtype     : {df_yes['Mapped_KE'].dtype}")
    logging.info(f"Mapped_KE null count: {df_yes['Mapped_KE'].isna().sum()}")

    df_yes["num_mapped_ke"] = df_yes["Mapped_KE"].apply(count_ke)

    logging.info(f"\n num_mapped_ke value counts (raw):")
    logging.info(df_yes["num_mapped_ke"].value_counts().sort_index())

    distribution = (
        df_yes["num_mapped_ke"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    distribution.columns = ["No_of_Mapped_KE", "No_of_Items"]

    logging.info(f"\n Distribution:")
    logging.info(distribution)

    output_file = OUTPUT_DIR / f"{name}_mapped_ke_distribution.csv"
    distribution.to_csv(output_file, index=False)

    logging.info(f"\n Saved → {output_file}")