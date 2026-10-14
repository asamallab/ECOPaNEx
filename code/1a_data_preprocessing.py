""" 
Step 1a: Data Preprocessing
---------------------------
Loads and cleans source files.
Here, the code loads AOP-Wiki Key Events, ECOTOX, CTD Disease and CTD Phenotype source files.
Then, it filters invalid KE IDs, builds query text per source, and drops unmapped or empty rows.
Finally, it saves cleaned files (in csv format) to output_dir: 
aop_ke_cleaned.csv, ecotox_processed.csv, ctd_disease_processed.csv, ctd_phenotype_processed.csv.
"""

import logging
import pandas as pd
import numpy as np
import re, ast
import sys
from pathlib import Path

# ARGUMENTS 
if len(sys.argv) != 3:
    print("Usage: python3 1a_data_preprocessing.py <data_dir> <output_dir>")
    sys.exit(1)

DATA_DIR   = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set up logging
log_file = OUTPUT_DIR / '1a_data_preprocessing.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

AOP_FILE    = DATA_DIR / 'AOPWiki_KE_details.tsv'
ECOTOX_FILE = DATA_DIR / 'ECOTOX_KE_info.tsv'
CTDD_FILE   = DATA_DIR / 'CTD_Disease_KE_info.tsv'
CTDP_FILE   = DATA_DIR / 'CTD_Phenotype_KE_info.tsv'

KE_ID_MAX = 10000 # Manually assigned KE IDs in previous studies. Filtered out here.

# DATA LOADING 
def load_aop(path):
    df = pd.read_csv(path, sep='\t')
    df = df[df['KE_iden'] <= KE_ID_MAX] # Filter out manually assigned KEs from previous studies.
    df = df[~df['KE_iden'].isin([294, 1114])] # KEs 294 and 1114 have word 'unknown' in their title.
    df = df[df['title'].notna() | df['process_name'].notna() | df['object_name'].notna()].copy()

    df = df.fillna('')

    df = df.groupby('KE_iden').agg({
        'title': lambda x: ' '.join(sorted(set([i.strip() for i in x if i.strip()]))),
        'process_name': lambda x: ' '.join(sorted(set([i.strip() for i in x if i.strip()]))),
        'object_name': lambda x: ' '.join(sorted(set([i.strip() for i in x if i.strip()])))
    }).reset_index()

    df['ke_text'] = (
        df['title'] + ' ' +
        df['process_name'] + ' ' +
        df['object_name']
    ).str.strip().str.replace(r'\s+', ' ', regex=True)

    return df.reset_index(drop=True)

ke_df = load_aop(AOP_FILE)
ke_df.to_csv(OUTPUT_DIR / 'aop_ke_cleaned.csv', index=False)

def filter_ke_ids(val):
    try:
        v = ast.literal_eval(str(val))
        ids = v if isinstance(v, list) else [v]
    except:
        ids = [val]

    return [
        str(x).strip().strip("'")
        for x in ids
        if str(x).strip().strip("'").isdigit()
        and int(str(x).strip().strip("'")) <= KE_ID_MAX
    ]

# BUILDING QUERIES
def build_ecotox_query(row):
    desc  = str(row.get('measurement_description', '')).strip()
    defn  = str(row.get('measurement_definition', '')).strip()
    trend = str(row.get('trend_description', '')).strip()

    query = f"{desc} {defn} {trend}"
    return re.sub(r'\s+', ' ', query).strip()


def build_ctd_disease_query(row):
    return str(row.get('DiseaseName', '')).strip()


def build_ctd_phenotype_query(row):
    return str(row.get('phenotypename', '')).strip()


def load_source(path, source_name):
    logging.info(f"Processing {source_name}")
    logging.info(f"FILE BEING READ: /{path.parent.name}/{path.name}")
    df_full = pd.read_csv(path, sep='\t')

    logging.info(f"Processing {source_name}")
    logging.info(f"Rows before filtering: {len(df_full)}")

    mapped_col = 'Info' if 'Info' in df_full.columns else 'Already mapped'

    df = df_full[
        df_full[mapped_col].astype(str).str.lower() == 'mapped'
    ].copy()

    logging.info(f"Rows after Info filter: {len(df)}")

    df = df.fillna('')

    if source_name == 'ECOTOX':
        df['query'] = df.apply(build_ecotox_query, axis=1)
        ke_col = 'Mapped_KE_ID'

    elif source_name == 'CTD_Disease':
        df['query'] = df.apply(build_ctd_disease_query, axis=1)
        ke_col = 'Mapped_KE'

    else:
        df['query'] = df.apply(build_ctd_phenotype_query, axis=1)
        ke_col = 'Mapped_KE'

    df['_valid_ke_ids'] = df[ke_col].apply(filter_ke_ids)

    before = len(df)
    df = df[df['_valid_ke_ids'].str.len() > 0].copy()
    after = len(df)

    logging.info(f"{source_name}: dropped {before - after} rows due to invalid KE")
    logging.info(f"{source_name}: final rows = {after}")

    df['_true_ke'] = df['_valid_ke_ids']

    return df.reset_index(drop=True)


ecotox_df = load_source(ECOTOX_FILE, 'ECOTOX')
ctdd_df   = load_source(CTDD_FILE,   'CTD_Disease')
ctdp_df   = load_source(CTDP_FILE,   'CTD_Phenotype')

def print_nan_counts(df, name):
    logging.info(f"\n{name} - NaN counts per column:")
    logging.info(df.isna().sum().sort_values(ascending=False))


print_nan_counts(ke_df, "AOP KE")
print_nan_counts(ecotox_df, "ECOTOX")
print_nan_counts(ctdd_df, "CTD Disease")
print_nan_counts(ctdp_df, "CTD Phenotype")

# SAVE PROCESSED FILES
ecotox_df.to_csv(OUTPUT_DIR / 'ecotox_processed.csv', index=False)
ctdd_df.to_csv(OUTPUT_DIR / 'ctd_disease_processed.csv', index=False)
ctdp_df.to_csv(OUTPUT_DIR / 'ctd_phenotype_processed.csv', index=False)

