# Improved/data_prep.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_data(filepath):
    """Loads raw data and applies global preprocessing steps."""
    print(f"Loading and cleaning data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Handle Missing Data: Drop rows where 'customer_sentiment' is null 
    df = df.dropna(subset=['customer_sentiment'])
    
    # 2. Drop unnecessary identifiers that shouldn't be used for training
    cols_to_drop = ['ticket_id', 'company_id']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    return df

def align_target_classes(y):
    """Ensures classification targets start at 0, required by LightGBM/XGBoost."""
    if y.min() == 1:
        y = y - 1
    return y