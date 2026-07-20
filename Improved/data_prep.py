# Improved/data_prep.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_data(filepath):
    print(f"Loading and cleaning data from {filepath}...")
    df = pd.read_csv(filepath)
 
    df = df.dropna(subset=['customer_sentiment'])
    
  
    cols_to_drop = ['ticket_id', 'company_id']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    return df

def align_target_classes(y):
    if y.min() == 1:
        y = y - 1
    return y