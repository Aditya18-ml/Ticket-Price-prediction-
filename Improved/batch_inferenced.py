# Improved/batch_inference.py
import os
import joblib
import pandas as pd
from Improved.config import PIPELINE_CONFIG, MODEL_DIR, OUTPUT_DATA_PATH

def run_unified_inference(input_filepath, output_filepath):
    print(f"Loading incoming batch data from {input_filepath}...")
    try:
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {input_filepath}.")
        return
    
    print("\n--- Starting Unified Inference ---")
    

    for model_name, config in PIPELINE_CONFIG.items():
        model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model {model_name} not found at {model_path}. Skipping...")
            continue
            
        print(f"Generating predictions for: {model_name}")
        

        model = joblib.load(model_path)
        

        missing_cols = [col for col in config['features'] if col not in df.columns]
        if missing_cols:
            print(f"WARNING: Missing features for {model_name}: {missing_cols}. Skipping...")
            continue
            
        X_new = df[config['features']]
        

        prediction_col_name = f"predicted_{config['target']}"
        df[prediction_col_name] = model.predict(X_new)
        

    df.to_csv(output_filepath, index=False)
    print(f"\nSuccess! All predictions saved to {output_filepath}")

if __name__ == "__main__":
    INPUT_DATA = '../Support_tickets.csv'
    
    run_unified_inference(INPUT_DATA, OUTPUT_DATA_PATH)