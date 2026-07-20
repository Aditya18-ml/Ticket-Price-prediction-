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
    
    # Loop through configurations to apply each model sequentially
    for model_name, config in PIPELINE_CONFIG.items():
        model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model {model_name} not found at {model_path}. Skipping...")
            continue
            
        print(f"Generating predictions for: {model_name}")
        
        # Load the specific trained model
        model = joblib.load(model_path)
        
        # Ensure the required features exist in the incoming data
        missing_cols = [col for col in config['features'] if col not in df.columns]
        if missing_cols:
            print(f"WARNING: Missing features for {model_name}: {missing_cols}. Skipping...")
            continue
            
        X_new = df[config['features']]
        
        # Generate predictions and append them as a new column
        prediction_col_name = f"predicted_{config['target']}"
        df[prediction_col_name] = model.predict(X_new)
        
    # Save the consolidated results
    df.to_csv(output_filepath, index=False)
    print(f"\nSuccess! All predictions saved to {output_filepath}")

if __name__ == "__main__":
    # For demonstration, reusing the original dataset to generate bulk predictions.
    # In production, replace this with the path to the incoming, unlabelled data.
    INPUT_DATA = '../Support_tickets.csv'
    
    run_unified_inference(INPUT_DATA, OUTPUT_DATA_PATH)