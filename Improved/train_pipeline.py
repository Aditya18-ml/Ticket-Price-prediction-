# Improved/train_pipeline.py

import os
import joblib
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

# UPDATE THESE THREE LINES:
from Improved.config import DATA_PATH, MODEL_DIR, PIPELINE_CONFIG
from Improved.data_prep import load_and_clean_data, align_target_classes
from utils.Evaluation_plot import featureimportance, confusionmatrix

def train_all_models():
    print("--- Initializing Multi-Model Training Pipeline ---")
    
    # 1. Load and Clean Data globally
    df = load_and_clean_data(DATA_PATH)
    
    # Ensure output directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("plots", exist_ok=True) # Directory for your visual reports
    
    for model_name, config in PIPELINE_CONFIG.items():
        print(f"\nTraining Model: {model_name.upper()}")
        
        target = config['target']
        features = config['features']
        task_type = config['task_type']
        params = config['params']
        
        # Isolate task-specific data
        clean_df = df.dropna(subset=[target])
        X = clean_df[features]
        y = clean_df[target]
        
        # Standardize target starting index for classifiers
        if task_type == 'classification':
            y = align_target_classes(y)
            model = LGBMClassifier(**params)
        else:
            model = LGBMRegressor(**params)
        
        print(f"Fitting {model_name} model with {len(X)} samples...")
        model.fit(X, y)
        
        # Save the model
        model_save_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
        joblib.dump(model, model_save_path)
        
        # --- GENERATE AND SAVE EVALUATION PLOTS ---
        # Note: In a real pipeline, you would split train/test here to generate these plots.
        # For simplicity, we are evaluating on the training data just to generate the requested plots.
        y_pred = model.predict(X)
        
        feature_plot_path = os.path.join("plots", f"{model_name}_feature_importance.png")
        featureimportance(features, model.feature_importances_, feature_plot_path)
        print(f"Saved Feature Importance plot to {feature_plot_path}")
        
        if task_type == 'classification':
            conf_matrix_path = os.path.join("plots", f"{model_name}_confusion_matrix.png")
            # Convert probabilities to hard predictions if necessary, else pass directly
            confusionmatrix(y, y_pred, conf_matrix_path)
            print(f"Saved Confusion Matrix plot to {conf_matrix_path}")

    print("\nTraining pipeline completed successfully.")

if __name__ == "__main__":
    train_all_models()