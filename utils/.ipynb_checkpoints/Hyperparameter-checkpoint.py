import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,GridSearchCV

def hyperparametertunning(model, scoring_method, param_grid, X_train, y_train, X_test):
    kf=KFold(n_splits=5,shuffle=True,random_state=42)
    grid_search = GridSearchCV(estimator=model,param_grid=param_grid,
                        cv=kf,scoring=scoring_method,n_jobs=-1,verbose=1                   
    )
    
    grid_search.fit(X_train, y_train)
    print(f"best parameters:{grid_search.best_params_}")
    print(f"best training scores:{grid_search.best_score_:.3f}")
    
    best=grid_search.best_estimator_
    y_pred = best.predict(X_test)
    
    importance = getattr(best, "feature_importances_", None)
    return y_pred,importance

def param_grid_lgbm():
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 4, 5],
        'num_leaves': [7, 15, 31],
        'min_child_samples': [50, 100]
    }

    return param_grid

def param_grid_xgb():
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0]
    }
    return param_grid

def param_grid_rf():
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 8, 12],
        'min_samples_leaf': [4, 10, 20] 
    }
    return param_grid
   