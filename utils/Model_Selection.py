import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import SVC,SVR                   
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier,XGBRegressor           
from lightgbm import LGBMClassifier,LGBMRegressor       

def model_classification(X_train,X_test,y_train,y_test):
    models = {
        'SVC': SVC(),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'xgboost': XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(random_state=42, objective='multiclass')
    }
    
    result = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
            
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro') 
        result.append({
            'Model': name,
            'accuracy': accuracy,
            'f1': f1
        })
    
    result_df = pd.DataFrame(result).sort_values(by='f1', ascending=False)
    return result_df
    
def model_regression(X_train,X_test,y_train,y_test):
    models={
        'SVR':SVR(),
        'decision_tree':DecisionTreeRegressor(random_state=42),
        'random_forest':RandomForestRegressor(random_state=42),
        'XGB':XGBRegressor(random_state=42),
        'LGBM':LGBMRegressor(random_state=42)
    }
    
    
    results=[]    
    for name,model in models.items():
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) 
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae      
        })
    
    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    return results_df