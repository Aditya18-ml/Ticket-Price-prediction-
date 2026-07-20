import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def featureimportance(important_col,importances,save_path):
    feature_imp_df = pd.DataFrame({
        'Feature': important_col,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=feature_imp_df, x='Importance', y='Feature', palette='viridis')
    plt.title('What Drives Ticket Priority? ')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def confusionmatrix(y_test,y_pred,save_path):
    print("Classification report")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['P1', 'P2', 'P3'], yticklabels=['P1', 'P2', 'P3'])
    plt.title('Confusion Matrix (Actual vs Predicted Categories)')
    plt.ylabel('Actual Priority')
    plt.xlabel('Predicted Priority')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    
def correlation(df,column):
    plt.figure(figsize=(12, 8))
    corr_matrix = df[column].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def residuals(y_test,y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Residuals Distribution ')
    plt.xlabel('Error ')
    plt.show()