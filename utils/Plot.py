import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

def plot_scatter(df, x_col, y_col, hue_col=None, title=None):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='deep')
    plt.title(title if title else f'{y_col} vs {x_col}')
    plt.tight_layout()
    plt.show()

def plot_boxplot(df, x_col, y_col, title=None):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=x_col, y=y_col, palette='Set2')
    plt.title(title if title else f'{y_col} Distribution by {x_col}')
    plt.tight_layout()
    plt.show()

def plot_countplot(df,column,title=None):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column, palette='muted')
    plt.title(title if title else f'Distribution of {column}')
    plt.tight_layout()
    plt.show()


def plot_histplot(df,column,title=None):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, kde=True, color='teal')
    plt.title(title if title else f'Distribution of {column}')
    plt.tight_layout()
    plt.show()
