"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def plot_relational_plot(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['popularity'], y=df['rating'])
    plt.xlabel('Popularity')
    plt.ylabel('Rating')
    plt.title('Popularity vs Rating')
    plt.savefig('relational_plot.png')

def plot_categorical_plot(df):
    plt.figure(figsize=(8, 6))
    sns.histplot(df['type'], discrete=True)
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.title('Distribution of Movie Types')
    plt.savefig('categorical_plot.png')

def plot_statistical_plot(df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['popularity', 'vote_count', 'vote_average', 'budget', 'revenue']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('statistical_plot.png')

def statistical_analysis(df, col):
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    df = df.drop(columns=['duration'], errors='ignore')
    df = df.dropna() 
    return df


def perform_clustering(df, col1, col2):
    X = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, X[col1], X[col2], kmeans.cluster_centers_[:, 0], 
      kmeans.cluster_centers_[:, 1]

def plot_clustered_data(labels, x, y, xkmeans, ykmeans):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, hue=labels, palette='viridis')
    plt.scatter(xkmeans, ykmeans, c='red', marker='X', s=100, label='Centroids')
    plt.xlabel('Popularity')  # Explicit column name
    plt.ylabel('Vote Average')  # Explicit column name
    plt.title('Clustering of Movies')
    plt.legend()
    plt.savefig('clustering.png')

def perform_fitting(df, col1, col2):
    X = df[[col1]].dropna()
    y = df[col2].loc[X.index]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    return X[col1], y, y_pred

def plot_fitted_data(X, y, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label='Actual Data')
    plt.plot(X, y_pred, color='red', label='Fitted Line')
    plt.xlabel('Budget')  # Explicit column name
    plt.ylabel('Revenue')  # Explicit column name
    plt.title('Linear Regression: Budget vs Revenue')
    plt.legend()
    plt.savefig('fitting.png')

def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    
    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)
    
    moments = statistical_analysis(df, 'popularity')
    print(f'Statistics for Popularity: {moments}')
    
    labels, x, y, xkmeans, ykmeans = perform_clustering(df, 'popularity', 'vote_average')
    plot_clustered_data(labels, x, y, xkmeans, ykmeans)
    
    X, y, y_pred = perform_fitting(df, 'budget', 'revenue')
    plot_fitted_data(X, y, y_pred)

if __name__ == '__main__':
    main()
