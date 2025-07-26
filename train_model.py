"""
train_model.py
This script trains a Random Forest classifier on FFT + PCA features from time-series data.
It also saves the trained pipeline (FFT -> StandardScaler -> PCA -> RandomForest).
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# -------------------------
# Custom FFT Transformer
# -------------------------
class FFTTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        return np.abs(fft(X))[:, :self.n_features_ // 2]


# -------------------------
# Helper Functions
# -------------------------
def plot_pca_variance(X_scaled, output_path="pca_variance.png"):
    """Plot cumulative explained variance to choose PCA components."""
    pca = PCA()
    pca.fit(X_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of PCA Components')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"PCA variance plot saved to {output_path}")


def plot_clusters(X_pca, clusters, output_path="kmeans_clusters.png"):
    """Plot KMeans clusters on first 2 PCA components."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1')
    plt.title("K-Means Clusters Based on Spectral Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"KMeans cluster plot saved to {output_path}")


# -------------------------
# Training Pipeline
# -------------------------
def train_model(csv_path="SleepTrain5000.csv", n_pca=60, n_clusters=5):
    """
    Train Random Forest with FFT + PCA features, using KMeans cluster labels.
    Saves rf_fft_pca_pipeline.joblib.
    """
    # 1. Load data
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 1:].values
    print(f"Data shape: {data.shape}")

    # 2. FFT + Scaling
    fft_features = np.abs(fft(data))[:, :data.shape[1] // 2]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fft_features)

    # 3. PCA for visualization
    plot_pca_variance(X_scaled)
    pca_full = PCA(n_components=n_pca)
    X_pca = pca_full.fit_transform(X_scaled)

    # 4. KMeans clustering (generate pseudo-labels)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    plot_clusters(X_pca, clusters)

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        data, clusters, stratify=clusters, test_size=0.2, random_state=42
    )

    # 6. Pipeline with FFT + Scaling + PCA + RF
    pipeline = Pipeline(steps=[
        ('fft', FFTTransformer()),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_pca)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # 7. Grid Search for RF
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [10, 20, 30, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__bootstrap': [True, False]
    }

    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, cv=5,
        n_jobs=-1, verbose=1, scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # 8. Evaluation
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 9. Save pipeline
    joblib.dump(best_model, "rf_fft_pca_pipeline.joblib")
    print("Saved trained pipeline as rf_fft_pca_pipeline.joblib")


if __name__ == "__main__":
    train_model()
