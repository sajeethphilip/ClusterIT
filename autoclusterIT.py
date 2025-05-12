import pandas as pd
import numpy as np
from hdbscan import HDBSCAN, validity
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from functools import partial

def automated_clustering(csv_path, label_col=None):
    """Automatically finds optimal clusters and generates results"""
    # Load data
    df = pd.read_csv(csv_path)
    features = df.drop(columns=[label_col] if label_col else []).values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Define search space
    param_bounds = {
        'min_cluster_size': (2, 50),
        'min_samples': (1, 20)
    }

    # Optimization objective
    def hdbscan_score(min_cluster_size, min_samples):
        clusterer = HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(X)
        return validity.validity_index(X, labels)  # DBCV score

    # Bayesian optimization
    optimizer = BayesianOptimization(
        f=hdbscan_score,
        pbounds=param_bounds,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=15)

    # Get best parameters
    best_params = optimizer.max['params']
    final_clusterer = HDBSCAN(
        min_cluster_size=int(best_params['min_cluster_size']),
        min_samples=int(best_params['min_samples']),
        cluster_selection_method='eom'
    )
    labels = final_clusterer.fit_predict(X)

    # Generate output
    result_df = df.copy()
    result_df['Cluster'] = labels

    # Label shifting logic
    if label_col and pd.api.types.is_numeric_dtype(df[label_col]):
        max_label = df[label_col].max()
        valid_clusters = labels >= 0
        result_df.loc[valid_clusters, 'Cluster'] += (max_label + 1)

    return result_df, best_params

# Usage
csv_file = input("Enter CSV file path: ")
label_col = input("Enter label column (or Enter to skip): ").strip() or None

result_df, params = automated_clustering(csv_file, label_col)

# Save results
output_csv = csv_file.replace('.csv', '_auto_clustered.csv')
result_df.to_csv(output_csv, index=False)

print(f"\nOptimal parameters found:")
print(f"min_cluster_size: {params['min_cluster_size']:.0f}")
print(f"min_samples: {params['min_samples']:.0f}")
print(f"Saved results to {output_csv}")
