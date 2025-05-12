import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN, validity
from bayes_opt import BayesianOptimization

def automated_clustering(csv_path, label_col=None):
    """Complete automated clustering with coadded spectra"""
    # Load and prepare data
    df = pd.read_csv(csv_path)  # This was missing in previous implementation
    features = df.drop(columns=[label_col]) if label_col else []
    feature_columns = features.columns.tolist()
    X = StandardScaler().fit_transform(features)

    # Bayesian optimization
    def hdbscan_score(min_cluster_size, min_samples):
        clusterer = HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(X)
        return validity.validity_index(X, labels)

    optimizer = BayesianOptimization(
        f=hdbscan_score,
        pbounds={'min_cluster_size': (2, 50), 'min_samples': (1, 20)},
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=15)

    # Final clustering
    best_params = optimizer.max['params']
    clusterer = HDBSCAN(
        min_cluster_size=int(best_params['min_cluster_size']),
        min_samples=int(best_params['min_samples']),
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(X)

    # Create result dataframe
    result_df = df.copy()
    result_df['Cluster'] = cluster_labels

    # Label shifting logic
    if label_col and pd.api.types.is_numeric_dtype(df[label_col]):
        max_label = df[label_col].max()
        valid_clusters = cluster_labels >= 0
        result_df.loc[valid_clusters, 'Cluster'] += (max_label + 1)

    # Generate coadded spectra plot
    def plot_coadded_spectra():
        plot_df = result_df[result_df['Cluster'] >= 0].copy()
        cluster_means = plot_df.groupby('Cluster')[feature_columns].mean()

        # Normalize each cluster's spectrum
        cluster_means = cluster_means.apply(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x,
            axis=1
        )

        fig = go.Figure()
        for cluster_id in cluster_means.index:
            fig.add_trace(go.Scatter(
                x=feature_columns,
                y=cluster_means.loc[cluster_id],
                mode='lines',
                name=f'Cluster {cluster_id}',
                opacity=0.7
            ))
        fig.update_layout(
            title="Normalized Coadded Spectra by Cluster",
            xaxis_title="Frequency Index",
            yaxis_title="Normalized Power",
            height=600,
            width=1000
        )
        return fig

    return result_df, best_params, plot_coadded_spectra()

# Usage remains the same
csv_file = input("Enter CSV file path: ")
label_col = input("Enter label column (or Enter to skip): ").strip() or None

result_df, params, coadd_fig = automated_clustering(csv_file, label_col)

# Save outputs
output_csv = csv_file.replace('.csv', '_clustered.csv')
result_df.to_csv(output_csv, index=False)

coadd_html = csv_file.replace('.csv', '_coadded.html')
coadd_fig.write_html(coadd_html)
coadd_fig.write_image(coadd_html.replace('.html', '.png'))

print(f"Results saved to:\n- {output_csv}\n- {coadd_html}\n- {coadd_html.replace('.html', '.png')}")
