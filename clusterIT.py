# cluster_analysis.py
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --------------------------
# User Input Section
# --------------------------
csv_file = input("Enter path to CSV file: ")
df = pd.read_csv(csv_file)

print("\nColumns in your dataset:")
print('\n'.join(df.columns))

label_col = input("\nEnter name of the label column (or press Enter if none): ").strip()
if label_col and label_col not in df.columns:
    print(f"Error: Column '{label_col}' not found!")
    exit()

# --------------------------
# Data Preparation
# --------------------------
features = df.drop(columns=[label_col] if label_col else []).values
feature_names = df.columns.tolist()
if label_col:
    feature_names.remove(label_col)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --------------------------
# Clustering with Label Shifting
# --------------------------
clusterer = HDBSCAN(
    min_cluster_size=int(input("\nMin cluster size (default 5): ") or 5),
    min_samples=int(input("Min samples (default 3): ") or 3),
    cluster_selection_method='eom'
)
cluster_labels = clusterer.fit_predict(scaled_features)

shift_info = ""
if label_col and pd.api.types.is_numeric_dtype(df[label_col]):
    max_label = df[label_col].max()
    mask = cluster_labels >= 0
    cluster_labels[mask] += (max_label + 1)
    if np.issubdtype(df[label_col].dtype, np.integer):
        cluster_labels = cluster_labels.astype(int)
    shift_info = f" (shifted from existing labels, starting at {max_label+1})"

# --------------------------
# Dimensionality Reduction
# --------------------------
projection_choice = input("Projection method [PCA/t-SNE] (default PCA): ").strip().lower()

if projection_choice == "t-sne":
    proj = TSNE(n_components=2, perplexity=30, random_state=42)
    proj_cols = ['tSNE1', 'tSNE2']
else:
    proj = PCA(n_components=2)
    proj_cols = ['PCA1', 'PCA2']

projection = proj.fit_transform(scaled_features)

result_df = df.copy()
result_df['Cluster'] = cluster_labels
result_df['Cluster_str'] = result_df['Cluster'].astype(str)
result_df[proj_cols[0]] = projection[:, 0]
result_df[proj_cols[1]] = projection[:, 1]

# --------------------------
# Interactive Visualizations
# --------------------------
# Cluster projection plot
cluster_fig = px.scatter(
    result_df,
    x=proj_cols[0],
    y=proj_cols[1],
    color='Cluster_str',
    hover_data=df.columns.tolist(),
    title=f"Cluster Visualization {shift_info}<br><sub>{proj_cols[0]}-{proj_cols[1]}</sub>",
    color_discrete_sequence=px.colors.qualitative.Dark24,
    labels={'Cluster_str': 'Cluster'}
)

cluster_fig.update_layout(
    dragmode='pan',
    hovermode='closest',
    plot_bgcolor='rgba(240,240,240,0.9)',
    height=800
)

# Coadded spectra plot
def create_coadded_plot():
    plot_df = result_df[result_df['Cluster'] >= 0].copy()
    if plot_df.empty:
        print("No clusters found for coadded spectra")
        return go.Figure()

    cluster_means = plot_df.groupby('Cluster')[feature_names].mean()
    cluster_means = cluster_means.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x,
        axis=1
    )

    fig = go.Figure()
    for cluster_id in cluster_means.index:
        fig.add_trace(go.Scatter(
            x=feature_names,
            y=cluster_means.loc[cluster_id],
            mode='lines',
            name=f'Cluster {cluster_id}',
            opacity=0.7,
            hovertemplate='Frequency: %{x}<br>Power: %{y:.2f}'
        ))

    fig.update_layout(
        title="Normalized Coadded Spectra by Cluster",
        xaxis_title="Frequency Index",
        yaxis_title="Normalized Power",
        height=600,
        width=1000,
        showlegend=True
    )
    return fig

coadd_fig = create_coadded_plot()

# --------------------------
# Save Results
# --------------------------
output_csv = csv_file.replace('.csv', '_clustered.csv')
result_df.to_csv(output_csv, index=False)

cluster_html = csv_file.replace('.csv', '_cluster_vis.html')
cluster_fig.write_html(cluster_html)
cluster_fig.write_image(cluster_html.replace('.html', '.png'))

coadd_html = csv_file.replace('.csv', '_coadded_spectra.html')
coadd_fig.write_html(coadd_html)
coadd_fig.write_image(coadd_html.replace('.html', '.png'))

print(f"\nResults saved to:")
print(f"- Clustered data: {output_csv}")
print(f"- Cluster visualization: {cluster_html}")
print(f"- Coadded spectra: {coadd_html}")
print("\nOpen HTML files in browser to interact with visualizations!")
