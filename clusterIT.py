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
# Create Visualizations
# --------------------------
def create_visualizations():
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
        height=800,
        width=1200
    )

    # Coadded spectra plot
    coadd_fig = go.Figure()
    plot_df = result_df[result_df['Cluster'] >= 0]

    if not plot_df.empty:
        cluster_means = plot_df.groupby('Cluster')[feature_names].mean()
        cluster_means = cluster_means.apply(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x,
            axis=1
        )

        for cluster_id in cluster_means.index:
            coadd_fig.add_trace(go.Scatter(
                x=feature_names,
                y=cluster_means.loc[cluster_id],
                mode='lines',
                name=f'Cluster {cluster_id}',
                opacity=0.7
            ))

    coadd_fig.update_layout(
        title="Normalized Coadded Spectra by Cluster",
        xaxis_title="Frequency Index",
        yaxis_title="Normalized Power",
        height=600,
        width=1200,
        showlegend=True
    )

    return cluster_fig, coadd_fig

# Generate figures
cluster_fig, coadd_fig = create_visualizations()

# --------------------------
# Save All Outputs
# --------------------------
base_name = csv_file.rsplit('.', 1)[0]

# Save data
data_output = f"{base_name}_clustered.csv"
result_df.to_csv(data_output, index=False)

# Save images
cluster_image = f"{base_name}_cluster_vis.png"
coadd_image = f"{base_name}_coadded_spectra.png"

cluster_fig.write_image(cluster_image, engine="kaleido")
coadd_fig.write_image(coadd_image, engine="kaleido")

# --------------------------
# Print Cluster Statistics
# --------------------------
# Get cluster counts, including noise (-1)
cluster_counts = result_df['Cluster'].value_counts().sort_index()

# Format output message
count_report = ["\nCluster membership:"]
for cluster_id, count in cluster_counts.items():
    if cluster_id == -1:
        count_report.append(f"- Noise points: {count} rows")
    else:
        count_report.append(f"- Cluster {cluster_id}: {count} rows")

print(f"\nResults saved to:")
print(f"- Clustered data: {data_output}")
print(f"- Cluster visualization: {cluster_image}")
print(f"- Coadded spectra: {coadd_image}")
print('\n'.join(count_report))
print("\nNote: Requires kaleido package for PNG export")

