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
import os

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
# Noise Relabeling Decision
# --------------------------
result_df = df.copy()
result_df['Cluster'] = cluster_labels

noise_mask = result_df['Cluster'] == -1
n_noise = noise_mask.sum()

if n_noise > 0:
    print(f"\nFound {n_noise} noise points ({n_noise/len(result_df):.1%} of data)")
    relabel = input("Treat noise points as a new cluster? (yes/no): ").strip().lower()

    if relabel == 'yes':
        existing_clusters = result_df['Cluster'].unique()
        valid_clusters = existing_clusters[existing_clusters >= 0]
        new_cluster_id = valid_clusters.max() + 1 if valid_clusters.size > 0 else 0
        result_df.loc[noise_mask, 'Cluster'] = new_cluster_id
        print(f"Relabeled noise as Cluster {new_cluster_id}")

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

result_df[proj_cols[0]] = projection[:, 0]
result_df[proj_cols[1]] = projection[:, 1]

# --------------------------
# Create Visualizations
# --------------------------
def create_visualizations():
    # Create cluster labels with proper names
    result_df['Cluster_str'] = result_df['Cluster'].apply(
        lambda x: f"Cluster {x}" if x != -1 else "Noise"
    )

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
    plot_df = result_df.copy()

    # Process clusters
    clusters_df = plot_df[plot_df['Cluster'] >= 0]
    if not clusters_df.empty:
        cluster_means = clusters_df.groupby('Cluster')[feature_names].mean()
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

    # Process noise (if still exists)
    noise_df = plot_df[plot_df['Cluster'] == -1]
    if not noise_df.empty:
        noise_mean = noise_df[feature_names].mean()
        noise_normalized = (noise_mean - noise_mean.min())/(noise_mean.max() - noise_mean.min())
        if noise_mean.max() != noise_mean.min(): noise_mean=noise_normalized
        coadd_fig.add_trace(go.Scatter(
            x=feature_names,
            y=noise_normalized,
            mode='lines',
            name='Noise',
            line=dict(dash='dot'),
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
# Generate Example Time Series Plots per Cluster
# --------------------------
example_files = []
color_map = {}
if label_col:
    unique_labels = df[label_col].unique()
    color_palette = px.colors.qualitative.Dark24
    color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}

clusters_to_plot = result_df['Cluster'].unique()

for cluster_id in clusters_to_plot:
    cluster_data = result_df[result_df['Cluster'] == cluster_id]
    if len(cluster_data) == 0:
        continue

    n_samples = min(5, len(cluster_data))
    samples = cluster_data.sample(n=n_samples, random_state=42)

    fig = go.Figure()
    for _, row in samples.iterrows():
        x = feature_names
        y = row[feature_names].values.astype(float)

        line_color = 'gray'  # Default for unlabeled data
        if label_col and color_map:
            label_value = row[label_col]
            line_color = color_map.get(label_value, 'gray')

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color=line_color),
            name=f"{label_col}: {label_value}" if label_col else "Sample",
            showlegend=label_col is not None
        ))

    cluster_title = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
    fig.update_layout(
        title=f"{cluster_title} Examples",
        xaxis_title="Feature",
        yaxis_title="Value",
        showlegend=label_col is not None,
        legend_title=label_col if label_col else None,
        height=600,
        width=1200
    )

    example_filename = f"{base_name}_cluster_{cluster_id}_examples.png"
    fig.write_image(example_filename, engine="kaleido")
    example_files.append(example_filename)

# --------------------------
# Print Cluster Statistics
# --------------------------
cluster_counts = result_df['Cluster'].value_counts().sort_index()
count_report = ["\nCluster membership:"]
label_insights = []

for cluster_id, count in cluster_counts.items():
    display_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
    count_report.append(f"- {display_name}: {count} rows")

    if label_col and cluster_id != -1:
        cluster_data = result_df[result_df['Cluster'] == cluster_id]
        label_counts = cluster_data[label_col].value_counts()
        total = len(cluster_data)
        top_labels = label_counts[label_counts > 0].sort_values(ascending=False)

        if not top_labels.empty:
            insights = [
                f"   ⚡ {label}: {cnt} ({cnt/total:.1%})"
                for label, cnt in top_labels.items()
            ]
            label_insights.append(f"{display_name} composition:\n" + "\n".join(insights))

# Noise analysis (if still exists)
if -1 in cluster_counts:
    noise_count = cluster_counts[-1]
    if label_col:
        noise_counts = result_df[result_df['Cluster'] == -1][label_col].value_counts()
        total_noise = noise_counts.sum()
        top_noise = noise_counts[noise_counts > 0].sort_values(ascending=False)

        if not top_noise.empty:
            insights = [
                f"   🔊 {label}: {cnt} ({cnt/total_noise:.1%})"
                for label, cnt in top_noise.items()
            ]
            label_insights.append("Noise composition:\n" + "\n".join(insights))

# Print outputs
print(f"\nResults saved to:")
print(f"- Clustered data: {data_output}")
print(f"- Cluster visualization: {cluster_image}")
print(f"- Coadded spectra: {coadd_image}")
for file in example_files:
    print(f"- Example time series plot: {file}")
print('\n'.join(count_report))

if label_insights:
    print("\nLabel Distribution Insights:")
    print('\n\n'.join(label_insights))
