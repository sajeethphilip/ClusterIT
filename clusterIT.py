# cluster_analysis.py
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=UserWarning)  # Suppress HDBSCAN warnings

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

# Shift cluster labels if numeric labels exist
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

# Create result dataframe
result_df = df.copy()
result_df['Cluster'] = cluster_labels
result_df['Cluster_str'] = result_df['Cluster'].astype(str)
result_df[proj_cols[0]] = projection[:, 0]
result_df[proj_cols[1]] = projection[:, 1]

# --------------------------
# Interactive Visualization
# --------------------------
fig = px.scatter(
    result_df,
    x=proj_cols[0],
    y=proj_cols[1],
    color='Cluster_str',
    hover_data=df.columns.tolist(),
    title=f"Cluster Visualization {shift_info}<br><sub>{proj_cols[0]}-{proj_cols[1]}</sub>",
    color_discrete_sequence=px.colors.qualitative.Dark24,
    labels={'Cluster_str': 'Cluster'}
)

fig.update_layout(
    dragmode='pan',
    hovermode='closest',
    plot_bgcolor='rgba(240,240,240,0.9)',
    height=800
)

# --------------------------
# Save Results
# --------------------------
output_csv = csv_file.replace('.csv', '_clustered.csv')
result_df.to_csv(output_csv, index=False)

html_output = csv_file.replace('.csv', '_interactive.html')
fig.write_html(html_output)

print(f"\nResults saved to:\n- {output_csv}\n- {html_output}")
print("\nOpen the HTML file in a browser to explore clusters interactively!")
print("Note: Noise points are shown as '-1' in cluster labels")
