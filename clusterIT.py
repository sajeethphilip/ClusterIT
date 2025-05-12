# cluster_analysis.py
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.preprocessing import StandardScaler

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
# Modern Clustering with HDBSCAN 0.8.33+
# --------------------------
clusterer = HDBSCAN(
    min_cluster_size=int(input("\nMin cluster size (default 5): ") or 5),
    min_samples=int(input("Min samples (default 3): ") or 3),
    cluster_selection_method='eom',
    prediction_data=True  # Enable new features in latest HDBSCAN
)
cluster_labels = clusterer.fit_predict(scaled_features)

# --------------------------
# Visualize with PCA/TSNE (No UMAP/TensorFlow)
# --------------------------
projection_choice = input("Projection method [PCA/t-SNE] (default PCA): ").strip().lower()

if projection_choice == "t-sne":
    proj = TSNE(n_components=2, perplexity=30, random_state=42)
    projection = proj.fit_transform(scaled_features)
    proj_cols = ['tSNE1', 'tSNE2']
else:
    proj = PCA(n_components=2)
    projection = proj.fit_transform(scaled_features)
    proj_cols = ['PCA1', 'PCA2']

result_df = df.assign(
    Cluster=cluster_labels.astype(str),
    **{proj_cols[0]: projection[:, 0], proj_cols[1]: projection[:, 1]}
)

# --------------------------
# Interactive Visualization
# --------------------------
fig = px.scatter(
    result_df,
    x=proj_cols[0],
    y=proj_cols[1],
    color='Cluster',
    hover_data=df.columns.tolist(),
    title=f"Cluster Visualization ({proj_cols[0]}-{proj_cols[1]})",
    color_discrete_sequence=px.colors.qualitative.Dark24
)
fig.update_layout(dragmode='pan', hovermode='closest')
fig.write_html(csv_file.replace('.csv', '_interactive.html'))
fig.show()

# --------------------------
# Save Results
# --------------------------
output_csv = csv_file.replace('.csv', '_clustered.csv')
result_df.to_csv(output_csv, index=False)
print(f"\nResults saved to:\n- {output_csv}\n- {csv_file.replace('.csv', '_interactive.html')}")
