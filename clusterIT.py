import pandas as pd
import numpy as np
import hdbscan
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import ipywidgets as widgets
from IPython.display import display

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

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --------------------------
# Clustering with HDBSCAN
# --------------------------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=int(input("\nEnter minimum cluster size (default 5): ") or 5),
    min_samples=int(input("Enter minimum samples (default 3): ") or 3),
    cluster_selection_method='eom'
)
cluster_labels = clusterer.fit_predict(scaled_features)

# Add clustering results to DataFrame
result_df = df.copy()
result_df['Cluster'] = cluster_labels
result_df['Cluster'] = result_df['Cluster'].astype(str)  # For better plotting

# --------------------------
# Dimensionality Reduction
# --------------------------
reducer = umap.UMAP(n_components=2, random_state=42)
projection = reducer.fit_transform(scaled_features)

# Add projection coordinates
result_df['UMAP1'] = projection[:, 0]
result_df['UMAP2'] = projection[:, 1]

# --------------------------
# Save Results
# --------------------------
output_csv = csv_file.replace('.csv', '_clustered.csv')
result_df.to_csv(output_csv, index=False)
print(f"\nSaved clustered data to: {output_csv}")

# --------------------------
# Interactive Visualization
# --------------------------
def create_figure(x_axis='UMAP1', y_axis='UMAP2'):
    fig = px.scatter(
        result_df,
        x=x_axis,
        y=y_axis,
        color='Cluster',
        hover_data=df.columns.tolist(),
        title="Cluster Visualization",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig.update_layout(
        dragmode='pan',
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.9)'
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    return fig

# Create interactive controls
axis_selector = widgets.Dropdown(
    options=[('UMAP Projection', 'UMAP'), ('PCA Projection', 'PCA')],
    value='UMAP',
    description='Projection:'
)

@widgets.interact(Projection=axis_selector)
def update_plot(Projection):
    if Projection == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        projection = pca.fit_transform(scaled_features)
        result_df['PCA1'] = projection[:, 0]
        result_df['PCA2'] = projection[:, 1]
        return create_figure('PCA1', 'PCA2').show()
    else:
        return create_figure().show()

# --------------------------
# Static Visualization
# --------------------------
print("\nSaving static visualizations...")
fig = create_figure()
fig.write_html(csv_file.replace('.csv', '_cluster_interactive.html'))
fig.write_image(csv_file.replace('.csv', '_cluster.png'), width=1200, height=800)

print("""
Operation complete!
- Open the HTML file for interactive visualization
- Check PNG file for static visualization
- Clustered CSV contains all original data + cluster labels
""")
