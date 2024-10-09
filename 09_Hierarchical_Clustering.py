import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

df = pd.read_csv('Shopping_Trends.csv')

# Exclude non-numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
X = df[numeric_columns].values

# Perform hierarchical clustering
linkage_matrix = linkage(X, 'ward')
dendrogram(linkage_matrix, labels=df['Customer ID'].values, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer ID')
plt.ylabel('Distance')
plt.show()

# Fit the hierarchical clustering model
n_clusters = 3
cluster_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
clusters = cluster_model.fit_predict(X)

# Add cluster labels to the original dataframe
df['Cluster'] = clusters

# Display the clustered data
print(df)
