import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
iris = load_iris()
print("Keys in the dataset:", iris.keys())
print("-------------------------------------------------------------")
print("Dataset Description:\n", iris['DESCR'])
print("-------------------------------------------------------------")
print("Iris Target Dataset:\n",iris.target)

df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("DataFrame with features:")
print(df)
print("-------------------------------------------------------------")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
print("Standardized Data:")
print(scaled_data)
print("-------------------------------------------------------------")

pca = PCA(n_components=2)
pca = PCA(0.95)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print("Original Data Shape:", scaled_data.shape)
print("-------------------------------------------------------------")
print("Transformed Data Shape (after PCA):", x_pca.shape)
print("-------------------------------------------------------------")

#Scatter Plot Graph
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=iris.target, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Iris Dataset')
plt.colorbar(label='Target Class')
plt.show()

#Variance Plot Graph
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 6))

plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o',
linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Principal Components')
plt.grid(True)
plt.show()
