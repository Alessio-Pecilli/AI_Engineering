# ==============================
# IMPORT LIBRARIES
# ==============================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
"""
# ==============================
# PLOT SETTINGS
# ==============================
plt.rcParams['figure.figsize'] = (16, 10)
sns.set_theme()
RANDOM_SEED = 2

# ==============================
# GENERATE SYNTHETIC DATASET
# ==============================
X, _ = make_blobs(
    n_samples=100, centers=3, n_features=2,
    cluster_std=0.50, random_state=RANDOM_SEED
)

# Stretch dataset a bit to make clusters less symmetric
X[:, 0] *= 20
X[:, 1] *= 6

sns.scatterplot(x=X[:, 0], y=X[:, 1], s=100)
plt.title("Clustering dataset")
plt.show()

# ==============================
# K-MEANS CLUSTERING
# ==============================
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=RANDOM_SEED)
kmeans.fit(X)

# Cluster assignments
y_kmeans = kmeans.predict(X)

# ==============================
# VISUALIZE CLUSTERS
# ==============================
cluster_map = {0: 'red', 1: 'blue', 2: 'green'}
colors = [cluster_map[label] for label in y_kmeans]

plt.scatter(X[:, 0], X[:, 1], c=colors, s=20, label="Points")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="black", s=200, marker="X", label="Centroids"
)
plt.title("K-Means Clustering")
plt.legend()
plt.show()

# ==============================
# DISTORTION / INERTIA
# ==============================
# Distortion: average distance to closest centroid
distortion = sum(
    np.sqrt(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))
)
# Inertia: sum of squared distances (from sklearn)
inertia = kmeans.inertia_

print("Distortion:", distortion)
print("Inertia:", inertia)

# ==============================
# ELBOW METHOD
# ==============================
ssd = {}  # Sum of squared distances for each k

for k in range(1, 10):
    km = KMeans(n_clusters=k, init='k-means++', random_state=RANDOM_SEED)
    km.fit(X)
    ssd[k] = km.inertia_

plt.plot(list(ssd.keys()), list(ssd.values()), marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of squared distances (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.show()
"""

# ==============================
# PLOT SETTINGS
# ==============================
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_theme()

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("mall_customers.csv")

# Select only two features for visualization
X = df[["Age", "Spending Score (1-100)"]].values

# ==============================
# ELBOW METHOD
# ==============================
ssd = {}  # Sum of squared distances (inertia) for each k

for k in range(1, 10):
    km = KMeans(n_clusters=k, init="k-means++", random_state=42)
    km.fit(X)
    ssd[k] = km.inertia_

# Plot Elbow Curve
plt.plot(list(ssd.keys()), list(ssd.values()), marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of squared distances (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.show()

# ==============================
# FIT FINAL MODEL (k=4 chosen from elbow)
# ==============================
k = 4
kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
kmeans.fit(X)

# Cluster assignments
y_kmeans = kmeans.predict(X)

# ==============================
# VISUALIZE CLUSTERS
# ==============================
cluster_map = {0: "red", 1: "blue", 2: "green", 3: "purple"}
colors = [cluster_map[label] for label in y_kmeans]

plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.6, edgecolors="k", label="Points")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="black", s=250, marker="X", label="Centroids"
)
plt.title("K-Means Clustering (Mall Customers)")
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
