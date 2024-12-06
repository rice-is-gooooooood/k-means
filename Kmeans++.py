import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. 生成样本数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. 应用 K-means (默认使用随机初始化)
kmeans_random = KMeans(n_clusters=4, init='random', n_init=10, random_state=0)
kmeans_random.fit(X)
labels_random = kmeans_random.labels_
centers_random = kmeans_random.cluster_centers_

# 3. 应用 K-means++ (默认使用K-means++)
kmeans_pp = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=0)
kmeans_pp.fit(X)
labels_pp = kmeans_pp.labels_
centers_pp = kmeans_pp.cluster_centers_

# 4. 可视化 K-means 和 K-means++ 的聚类结果
plt.figure(figsize=(12, 6))

# 子图1: K-means 随机初始化
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels_random, s=50, cmap='viridis')
plt.scatter(centers_random[:, 0], centers_random[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-means (Random Initialization)")

# 子图2: K-means++ 初始化
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_pp, s=50, cmap='viridis')
plt.scatter(centers_pp[:, 0], centers_pp[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-means++ Initialization")

# 显示结果
plt.show()