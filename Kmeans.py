import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成样本数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 应用 K-means 算法
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# 预测簇标签
labels = kmeans.predict(X)

# 可视化结果
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75) # 显示中心点
plt.show()






