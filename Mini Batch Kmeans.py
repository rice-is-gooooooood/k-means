from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成数据
X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=0.60, random_state=0)

# 创建 MiniBatchKMeans 实例
mbk = MiniBatchKMeans(n_clusters=4, batch_size=100)

# 拟合数据
mbk.fit(X)

# 预测簇标签
labels = mbk.predict(X)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
centers = mbk.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')  # 显示中心点
plt.title('Mini-Batch K-Means Clustering')
plt.show()

