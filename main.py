import torch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the data
dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv')
X = dataset.iloc[:, 3:].values

# Convert data to PyTorch tensor
X_tensor = torch.FloatTensor(X)

# Using dendrogram to find the optimal number of clusters
linked = linkage(X, method='ward')

# Plotting the dendrogram
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Customers')  # observation points
plt.ylabel('Euclidean Distance')
plt.show()

# Implementing Hierarchical Clustering manually
def hierarchical_clustering(X, num_clusters):
    clusters = torch.arange(len(X)).unsqueeze(1).float()

    while len(clusters) > num_clusters:
        min_distance = float('inf')
        merge_indices = None

        for i in range(len(clusters) - 1):
            for j in range(i + 1, len(clusters)):
                distance = torch.dist(X[clusters[i].long()], X[clusters[j].long()])
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        i, j = merge_indices
        new_cluster = torch.cat((clusters[i], clusters[j]))
        clusters = torch.cat((clusters[:i], clusters[i + 1:j], clusters[j + 1:], new_cluster.unsqueeze(0)))

    return clusters

# Getting the cluster assignments
num_clusters = 5
cluster_assignments = hierarchical_clustering(X_tensor, num_clusters)

# Visualizing the clusters
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(num_clusters):
    plt.scatter(X_tensor[cluster_assignments[i].long()][:, 0], X_tensor[cluster_assignments[i].long()][:, 1],
                s=100, c=colors[i], label=f'Cluster {i + 1}')

plt.title('Clusters of Customers (Manual Hierarchical Clustering)')
plt.xlabel('Annual income in ($k)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
