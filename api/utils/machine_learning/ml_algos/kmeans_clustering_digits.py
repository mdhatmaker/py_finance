from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# https://www.askpython.com/python/examples/top-machine-learning-algorithms
# https://www.askpython.com/python/examples/plot-k-means-clusters-python


"""
Digits dataset contains images of size 8×8 pixels, which is flattened to create a feature vector of
length 64. We used PCA to reduce the number of dimensions so that we can visualize the results using
a 2D Scatter plot.
"""
def prepare_data_digits():
    # Load Data
    data = load_digits().data
    pca = PCA(2)
    # Transform the data
    df = pca.fit_transform(data)
    print(df.shape)
    return df


"""
Here in the digits dataset we already know that the labels range from 0 to 9, so we have 10 classes
(or clusters).

But in real-life challenges when performing K-means the most challenging task is to determine the
number of clusters.

There are various methods to determine the optimum number of clusters, i.e. Elbow method, Average
Silhouette method. But determining the number of clusters will be the subject of another talk.
"""
# Apply K-mean to our data to create clusters.
def apply_kmeans(df, n_clusters=10):
    # Initialize the class object
    kmeans = KMeans(n_clusters=n_clusters)
    # predict the labels of clusters.
    label = kmeans.fit_predict(df)
    print(label)
    # kmeans.fit_predict method returns the array of cluster labels each data point belongs to.
    return label, kmeans


def visualize_cluster_label(df, label, label_numbers):
    for i in label_numbers:
        # filter rows of original data
        filtered_label = df[label == i]
        # plotting the results
        plt.scatter(filtered_label[:, 0], filtered_label[:, 1], label=i)
    plt.show()


def visualize_cluster_all_labels(df, label):
    # Getting unique labels
    u_labels = np.unique(label)
    # plotting the results:
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.legend()
    plt.show()


def plot_cluster_centroids(df, label, kmeans):
    # Getting the Centroids
    centroids = kmeans.cluster_centers_
    u_labels = np.unique(label)
    # plotting the results:
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    plt.legend()
    plt.show()


def run_plot_kmeans_clustering_digits():
    df = prepare_data_digits()
    label, kmeans = apply_kmeans(df, n_clusters=10)
    visualize_cluster_label(df, label, [0])
    visualize_cluster_label(df, label, [2, 8])
    visualize_cluster_all_labels(df, label)
    plot_cluster_centroids(df, label, kmeans)


if __name__ == "__main__":
    run_plot_kmeans_clustering_digits()

