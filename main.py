import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt
# import random


def parse_args():
    parser = argparse.ArgumentParser(description='Run random forrest with specified input arguments')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='Heart-counts.csv',
                        help='data path')

    a = parser.parse_args()
    return (a.n_clusters, a.data)


def read_data(data_path):
    return anndata.read_csv(data_path)


def preprocess_data(adata: anndata.AnnData, scale: bool = True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)


def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)
    # Your code
    # random
    all_silhouette = []
    for n_classifiers in range(2, 10):
        kmeans = KMeans(n_classifiers, init='random')
        res = kmeans.fit(X)
        s = kmeans.silhouette(res, X)
        all_silhouette.append(s)

    plt.plot([i for i in range(2, 10)], all_silhouette)
    plt.title('random method')
    plt.xlabel('k')
    plt.ylabel('silhouette')
    plt.show()

    # kmeans++
    all_silhouette_2 = []
    for n_classifiers in range(2, 10):
        kmeans = KMeans(n_classifiers, init='kmeans++')
        res = kmeans.fit(X)
        s = kmeans.silhouette(res, X)
        all_silhouette_2.append(s)

    plt.plot([i for i in range(2, 10)], all_silhouette_2)
    plt.title('kmeans++ method')
    plt.xlabel('k')
    plt.ylabel('silhouette')
    plt.show()

    # the best K is 7
    kmeans = KMeans(7, 'kmeans++')
    res = kmeans.fit(X)
    clustering=None
    visualize_cluster(heart.X, res, clustering)


def visualize_cluster(x, y, clustering):
    # Your code
    X_reduced = PCA(x, 2)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, marker='+', cmap=plt.cm.Spectral)
    plt.title('heart data reduced')
    plt.xlabel('dimension1')
    plt.ylabel('dimension2')
    plt.show()


if __name__ == '__main__':
    main()
