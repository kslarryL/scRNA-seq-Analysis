import numpy as np
import random

class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        while iteration < self.max_iter:
            # your code
            dist = self.euclidean_distance(X, self.centroids)
            clustering = dist.argmin(axis=1)
            self.update_centroids(clustering, X)
            iteration = iteration + 1
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        #your code
        for c in range(self.n_clusters):
            ids = np.where(clustering == c)
            self.centroids[c] = np.mean(X[ids[0]], axis = 0)

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            # your code
            n, m = X.shape
            ids = random.sample(range(n), self.n_clusters)
            self.centroids = X[ids]
        elif self.init == 'kmeans++':
            # your code
            # firstly, pick one point
            n = X.shape[0]
            all_ids = [i for i in range(n)]
            temp_ids = [-1] * self.n_clusters
            temp_ids[0] = random.sample(range(n), 1)[0]
            
            # remove
            all_ids.remove(temp_ids[0])
            
            for i in range(1,self.n_clusters):
                temp_len = len(all_ids)
                weights = [1000000] * temp_len
                # find the nearest centroid
                for j in range(temp_len):
                    # selected centroid
                    for tt in range(i):
                        # select minest
                        weights[j] = min(weights[j], np.linalg.norm(X[all_ids[j]] - temp_ids[tt]))
                
                # get k-th centroid
                temp_ids[i] = random.choices(all_ids, weights=weights, k=1)[0]
                # remove
                all_ids.remove(temp_ids[i])
            self.centroids = X[temp_ids]
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        n = X1.shape[0]
        t = X2.shape[0]
        dist = np.zeros([n, t])
        for i in range(n):
            for j in range(t):
                dist[i, j] = np.linalg.norm(X1[i] - X2[j])
        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code
        n = X.shape[0]
        s = np.zeros(n)
        for i in range(n):
            # cal ai
            label = clustering[i]
            ids = np.where(clustering == label)[0].tolist()
            ids.remove(i)
            num = len(ids)
            a_i = 0
            for j in ids:
                a_i = a_i + np.linalg.norm(X[i] - X[j])
            a_i = a_i / num
            
            # cal bi
            bs = []
            all_labels = [i for i in range(self.n_clusters)]
            all_labels.remove(label)
            for k in all_labels:
                temp_ids = np.where(clustering == k)[0].tolist()
                dis = 0
                for p in temp_ids:
                    dis = dis + np.linalg.norm(X[i] - X[p])
                dis = dis / len(temp_ids)
                bs.append(dis)
            b_i = min(bs)
            
            # cal si
            s[i] = (b_i - a_i) / max(a_i, b_i)
        return np.mean(s)