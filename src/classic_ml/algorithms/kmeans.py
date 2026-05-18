# src/classic_ml/algorithms/kmeans.py

import random
import math

class KMeans:
    def __init__(self, k=3, max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = []
        self.clusters = []
        self.iterations = 0
        self.converged = False

    @staticmethod
    def euclidean(a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def initialize_centroids(self, X):
        if len(X) < self.k:
            self.centroids = X[:]
        else:
            self.centroids = random.sample(X, self.k)

    def fit(self, X):
        if not X:
            return
        
        self.initialize_centroids(X)
        
        for i in range(self.max_iter):
            self.iterations = i + 1
            clusters = [[] for _ in self.centroids]
            for x in X:
                distances = [self.euclidean(x, c) for c in self.centroids]
                idx = distances.index(min(distances))
                clusters[idx].append(x)
            
            self.clusters = clusters
            
            new_centroids = []
            for cluster in clusters:
                if not cluster:
                    new_centroids.append(random.choice(X))
                    continue
                m = len(cluster[0])
                mean = [sum(point[i] for point in cluster) / len(cluster) for i in range(m)]
                new_centroids.append(mean)
            
            shifts = [self.euclidean(c, nc) for c, nc in zip(self.centroids, new_centroids)]
            self.centroids = new_centroids
            
            if max(shifts) < self.tol:
                self.converged = True
                break
        
        return self.centroids, self.clusters

    def predict(self, x):
        if not self.centroids:
            return None
        distances = [self.euclidean(x, c) for c in self.centroids]
        return distances.index(min(distances))
