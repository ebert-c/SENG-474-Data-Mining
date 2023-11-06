import numpy as np
import random


class Cluster:

    def __init__(self, center):
        self.center = center
        self.cluster = np.array([])

    def __cmp__(self, other):
        return (self.cluster == other.cluster).all()

    def calc_new_center(self):
        self.center = np.sum(self.cluster) / len(self.cluster)


def uniform_init(k, data):
    return random.sample(data, k)


def k_means_plus_plus_init(k, data):
    centers = random.sample(data, 1)
    for i in range(k - 1):
        distances = []
        for d in data:
            center_dists = []
            for c in centers:
                center_dists.append(np.sqrt(np.sum((d - c) ** 2)))
            distances.append(min(center_dists))
        centers.append(max(distances))


def add_to_cluster(p1, clusters):
    distances = []
    for c in clusters:
        distances.append(np.sqrt(np.sum((p1 - c.center) ** 2)))
    clusters[distances.index(min(distances))].cluster.append(p1)


def update_centers(clusters):
    for c in clusters:
        c.calc_new_center()


def lloyd(init_alg, k, data):
    centers = init_alg(k, data)
    clusters = [Cluster(c) for c in centers]
    prev_clusters = [Cluster(c) for c in centers]
    MAX = 1000
    increment = 1
    while increment < MAX:
        for d in data:
            add_to_cluster(d, clusters)
        update_centers(clusters)
        for i in range(len(clusters)):
            if clusters[i] != prev_clusters:
                break
        else:
            return clusters, increment

        prev_clusters = list(clusters)
        increment += 1
