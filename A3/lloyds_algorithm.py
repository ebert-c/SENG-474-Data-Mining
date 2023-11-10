import numpy as np
import random
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as scp
import matplotlib.pyplot as plt


class Cluster:

    def __init__(self, center):
        self.center = center
        self.cluster = []

    def __cmp__(self, other):
        return (self.cluster == other.cluster).all()

    def __repr__(self):
        return f"{self.center}"

    def calc_new_center(self):
        if len(self.cluster) == 0:
            return
        self.center = np.divide(np.sum(self.cluster, axis=0), len(self.cluster))

    def calc_cost(self):
        cost = 0
        for i in self.cluster:
            cost += np.sum((self.center - i) ** 2)
        return cost


def main():
    dataset1 = np.loadtxt("dataset1.csv", delimiter=",")
    dataset2 = np.loadtxt("dataset2.csv", delimiter=",")
    datasets = [dataset1, dataset2]

    # test_lloyds(datasets, uniform_init, "uniform_lloyds")
    # test_lloyds(datasets, k_means_plus_plus_init, "k_means_lloyds")
    test_hierarchy(datasets, 'single', 'hierarchy_single')
    # test_hierarchy(datasets, 'average', 'hierarchy_average')


def test_hierarchy(datasets, linkage, name):
    for i in range(len(datasets)):
        dataset = datasets[i]
        Z = scp.linkage(dataset, method=linkage, metric='euclidean')
        plt.figure(i)
        dn = scp.dendrogram(Z, truncate_mode='lastp')
        plt.savefig(f"{name}{i}.svg", format='svg')



def test_lloyds(datasets, init_alg, name):
    for i in range(len(datasets)):
        dataset = datasets[i]
        costs = []
        ks = []
        for k in range(2, 20):
            sub_cost = []
            for j in range(3):
                cost, increment = lloyd(init_alg, k, dataset)
                sub_cost.append(cost)
            costs.append(min(sub_cost))
            ks.append(k)
        plt.figure(i)
        plt.plot(ks, costs)
        plt.xticks(ks)
        plt.savefig(f'{name}{i}.svg', format='svg')


def uniform_init(k, data):
    indexes = [i for i in range(len(data))]
    chosen = random.sample(indexes, k)
    return [data[c] for c in chosen]


def k_means_plus_plus_init(k, data):
    centers = uniform_init(1, data)
    for i in range(k - 1):
        distances = []
        for d in data:
            center_dists = []
            for c in centers:
                center_dists.append(np.sqrt(np.sum((d - c) ** 2)))
            distances.append(min(center_dists))
        total = sum(distances)
        normalized_distance = [x / total for x in distances]
        center = random.choices(data, weights=normalized_distance)
        centers.append(center)
    return centers


def add_to_cluster(p1, clusters):
    distances = []
    for c in clusters:
        distances.append(np.sqrt(np.sum((p1 - c.center) ** 2)))
    min_cluster = clusters[distances.index(min(distances))]
    min_cluster.cluster.append(p1)


def update_centers(clusters):
    for c in clusters:
        c.calc_new_center()


def lloyd(init_alg, k, data):
    centers = init_alg(k, data)
    clusters = [Cluster(c) for c in centers]
    prev_clusters = [Cluster(c) for c in centers]
    MAX = 2000
    increment = 1
    while increment < MAX:
        for d in data:
            add_to_cluster(d, clusters)
        for c in clusters:
            c.calc_new_center()
        for i in range(len(clusters)):
            if clusters[i] != prev_clusters[i]:
                break
        else:
            cost = 0
            for c in clusters:
                cost += c.calc_cost()
            return cost, increment

        prev_clusters = list(clusters)
        for c in clusters:
            c.cluster = []
        increment += 1

    print("MAX reached")


if __name__ == "__main__":
    main()
