"""
Programmed by William Samuelson
Implements clustering algorithms
"""
from sklearn.cluster import AffinityPropagation, MeanShift, DBSCAN, SpectralClustering, KMeans, AgglomerativeClustering, Birch
import numpy as np
from functools import partial

from GPFramework.data import EmadeDataPair
from GPFramework.wrapper_methods import RegistryWrapperCM

cmw = RegistryWrapperCM([EmadeDataPair])

def affinity_propagation_helper(damping=0.5):
    damping = 0.5 if damping < 0.5 else (.99999 if damping >= .99999 else damping)
    return AffinityPropagation(damping=damping, copy=False)

affinity_propagation = cmw.register("AffinityPropagationClustering", "test_affinity_propagation", affinity_propagation_helper, None, [float])
affinity_propagation.__doc__ = """
Performs affinity propagation clustering of data

Args:
    damping: Extent to which the current value is maintained relative to incoming values

Returns:
    Clustered data
"""

mean_shift = cmw.register("MeanShiftClustering", "test_mean_shift", MeanShift, None, [])
mean_shift.__doc__ = """
Performs mean shift clustering using a flat kernel

Returns:
    Clustered data
"""

def db_scan_helper(eps=0.5, p=2):
    eps = np.nextafter(0, 1) if eps <= 0 else eps
    p = 0 if p < 0 else p
    return DBSCAN(eps=eps, p=p)

db_scan = cmw.register("DBSCANClustering", "test_db_scan", db_scan_helper, None, [float, float])
db_scan.__doc__ = """
Performs DBSCAN clustering

Args:
    eps: Maximum distance between two samples for them to be considered as in the same neighborhood
    p: Power of the Minkowski metric to be used to calculate distance between points

Returns:
    Clustered data
"""

def spectral_cluster_helper(n_clusters=8, gamma=1.0):
    n_clusters = 1 if n_clusters < 1 else n_clusters
    n_clusters = int(n_clusters)
    gamma = np.nextafter(0, 1) if gamma <= 0 else gamma
    return SpectralClustering(n_clusters=n_clusters, gamma=gamma)

spectral_cluster = cmw.register("SpectralClustering", "test_spectral_cluster", spectral_cluster_helper, None, [int, float])
spectral_cluster.__doc__ = """
Performs spectral clustering

Args:
    n_clusters      Dimension of the projection subspace
    gamma           Kernel coefficient for rbf kernel

Returns:
    Clustered data
"""

def k_means_cluster_helper(n_clusters=8):
    n_clusters = 1 if n_clusters < 1 else n_clusters
    n_clusters = int(n_clusters)
    return KMeans(n_clusters=n_clusters)

k_means_cluster = cmw.register("k_means_clustering", "test_k_means_cluster", k_means_cluster_helper, None, [int])
k_means_cluster.__doc__ = """
Performs K-means clustering

Args:
    n_clusters      Number of clusters to form as well as the number of centroids to generate

Returns:
    Clustered data
"""

def agglomerative_cluster_helper(n_clusters=2):
    n_clusters = 1 if n_clusters < 1 else n_clusters
    n_clusters = int(n_clusters)
    return AgglomerativeClustering(n_clusters=n_clusters)

agglomerative_cluster = cmw.register("agglomerative_clustering", "test_agglomerative_cluster", agglomerative_cluster_helper, None, [int])
agglomerative_cluster.__doc__ = """
Performs agglomerative clustering

Args:
    n_clusters      Number of clusters to find

Returns:
    Clustered data
"""

def birch_cluster_helper(threshold=0.5, branching_factor=50, n_clusters=3):
    threshold = np.nextafter(0, 1) if threshold <= 0 else threshold
    branching_factor = 1 if branching_factor < 1 else branching_factor
    branching_factor = int(branching_factor)
    n_clusters = 1 if n_clusters < 1 else n_clusters
    n_clusters = int(n_clusters)
    return Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters)

birch_cluster = cmw.register("birch_clustering", "test_birch_cluster", birch_cluster_helper, None, [float, int, int])
birch_cluster.__doc__ = """
Performs birch clustering

Args:
    threshold           Maximum radius of the subcluster obtained by merging a new sample and the closest subcluster
    branching_factor    Maximum number of CF subclusters in each node
    n_clusters          Number of clusters after the final clustering step

Returns:
    Clustered data
"""
