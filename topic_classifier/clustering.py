"""
Phase 3: Clustering and Cluster Validation
"""
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from topic_classifier.config import get_hdbscan_params


def cluster_embeddings(embeddings, method="hdbscan", k=None):
    """
    Cluster embeddings using HDBSCAN or KMeans.
    Returns labels and the fitted clusterer.
    """
    if method == "hdbscan":
        params = get_hdbscan_params()
        clusterer = HDBSCAN(**params)
        labels = clusterer.fit_predict(embeddings)
    elif method == "kmeans":
        if k is None:
            raise ValueError("k must be provided for kmeans clustering")
        clusterer = KMeans(n_clusters=k)
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return labels, clusterer


def evaluate_clusters(embeddings, labels):
    """
    Compute internal clustering metrics: silhouette, Davies-Bouldin, Calinski-Harabasz.
    """
    metrics = {}
    unique = set(labels)
    if len(unique) > 1:
        metrics["silhouette"] = silhouette_score(embeddings, labels)
        metrics["davies_bouldin"] = davies_bouldin_score(embeddings, labels)
        metrics["calinski_harabasz"] = calinski_harabasz_score(embeddings, labels)
    else:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["calinski_harabasz"] = None
    return metrics
