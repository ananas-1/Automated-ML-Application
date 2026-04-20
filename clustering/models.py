"""
Clustering Models
"""

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


def get_clustering_models():
    """
    Get available clustering models
    
    Returns:
        Dictionary with model name and model instance
    """
    models = {
        'K-Means': KMeans(n_clusters=3, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Hierarchical': AgglomerativeClustering(n_clusters=3)
    }
    return models
