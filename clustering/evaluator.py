"""
Clustering Evaluator - Calculate metrics
"""

from sklearn.metrics import silhouette_score


def evaluate_clustering_model(model, X):
    """
    Evaluate clustering model
    
    Args:
        model: Trained model
        X: Features
        
    Returns:
        Dictionary with all metrics
    """
    labels = model.labels_
    
    metrics = {
        'silhouette_score': float(silhouette_score(X, labels)),
        'n_clusters': len(set(labels))
    }
    
    return metrics
