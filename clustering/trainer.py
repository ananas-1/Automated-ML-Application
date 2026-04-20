"""
Clustering Trainer
"""

from sklearn.metrics import silhouette_score
from clustering.models import get_clustering_models


def train_clustering_models(X):
    """
    Train all clustering models and select the best
    
    Args:
        X: Features
        
    Returns:
        Tuple of (best_model, best_model_name)
    """
    models = get_clustering_models()
    best_model = None
    best_model_name = None
    best_score = -1
    
    for model_name, model in models.items():
        try:
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = model_name
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    return best_model, best_model_name
