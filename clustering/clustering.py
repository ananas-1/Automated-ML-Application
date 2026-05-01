import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import joblib
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocessing import run_preprocessing 


 
def run_clustering(df, n_clusters=3, save_path=None):
    prep_result = run_preprocessing(df, task="clustering")
 
    X         = prep_result["X_train"]    
    pipeline  = prep_result["pipeline"]      
 
    # Model 1: KMeans
    kmeans        = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels_kmeans = kmeans.fit_predict(X)
    score_kmeans  = silhouette_score(X, labels_kmeans)
 
    # Model 2: Agglomerative Clustering
    agg        = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agg = agg.fit_predict(X)
    score_agg  = silhouette_score(X, labels_agg)
 
    if score_kmeans >= score_agg:
        best_model  = kmeans
        best_labels = labels_kmeans
        best_score  = score_kmeans
        best_name   = "KMeans"
    else:
        best_model  = agg
        best_labels = labels_agg
        best_score  = score_agg
        best_name   = "Agglomerative Clustering"

    unique, counts = np.unique(best_labels, return_counts=True)
    cluster_sizes  = {int(u): int(c) for u, c in zip(unique, counts)}
 
    if save_path:
        bundle = {
            "model":      best_model,
            "pipeline":   pipeline,    
        }
        joblib.dump(bundle, save_path)
        print(f"Model saved → {save_path}")
 

    return {
        "best_name":     best_name,
        "best_score":    round(float(best_score),   4),
        "kmeans_score":  round(float(score_kmeans), 4),
        "agg_score":     round(float(score_agg),    4),
        "n_clusters":    n_clusters,
        "cluster_sizes": cluster_sizes,
        "labels":        best_labels.tolist(),
        "best_model":    best_model,     # the Backend uses this for /download_model
        "pipeline":      pipeline,
    }
 
