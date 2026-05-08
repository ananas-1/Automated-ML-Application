import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import joblib
from preprocessing.preprocessing import run_preprocessing 

def get_cluster_labels(df, labels):

    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    
    cluster_summaries = {}
    numeric_cols = df_clustered.select_dtypes(include=[np.number]).columns.drop('Cluster', errors='ignore')
    
    if len(numeric_cols) > 0:
        global_means = df_clustered[numeric_cols].mean()
        global_std = df_clustered[numeric_cols].std().replace(0, 1e-9)
    
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        if cluster_id == -1:
            cluster_summaries[-1] = "Noise / Outliers"
            continue
            
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        
        if len(numeric_cols) > 0:
            cluster_means = cluster_data[numeric_cols].mean()

            z_scores = (cluster_means - global_means) / global_std
            
            top_features = z_scores.abs().sort_values(ascending=False).head(3).index.tolist()
            
            desc = []
            for feat in top_features:
                val = z_scores[feat]
                if val > 0.5:
                    desc.append(f"High {feat}")
                elif val < -0.5:
                    desc.append(f"Low {feat}")
                else:
                    desc.append(f"Average {feat}")
            
            label = ", ".join(desc)
            if not label:
                label = f"Cluster {cluster_id}"
        else:

            cat_cols = df_clustered.select_dtypes(exclude=[np.number]).columns.drop('Cluster', errors='ignore')
            if len(cat_cols) > 0:
                desc = []
                for col in cat_cols[:3]:
                    mode_val = cluster_data[col].mode().iloc[0] if not cluster_data[col].mode().empty else "Unknown"
                    desc.append(f"{col}: {mode_val}")
                label = ", ".join(desc)
            else:
                label = f"Cluster {cluster_id}"
            
        cluster_summaries[int(cluster_id)] = label
        
    return cluster_summaries

def find_optimal_k(X, max_k=10):

    best_k = 2
    best_score = -1
    max_k = min(max_k, X.shape[0] - 1)
    
    if max_k < 2:
        return 2
        
    sample_size = min(10000, X.shape[0]) if X.shape[0] > 10000 else None

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
            if score > best_score:
                best_score = score
                best_k = k
    return best_k

def find_optimal_eps(X, min_samples=5):
 
    from sklearn.neighbors import NearestNeighbors
    
    X_arr = X.toarray() if hasattr(X, "toarray") else np.array(X)
    
    if X_arr.shape[0] > 5000:
        indices = np.random.choice(X_arr.shape[0], 5000, replace=False)
        X_sample = X_arr[indices]
    else:
        X_sample = X_arr
        
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(X_sample)
    distances, _ = neigh.kneighbors(X_sample)

    k_distances = np.sort(distances[:, min_samples - 1])
    
    recommended_eps = np.percentile(k_distances, 90)
    
    return max(0.1, float(recommended_eps))


def run_clustering(df, n_clusters="auto", save_path=None):
    prep_result = run_preprocessing(df, task="clustering")
 
    X         = prep_result["X_train"]    
    pipeline  = prep_result["pipeline"]      
 
    if n_clusters == "auto":
        n_clusters = find_optimal_k(X, max_k=10)

    sample_size = min(10000, X.shape[0]) if X.shape[0] > 10000 else None

    # Model 1: KMeans
    kmeans        = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels_kmeans = kmeans.fit_predict(X)
    if len(np.unique(labels_kmeans)) > 1:
        score_kmeans  = silhouette_score(X, labels_kmeans, sample_size=sample_size, random_state=42)
    else:
        score_kmeans = -1
 
    # Model 2: DBSCAN
    optimal_eps   = find_optimal_eps(X)
    dbscan        = DBSCAN(eps=optimal_eps, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X)
    
    unique_labels = set(labels_dbscan)
    if -1 in unique_labels: unique_labels.remove(-1)
    
    if len(unique_labels) > 1:
        score_dbscan = silhouette_score(X, labels_dbscan, sample_size=sample_size, random_state=42)
    else:
        score_dbscan = -1
 
    if score_kmeans >= score_dbscan:
        best_model  = kmeans
        best_labels = labels_kmeans
        best_score  = score_kmeans
        best_name   = "KMeans"
    else:
        best_model  = dbscan
        best_labels = labels_dbscan
        best_score  = score_dbscan
        best_name   = "DBSCAN"

    unique, counts = np.unique(best_labels, return_counts=True)
    cluster_sizes  = {int(u): int(c) for u, c in zip(unique, counts)}
 
    # labels
    cluster_labels = get_cluster_labels(df, best_labels)

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
        "dbscan_score":  round(float(score_dbscan), 4),
        "n_clusters":    n_clusters if best_name == "KMeans" else len(unique_labels),
        "cluster_sizes": cluster_sizes,
        "cluster_labels": cluster_labels,
        "labels":        best_labels.tolist(),
        "best_model":    best_model,     
        "pipeline":      pipeline,
    }
