
import pandas as pd
import os
from clustering import run_clustering

# ── Load Dataset ───────────────────────────────────────────────
print("=" * 55)
print("Clustering Test")
print("=" * 55)

csv_path = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
df = pd.read_csv(csv_path)

print(f"\n Dataset : Mall_Customers.csv")
print(f" Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
print(f" Columns : {df.columns.tolist()}")
print(f"\n First 5 rows:")
print(df.head())

# ── Run Clustering ─────────────────────────────────────────────
print("\n" + "-" * 55)
print(" Running Clustering Pipeline...")
print("-" * 55)

results = run_clustering(
    df         = df,                         
)

# ── Print Results ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("              RESULTS")
print("=" * 55)

print(f"\n  Model Comparison:")
print(f"    KMeans Silhouette Score            : {results['kmeans_score']}")
print(f"    Agglomerative Silhouette Score     : {results['agg_score']}")

print(f"\n  Best Model  : {results['best_name']}")
print(f"  Best Score  : {results['best_score']}  (Silhouette Score)")

print(f"\n  Number of Clusters (K) : {results['n_clusters']}")

print(f"\n  Cluster Distribution:")
for cluster_id, count in results["cluster_sizes"].items():
    print(f"    Cluster {cluster_id} : {count} data points")


print("\n" + "=" * 55)