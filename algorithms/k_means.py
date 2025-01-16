from sklearn.cluster import KMeans

def k_means(X, y=None):
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X)
    return {
        "status": "success",
        "message": f"K-Means Clustering completed. Cluster centers: {model.cluster_centers_.tolist()}"
    }
