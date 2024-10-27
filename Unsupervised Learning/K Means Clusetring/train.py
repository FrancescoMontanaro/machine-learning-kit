import numpy as np
from k_means import KMeans
from sklearn.datasets import make_blobs


if __name__ == '__main__':
    # Setting the random seed
    np.random.seed(42)

    # Creating the dataset
    X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40) # type: ignore

    # Extracting the number of clusters
    clusters = len(np.unique(y))

    # Initializing the KMeans class
    kmeans = KMeans(k=clusters, max_iterations=150, plot_steps=True)

    # Extracting the clusters
    y_pred = kmeans.predict(X)

    # Plotting the clusters
    kmeans.plot()
