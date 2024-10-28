from pca import PCA
from sklearn import datasets
import matplotlib.pyplot as plt

# Loading the dataset
data = datasets.load_iris()

# Extracting the features and the labels
x, y = data.data, data.target # type: ignore

# Creating the PCA instance
pca = PCA(n_components=2)

# Fitting the model
pca.fit(x)

# Transforming the dataset
x_transformed = pca.transform(x)

# Plotting the dataset
plt.scatter(x_transformed[:, 0], x_transformed[:, 1], c=y, cmap="viridis")
plt.show()