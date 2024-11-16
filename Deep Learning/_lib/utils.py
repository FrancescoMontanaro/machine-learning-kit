import numpy as np
import tensorflow as tf
from typing import Optional
import matplotlib.pyplot as plt


def get_batch(tensor: tf.Tensor, batch_size: int, index: int) -> tf.Tensor:
    """
    Method to extract a batch from a tensor of any shape.
    The batch is extracted along the first axis.
    
    Parameters:
    - tensor (tf.Tensor): Input tensor
    - batch_size (int): Size of the batch
    - index (int): Index of the batch to extract
    """
    
    # Start index of the batch
    begin = [index * batch_size] + [0] * (len(tensor.shape) - 1)
    
    # Size of the batch
    size = [batch_size] + [-1] * (len(tensor.shape) - 1)
    
    # Extract the batch
    return tf.slice(tensor, begin, size)


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Method to split the dataset into training and testing sets
    
    Parameters:
    - X (tf.Tensor): Features of the dataset
    - y (tf.Tensor): Target of the dataset
    - test_size (float): Proportion of the dataset to include in the test split
    - seed (int): Random seed for reproducibility
    
    Returns:
    - tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and testing sets
    """
    
    # Calculate the number of samples in the test set
    num_samples = X.shape[0]
    test_size = int(num_samples * test_size)
    
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)  # Use default_rng for reproducibility with a seed
    rng.shuffle(indices)
    
    # Split indices into train and test sets
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    # Use indices to gather training and testing sets
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Return the training and testing sets
    return X_train, X_test, y_train, y_test


def shuffle_data(X: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Method to shuffle the dataset
    
    Parameters:
    - X (tf.Tensor): Features of the dataset
    - y (tf.Tensor): Target of the dataset
    
    Returns:
    - tuple[tf.Tensor, tf.Tensor]: Shuffled features and target
    """
    
    # Get the number of samples
    n_samples = X.shape[0]
    
    # Generate random indices
    indices = tf.random.shuffle(tf.range(n_samples))
    
    # Shuffle the dataset
    X_shuffled = tf.gather(X, indices)
    y_shuffled = tf.gather(y, indices)
    
    # Return the shuffled dataset
    return X_shuffled, y_shuffled


def plot_data(datasets: list[dict], title: str, xlabel: str, ylabel: str) -> None:
    """
    Method to plot multiple sets of samples in a 2D space, with options for scatter and line plots.
    
    Parameters:
    - datasets (list[dict]): List of dictionaries, each containing:
        - 'points' (np.ndarray): Dataset to plot with shape (n_samples, 2)
        - 'label' (str): Legend label for the dataset
        - 'color' (str): Color for the dataset plot
        - 'size' (int or float, optional): Size of the points for scatter plot (ignored for line plot)
        - 'plot_type' (str): Type of plot, either 'scatter' or 'line'
    - title (str): Title of the plot
    - xlabel (str): Label of the x-axis
    - ylabel (str): Label of the y-axis
    """
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Plot each dataset with its respective color, legend, and style
    for data in datasets:
        points = np.squeeze(data['points'])  # Remove any singleton dimensions
        if points.ndim == 1:  # Reshape to (n_samples, 2) if necessary
            points = points.reshape(-1, 2)
        
        if data['plot_type'] == 'scatter':
            plt.scatter(points[:, 0], points[:, 1], s=data.get('size', 10), c=data['color'], label=data['label'])
        elif data['plot_type'] == 'line':
            # Sort points by the x-axis to ensure a smooth line
            sorted_points = points[points[:, 0].argsort()]
            plt.plot(sorted_points[:, 0], sorted_points[:, 1], color=data['color'], label=data['label'], linewidth=2)
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add the legend
    plt.legend()
    
    # Show the plot
    plt.show()
    
    
def plot_history(train_loss: np.ndarray, valid_loss: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    """
    Method to plot the training and validation loss
    
    Parameters:
    - train_loss (np.ndarray): Training losses
    - valid_loss (np.ndarray): Validation losses
    - title (str): Title of the plot
    - xlabel (str): Label of the x-axis
    - ylabel (str): Label of the y-axis
    """
    
    # Create a new figure
    plt.figure(figsize=(8, 4))
    
    # Plot the training and validation loss
    plt.plot(train_loss, label="Training loss", color="blue")
    plt.plot(valid_loss, label="Validation loss", color="orange")
    
    # Set the title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add the legend
    plt.legend()
    
    # Show the plot
    plt.show()
 
 
def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> None:
    """
    Method to plot the confusion matrix
    
    Parameters:
    - cm (np.ndarray): Confusion matrix
    - title (str): Title of the plot
    """
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the confusion matrix
    ax.matshow(cm, cmap=plt.colormaps['Blues'], alpha=0.3)
    
    # Add the class labels
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=str(int(cm[i, j])), va='center', ha='center')
    
    # Showing the confusion matrix
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('Actuals', fontsize=12)
    plt.title(title, fontsize=12)
    plt.show()
   
    
def one_hot_encoding(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Method to perform one-hot encoding on the target variable
    
    Parameters:
    - y (np.ndarray): Target variable
    - n_classes (int): Number of classes
    
    Returns:
    - np.ndarray: One-hot encoded target variable
    """
    
    # Initialize the one-hot encoded target variable
    one_hot = np.zeros((y.shape[0], n_classes))
    
    # Set the appropriate index to 1
    one_hot[np.arange(y.shape[0]), y] = 1
    
    # Return the one-hot encoded target variable
    return one_hot.astype(int)