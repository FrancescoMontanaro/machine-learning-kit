import numpy as np
import tensorflow as tf


def mean_absolute_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """
    Compute the mean absolute error of the model.

    Parameters:
    - y_true (tf.Tensor): True target variable
    - y_pred (tf.Tensor): Predicted target variable

    Returns:
    - float: Mean absolute error of the model
    """
    
    # Compute the mean absolute error
    return float(tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred))))


def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """
    Compute the accuracy of the model.

    Parameters:
    - y_true (tf.Tensor): True target variable
    - y_pred (tf.Tensor): Predicted target variable

    Returns:
    - float: Accuracy of the model
    """
    
    # If the rank of the input tensors is greater than 1, take the argmax
    if len(y_true.shape) > 1:
        y_true = tf.argmax(y_true, axis=-1)
        
    if len(y_pred.shape) > 1:
        y_pred = tf.argmax(y_pred, axis=-1)
    
    # Compute the accuracy
    return float(tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32), axis=-1))
    

def precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """
    Compute the precision of the model.

    Parameters:
    - y_true (tf.Tensor): True target variable
    - y_pred (tf.Tensor): Predicted target variable

    Returns:
    - float: Precision of the model
    """
    
    # Compute the true positives and false positives
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), tf.float32))
    
    # Return the precision
    return float(tf.divide(tp, tf.add(tp, fp)))


def recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """
    Compute the recall of the model.

    Parameters:
    - y_true (tf.Tensor): True target variable
    - y_pred (tf.Tensor): Predicted target variable

    Returns:
    - float: Recall of the model
    """
    
    # Compute the true positive and false negative
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0)), tf.float32))
    
    # Return the recall
    return float(tf.divide(tp, tf.add(tp, fn)))


def f1_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """
    Compute the F1 score of the model.

    Parameters:
    - y_true (tf.Tensor): True target variable
    - y_pred (tf.Tensor): Predicted target variable

    Returns:
    - float: F1 score of the model
    """
    
    # Compute the precision and recall
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    # Compute the F1 score
    return 2 * (prec * rec) / (prec + rec)


def confusion_matrix(num_classes: int, y_true: tf.Tensor, y_pred: tf.Tensor) -> np.ndarray:
    """
    Compute the confusion matrix of the model.

    Parameters:
    - num_classes (int): Number of classes
    - y_true (tf.Tensor): True target variable
    - y_pred (tf.Tensor): Predicted target variable

    Returns:
    - np.ndarray: Confusion matrix of the model
    """
    
    # If the rank of the input tensors is greater than 1, take the argmax
    if len(y_true.shape) > 1:
        y_true = tf.argmax(y_true, axis=-1)
        
    if len(y_pred.shape) > 1:
        y_pred = tf.argmax(y_pred, axis=-1)
    
    # Compute the confusion matrix
    return tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)