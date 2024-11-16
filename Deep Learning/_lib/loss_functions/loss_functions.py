import tensorflow as tf

from .base import LossFn

epsilon = 1e-12  # Small constant for numerical stability

class MeanSquareError(LossFn):
    
    ### Magic methods ###

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the mean squared error loss.
        
        Parameters:
        - y_true (tf.Tensor): True target variable
        - y_pred (tf.Tensor): Predicted target variable
        
        Returns:
        - float: Loss value
        """
        
        # Compute and return the loss
        return tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true))) 


    ### Public methods ###

    def gradient(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradient of the loss with respect to y_pred.
        
        Parameters:
        - y_true (tf.Tensor): True target variable
        - y_pred (tf.Tensor): Predicted target variable
        
        Returns:
        - tf.Tensor: Gradient of the loss with respect to y_pred
        """
        
        # Extract the batch size
        batch_size = y_true.shape[0]
        
        # Return the gradient
        return tf.multiply(tf.cast(tf.divide(2.0, batch_size), dtype=y_pred.dtype), tf.subtract(y_pred, y_true))
    
    
class CrossEntropy(LossFn):
        
    ### Magic methods ###

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the cross-entropy loss.
        
        Parameters:
        - y_true (tf.Tensor): True target variable
        - y_pred (tf.Tensor): Predicted target variable
        
        Returns:
        - float: Loss value
        """
        
        # Extract the batch size
        batch_size = y_true.shape[0]

        # Calculate the negative log likelihood
        return tf.negative(
            tf.reduce_sum(
                tf.divide(
                    tf.multiply(tf.cast(y_true, dtype=y_pred.dtype), tf.math.log(tf.clip_by_value(y_pred, epsilon, 1 - epsilon))),
                    batch_size
                )
            )
        )


    ### Public methods ###

    def gradient(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradient of the loss with respect to y_pred.
        
        Parameters:
        - y_true (tf.Tensor): True target variable
        - y_pred (tf.Tensor): Predicted target variable
        
        Returns:
        - tf.Tensor: Gradient of the loss with respect to y_pred
        """
        
        # Extract the batch size
        batch_size = y_true.shape[0]

        # Compute the gradient
        return tf.divide(tf.subtract(y_pred, tf.cast(y_true, dtype=y_pred.dtype)), batch_size)