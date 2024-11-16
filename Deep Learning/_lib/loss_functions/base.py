import tensorflow as tf


class LossFn:
    
    ### Magic methods ###

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the loss.

        Parameters:
        - y_true (tf.Tensor): True target variable
        - y_pred (tf.Tensor): Predicted target variable

        Returns:
        - tf.Tensor: Loss value
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method '__call__' is not implemented.")


    ### Public methods ###

    def gradient(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradient of the loss with respect to y_pred.

        Parameters:
        - y_true (tf.Tensor): True target variable
        - y_pred (tf.Tensor): Predicted target variable

        Returns:
        - tf.Tensor: Gradient of the loss with respect to y_pred
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'gradient' is not implemented.")
