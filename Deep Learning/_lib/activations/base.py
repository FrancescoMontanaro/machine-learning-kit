import tensorflow as tf


class Activation:
    
    ### Magic methods ###

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the output of the activation function.

        Parameters:
        - x (tf.Tensor): Input to the activation function

        Returns:
        - tf.Tensor: Output of the activation function
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method '__call__' is not implemented.")


    ### Public methods ###

    def derivative(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the activation function.

        Parameters:
        - x (tf.Tensor): Input to the activation function

        Returns:
        - tf.Tensor: Derivative of the activation function
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'derivative' is not implemented.")

