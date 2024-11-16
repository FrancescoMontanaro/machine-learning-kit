import tensorflow as tf
from typing import Optional

from .base import Layer


class Flatten(Layer):
    
    ### Magic methods ###
    
    def __init__(self, name: Optional[str] = None) -> None:
        """
        Class constructor 
        
        Parameters:
        - name (str): Name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Initialize the input shape
        self.input_shape = None
        
        
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the Flatten layer
        
        Parameters:
        - x (tf.Tensor): Input data. Shape: (Batch size, ...)
        
        Returns:
        - tf.Tensor: Flattened input data
        """
        
        # Store the input shape
        self.input_shape = x.shape
        
        # Initialize the layer params
        if not self.initialized:
            self.init_params(x)
        
        # Perform the forward pass
        return self.forward(x)
        
        
    ### Public methods ###
    
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass through the Flatten layer
        
        Parameters:
        - x (tf.Tensor): Input data. Shape: (Batch size, ...)
        
        Returns:
        - tf.Tensor: Flattened input data
        """
    
        # Extract the output dimensions
        output_shape = self.output_shape() # (batch_size, num_features)
        
        # Flatten the input data
        return tf.reshape(x, output_shape)
    
    
    def backward(self, grad: tf.Tensor) -> tf.Tensor:
        """
        Backward pass through the Flatten layer
        
        Parameters:
        - grad (tf.Tensor): Gradient of the loss with respect to the output of the Flatten layer
        
        Returns:
        - tf.Tensor: Gradient of the loss with respect to the input of the Flatten layer
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Assert that the input shape is set
        assert isinstance(self.input_shape, tf.TensorShape), "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Reshape the gradient
        return tf.reshape(grad, self.input_shape)
    
    
    def output_shape(self) -> tf.TensorShape:
        """
        Function to compute the output shape of the layer.
        
        Returns:
        - tuple: shape of the output data
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Assert that the input shape is set
        assert isinstance(self.input_shape, tf.TensorShape), "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Extract the dimensions
        batch_size = self.input_shape[0]
        num_features = tf.reduce_prod(self.input_shape[1:])
        
        # Compute the output shape
        return tf.TensorShape((batch_size, num_features))
    
    
    def init_params(self, x: tf.Tensor) -> None:
        """
        Method to initialize the parameters of the Flatten layer
        
        Parameters:
        - x (tf.Tensor): Input data
        """
        
        # Set the initialization flag
        self.initialized = True