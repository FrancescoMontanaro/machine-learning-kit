import tensorflow as tf
from typing import Optional

from .base import Layer


class Dropout(Layer):
    
    ### Magic methods ###
    
    def __init__(self, rate: float, name: Optional[str] = None) -> None:
        """
        Initialize the dropout layer.
        
        Parameters:
        - rate (float): The dropout rate.
        - name (str): The name of the layer.
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Initialize the dropout rate
        self.rate = rate
        self.mask = None
        
    
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Method to set the input shape of the layer.
        
        Parameters:
        - x (tf.Tensor): Input data.
        
        Returns:
        - tf.Tensor: Output of the layer after the forward pass.
        """
        
        # Initialize the layer params
        if not self.initialized:
            self.init_params(x)
        
        # Forward pass
        return self.forward(x)
    
    
    ### Public methods ###
    
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the dropout layer.
        
        Parameters:
        - x (tf.Tensor): The input tensor.
        
        Returns:
        - tf.Tensor: The output tensor.
        """
        
        # Generate the mask
        self.mask = tf.where(tf.greater(tf.random.uniform(x.shape), self.rate), tf.ones_like(x), tf.zeros_like(x))
        
        if self.training:
            # Scale the output during training
            return tf.divide(tf.multiply(x, self.mask), tf.cast(tf.subtract(1.0, self.rate), dtype=x.dtype))

        # Return the output during inference
        return x
    
    
    def backward(self, grad_output: tf.Tensor) -> tf.Tensor:
        """
        Backward pass of the dropout layer (layer i)
        
        Parameters:
        - grad_output (tf.Tensor): The gradient of the loss with respect to the output of the layer: dL/dO_i
        
        Returns:
        - tf.Tensor: The gradient of the loss with respect to the input of the layer: dL/dX_i ≡ dL/dO_{i-1}
        """
        
        # Scale the gradient by the mask
        return tf.divide(tf.multiply(grad_output, self.mask), tf.cast(tf.subtract(1.0, self.rate), dtype=grad_output.dtype))  # dL/dX_i ≡ dL/dO_{i-1}, to pass to the previous layer
    
    
    def output_shape(self) -> tf.TensorShape:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tf.TensorShape: Shape of the output data
        """
        
        return self.input_shape
    
    
    def init_params(self, x: tf.Tensor) -> None:
        """
        Method to initialize the parameters of the layer
        
        Parameters:
        - x (tf.Tensor): Input data
        """
        
        # Save the input shape
        self.input_shape = x.shape
        
        # The layer is initialized
        self.initialized = True