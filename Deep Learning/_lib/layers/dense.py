import tensorflow as tf
from typing import Optional

from .base import Layer
from ..optimizers import Optimizer
from ..activations import Activation
 
 
class Dense(Layer):
    
    ### Magic methods ###
    
    def __init__(self, num_units: int, activation: Optional[Activation] = None, name: Optional[str] = None) -> None:
        """
        Class constructor
        
        Parameters:
        - num_units (int): Number of units in the layer
        - activation (Callable): Activation function of the layer. Default is ReLU
        - name (str): Name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Store the activation function
        self.activation = activation
        self.num_units = num_units
        
        # Initialize the weights and bias
        self.weights = None
        self.bias = None
        
        
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Method to set the input shape of the layer and initialize the weights and bias
        
        Parameters:
        - x (tf.Tensor): Input data. Shape: (Batch size, number of features)
        
        Returns:
        - tf.Tensor: Output of the layer after the forward pass
        
        Raises:
        - ValueError: If the input shape is not 2D
        - AssertionError: If the batch size and number of features are not integers
        """
        
        # Check if the input shape is valid
        if len(x.shape) != 2:
            raise ValueError(f"Invalid input shape. Input must be a 2D array. The shape must be (Batch size, number of features). Got shape: {len(x.shape)}")
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params(x)
            
        # Return the output of the layer
        return self.forward(x)
    
    
    ### Public methods ###
    
    def parameters(self) -> dict:
        """
        Method to return the parameters of the layer
        
        Returns:
        - dict: Dictionary containing the weights and bias of the layer
        """
        
        return {"weights": self.weights, "bias": self.bias}
    

    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Method to compute the output of the layer
        
        Parameters:
        - x (tf.Tensor): Features of the dataset
        
        Returns:
        - tf.Tensor: Output of the layer
        
        Raises:
        - AssertionError: If the weights and bias are not initialized
        """
        
        # Assert that the weights and bias are initialized
        assert isinstance(self.weights, tf.Tensor), "Weights are not initialized. Please call the layer with some input data to initialize the weights."
        assert isinstance(self.bias, tf.Tensor), "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        # Store the input for the backward pass
        self.input = x
        
        # Compute the linear combination of the weights and features
        self.linear_comb = tf.add(tf.matmul(x, self.weights), self.bias)
        
        # Return the output of the neuron
        return self.activation(self.linear_comb) if self.activation is not None else self.linear_comb
    
    
    def backward(self, loss_gradient: tf.Tensor) -> tf.Tensor:
        """
        Backward pass of the layer (layer i)
        
        Parameters:
        - loss_gradient (tf.Tensor): Gradient of the loss with respect to the output of the layer: dL/dO_i
        
        Returns:
        - tf.Tensor: Gradient of the loss with respect to the input of the layer: dL/dX_i ≡ dL/dO_{i-1}
        
        Raises:
        - AssertionError: If the weights and bias are not initialized
        """
        
        # Assert that the weights and bias are initialized
        assert isinstance(self.optimizer, Optimizer), "Optimizer is not set. Please set an optimizer before training the model."
        assert isinstance(self.weights, tf.Tensor), "Weights are not initialized. Please call the layer with some input data to initialize the weights."
        assert isinstance(self.bias, tf.Tensor), "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        # Apply the activation derivative if the activation function is set
        if self.activation is not None:
            # Multiply the incoming gradient by the activation derivative
            loss_gradient = tf.multiply(loss_gradient, tf.cast(self.activation.derivative(self.linear_comb), dtype=loss_gradient.dtype)) # dL/dO_i * dO_i/dX_i, where dO_i/dX_i is the activation derivative

        # Compute gradients with respect to input, weights, and biases
        grad_input = tf.matmul(loss_gradient, tf.transpose(self.weights)) # dL/dX_i ≡ dL/dO_{i-1}, the gradient with respect to the input
        grad_weights = tf.matmul(tf.transpose(self.input), loss_gradient) # dL/dW_ij, the gradient with respect to the weights
        grad_bias = tf.reduce_sum(loss_gradient, axis=0) # dL/db_i, the gradient with respect to the bias

        # Update weights
        self.weights = self.optimizer.update(
            layer = self,
            param_name = "weights",
            params = self.weights,
            grad_params = grad_weights
        )
        
        # Update bias
        self.bias = self.optimizer.update(
            layer = self,
            param_name = "bias",
            params = self.bias,
            grad_params = grad_bias
        )

        return grad_input # dL/dX_i ≡ dL/dO_{i-1}, to pass to the previous layer
    
    
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        
        Raises:
        - AssertionError: If the weights and bias are not initialized
        """
        
        # Assert that the weights and bias are initialized
        assert isinstance(self.weights, tf.Tensor), "Weights are not initialized. Please call the layer with some input data to initialize the weights."
        assert isinstance(self.bias, tf.Tensor), "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        # Return the number of parameters in the layer
        return int(tf.add(tf.size(self.weights), tf.size(self.bias)))
    
    
    def output_shape(self) -> tf.TensorShape:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tf.TensorShape: Shape of the output data
        """
        
        # Unpack the input shape for better readability
        batch_size, _ = self.input_shape
        
        # The output shape
        return tf.TensorShape((batch_size, self.num_units)) # (Batch size, number of units)
    
        
    def init_params(self, x: tf.Tensor) -> None:
        """
        Method to initialize the weights and bias of the layer
        
        Parameters:
        - x (tf.Tensor): Input data
        """
        
        # Save the input shape
        self.input_shape = x.shape
        
        # Unpack the input shape for better readability
        _, num_features = self.input_shape
        
        # Initialize the weights and bias with random values
        self.weights = tf.cast(tf.random.uniform((num_features, self.num_units), tf.negative(tf.sqrt(tf.divide(1.0, num_features))), tf.sqrt(tf.divide(1.0, num_features))), dtype=x.dtype)
        self.bias = tf.cast(tf.zeros(self.num_units), dtype=x.dtype)
        
        # Update the initialization flag
        self.initialized = True