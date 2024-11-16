import numpy as np
import tensorflow as tf
from typing import Optional

from .base import Layer
from ..optimizers import Optimizer


class BatchNormalization(Layer):
    
    ### Magic methods ###
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.1, name: Optional[str] = None) -> None:
        """
        Initialize the batch normalization layer.
        
        Parameters:
        - epsilon (float): The epsilon parameter for numerical stability.
        - momentum (float): The momentum parameter for the moving average.
        - name (str): The name of the layer.
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Initialize the epsilon parameter
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Initialize the scale and shift parameters
        self.gamma = None
        self.beta = None
        
        # Initialize the mean and variance parameters
        self.running_mean = None
        self.running_var = None
     
     
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Method to set the input shape of the layer and initialize the scale and shift parameters.
        
        Parameters:
        - x (tf.Tensor): Input data. The shape can be of any dimension:
            (batch_size, num_features) for 1D data
            (batch_size, sequence_length, num_features) for 2D data
            (batch_size, height, width, n_channels ≡ n_features) for 3D data
            (batch_size, time, height, width, n_channels ≡ n_features) for 4D data
            ...
        
        Returns:
        - tf.Tensor: Output of the layer after the forward pass
        """
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params(x)
        
        # Forward pass
        return self.forward(x)

     
    ### Public methods ###
    
    def parameters(self) -> dict:
        """
        Method to return the parameters of the layer
        
        Returns:
        - dict: The parameters of the layer
        """
        
        return {
            "gamma": self.gamma,
            "beta": self.beta
        }
    
    
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the batch normalization layer.
        
        Parameters:
        - x (tf.Tensor): Input data. The shape can be of any dimension:
            (batch_size, num_features) for 1D data
            (batch_size, sequence_length, num_features) for 2D data
            (batch_size, height, width, n_channels ≡ n_features) for 3D data
            (batch_size, time, height, width, n_channels ≡ n_features) for 4D data
            ...
        
        Returns:
        - tf.Tensor: The normalized tensor.
        
        Raises:
        - AssertionError: If the gamma and beta parameters are not initialized or the running mean and variance are not set.
        """
        
        # Assert that the gamma and beta parameters are initialized and the running mean and variance are set
        assert isinstance(self.gamma, tf.Tensor), "Gamma parameter is not initialized. Please call the layer with input data."
        assert isinstance(self.beta, tf.Tensor), "Beta parameter is not initialized. Please call the layer with input data."
        assert isinstance(self.running_mean, tf.Tensor), "Running mean is not initialized. Please call the layer with input data."
        assert isinstance(self.running_var, tf.Tensor), "Running variance is not initialized. Please call the layer with input data."
        
        # Determine axes to compute mean and variance: all except the last dimension (features)
        axes = tuple(range(len(x.shape) - 1))
        
        # The layer is in training phase
        if self.training:
            # Calculate batch mean and variance
            batch_mean = tf.reduce_mean(x, axis=axes, keepdims=True)
            batch_var = tf.math.reduce_variance(x, axis=axes, keepdims=True)

            # Update running mean and variance
            self.running_mean = tf.add(tf.multiply(self.momentum, self.running_mean), tf.multiply(1 - self.momentum, batch_mean))
            self.running_var = tf.add(tf.multiply(self.momentum, self.running_var), tf.multiply(1 - self.momentum, batch_var))

            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
           
        # The layer is in inference phase 
        else:
            # Use running statistics for normalization
            mean = self.running_mean
            var = self.running_var
            
        # Normalize the input
        self.X_centered = tf.subtract(x, mean)
        self.stddev_inv = tf.math.rsqrt(tf.add(var, self.epsilon))
        self.X_norm = tf.multiply(self.X_centered, self.stddev_inv)
            
        # Scale and shift
        return tf.add(tf.multiply(self.gamma, self.X_norm), self.beta)
    
    
    def backward(self, grad_output: tf.Tensor) -> tf.Tensor:
        """
        Backward pass of the batch normalization layer (layer i)
        
        Parameters:
        - grad_output (tf.Tensor): The gradient of the loss with respect to the output of the layer: dL/dO_i
        
        Returns:
        - tf.Tensor: The gradient of the loss with respect to the input of the layer: dL/dX_i ≡ dL/dO_{i-1}
        
        Raises:
        - AssertionError: If the gamma and beta parameters are not initialized or the optimizer is not set.
        """
        
        # Assert that the gamma and beta parameters are initialized and the optimizer is set
        assert isinstance(self.optimizer, Optimizer), "Optimizer is not set. Please set an optimizer before training the model."
        assert isinstance(self.gamma, tf.Tensor), "Gamma parameter is not initialized. Please call the layer with input data."
        assert isinstance(self.beta, tf.Tensor), "Beta parameter is not initialized. Please call the layer with input data."
        
        # Number of samples in the batch
        batch_size = grad_output.shape[0]
        
        # Determine axes: all except the last dimension (features)
        axes = tuple(range(len(grad_output.shape) - 1))
        
        # Gradients of beta and gamma
        d_beta = tf.reduce_sum(grad_output, axis=axes, keepdims=True) # dL/d_beta = dL/dO_i, the gradient with respect to the beta parameter
        d_gamma = tf.reduce_sum(tf.multiply(grad_output, self.X_norm), axis=axes, keepdims=True) # dL/d_gamma = dL/dO_i * X_norm, the gradient with respect to the gamma parameter

        # Gradient with respect to the normalized input
        dx_hat = tf.multiply(grad_output, self.gamma)
        
        # Gradient with respect to variance
        d_var = tf.multiply(tf.reduce_sum(tf.multiply(dx_hat, self.X_centered), axis=axes), -0.5, tf.pow(self.stddev_inv, 3))
        
        # Gradient with respect to mean
        d_mean = tf.add(tf.reduce_sum(tf.multiply(dx_hat, -self.stddev_inv), axis=axes), tf.multiply(d_var, tf.reduce_mean(tf.multiply(-2, self.X_centered), axis=axes)))

        # Gradient with respect to input x
        dx = tf.add( # dL/dX_i ≡ dL/dO_{i-1}
            tf.multiply(dx_hat, self.stddev_inv),
            tf.divide(
                tf.multiply(
                    self.X_centered,
                    tf.multiply(2, d_var),
                ),
                batch_size
            ),
            tf.divide(
                d_mean,
                batch_size
            )
        )
        
        # Update the gamma parameter
        self.gamma = self.optimizer.update(
            layer = self,
            param_name = "gamma",
            params = self.gamma,
            grad_params = d_gamma
        )
        
        # Update the beta parameter
        self.beta = self.optimizer.update(
            layer = self,
            param_name = "beta",
            params = self.beta,
            grad_params = d_beta
        )
        
        return dx # dL/dX_i ≡ dL/dO_{i-1}, to pass to the previous layer
    
    
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        
        Raises:
        - AssertionError: If the gamma and beta parameters are not initialized
        """
        
        # Assert that the gamma and beta parameters are initialized
        assert isinstance(self.gamma, tf.Tensor), "Gamma parameter is not initialized. Please call the layer with input data."
        assert isinstance(self.beta, tf.Tensor), "Beta parameter is not initialized. Please call the layer with input data."
        
        # Return the number of parameters in the layer
        return int(tf.add(tf.size(self.gamma), tf.size(self.beta)))
    
    
    def output_shape(self) -> tf.TensorShape:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tf.TensorShape: The shape of the output of the layer
        """
        
        return self.input_shape
    
    
    def init_params(self, x: tf.Tensor) -> None:
        """
        Method to initialize the parameters of the layer
        
        Parameters:
        - x (tf.Tensor): Input data
        """
        
        # Store the input dimension
        self.input_shape = x.shape
        
        # Extract the shape of the parameters
        num_features = self.input_shape[-1]
        
        # Initialize the scale and shift parameters
        self.gamma = tf.cast(tf.ones(num_features), dtype=x.dtype)
        self.beta = tf.cast(tf.zeros(num_features), dtype=x.dtype)
        
        # Initialize the running mean and variance
        self.running_mean = tf.cast(tf.zeros(num_features), dtype=x.dtype)
        self.running_var = tf.cast(tf.ones(num_features), dtype=x.dtype)
        
        # Update the initialization flag
        self.initialized = True
 

class LayerNormalization(Layer):
    
    ### Magic methods ###
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.1, name: Optional[str] = None) -> None:
        """
        Initialize the layer normalization layer.
        
        Parameters:
        - epsilon (float): The epsilon parameter for numerical stability.
        - momentum (float): The momentum parameter for the moving average.
        - name (str): The name of the layer.
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Initialize the epsilon parameter
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Initialize the scale and shift parameters
        self.gamma = None
        self.beta = None
     
     
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Method to set the input shape of the layer and initialize the scale and shift parameters
        
        Parameters:
        - x (tf.Tensor): Input data. The shape can be of any dimension:
            (batch_size, num_features) for 1D data
            (batch_size, sequence_length, num_features) for 2D data
            (batch_size, height, width, n_channels ≡ n_features) for 3D data
            (batch_size, time, height, width, n_channels ≡ n_features) for 4D data
            ...
        
        Returns:
        - tf.Tensor: Output of the layer after the forward pass
        
        Raises:
        - ValueError: If the input shape is not 2D
        """
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params(x)
        
        # Forward pass
        return self.forward(x)
     
     
    ### Public methods ###
    
    def parameters(self) -> dict:
        """
        Method to return the parameters of the layer
        
        Returns:
        - dict: The parameters of the layer
        """
        
        return {
            "gamma": self.gamma,
            "beta": self.beta
        }
    
    
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass of the layer normalization layer.
        
        Parameters:
        - x (tf.Tensor): Input data. The shape can be of any dimension:
            (batch_size, num_features) for 1D data
            (batch_size, sequence_length, num_features) for 2D data
            (batch_size, height, width, n_channels ≡ n_features) for 3D data
            (batch_size, time, height, width, n_channels ≡ n_features) for 4D data
            ...
        
        
        Returns:
        - tf.Tensor: The normalized tensor.
        """
        
        # Extract the axis along which to compute the mean and variance: all except the batch dimension
        axes = tuple(range(1, len(x.shape)))
        
        # Compute mean and variance along the feature dimension for each sample        
        layer_mean = tf.reduce_mean(x, axis=axes, keepdims=True)
        layer_var = tf.math.reduce_variance(x, axis=axes, keepdims=True)
        
        # Normalize the input        
        self.X_centered = tf.subtract(x, layer_mean)
        self.stddev_inv = tf.math.rsqrt(layer_var + self.epsilon)
        self.X_norm = tf.multiply(self.X_centered, self.stddev_inv)
        
        # Scale and shift
        return tf.add(tf.multiply(self.gamma, self.X_norm), self.beta)
    
    
    def backward(self, grad_output: tf.Tensor) -> tf.Tensor:
        """
        Backward pass of the layer normalization layer (layer i)
        
        Parameters:
        - grad_output (tf.Tensor): The gradient of the loss with respect to the output of the layer: dL/dO_i
        
        Returns:
        - tf.Tensor: The gradient of the loss with respect to the input of the layer: dL/dX_i ≡ dL/dO_{i-1}
        
        Raises:
        - AssertionError: If the gamma and beta parameters are not initialized or the optimizer is not set.
        """
        
        # Assert that the gamma and beta parameters are initialized and the optimizer is set
        assert isinstance(self.optimizer, Optimizer), "Optimizer is not set. Please set an optimizer before training the model."
        assert isinstance(self.gamma, tf.Tensor), "Gamma parameter is not initialized. Please call the layer with input data."
        assert isinstance(self.beta, tf.Tensor), "Beta parameter is not initialized. Please call the layer with input data."
        
        # Extract the shape of the input
        batch_size = grad_output.shape[0]
        
        # Gradients of beta and gamma
        d_beta = tf.reduce_sum(grad_output, axis=tuple(range(len(grad_output.shape) - 1)), keepdims=True)
        d_gamma = tf.reduce_sum(tf.multiply(grad_output, self.X_norm), axis=tuple(range(len(grad_output.shape) - 1)), keepdims=True)
        
        # Gradient with respect to the normalized input
        dx_hat = tf.multiply(grad_output, self.gamma)
        
        # Gradient with respect to variance
        d_var = tf.multiply(tf.reduce_sum(tf.multiply(dx_hat, self.X_centered), axis=tuple(range(1, len(grad_output.shape)))), -0.5, tf.pow(self.stddev_inv, 3))
        
        # Gradient with respect to mean
        d_mean = tf.add(
            tf.reduce_sum(tf.multiply(dx_hat, tf.negative(self.stddev_inv)), axis=tuple(range(1, len(grad_output.shape))), keepdims=True),
            tf.multiply(d_var, tf.reduce_mean(tf.multiply(-2, self.X_centered), axis=tuple(range(1, len(grad_output.shape))), keepdims=True))
        )
        
        # Gradient with respect to input x
        dx = tf.add(
            tf.multiply(dx_hat, self.stddev_inv),
            tf.divide(
                tf.multiply(
                    tf.multiply(2, d_var),
                    self.X_centered
                ),
                batch_size
            ),
            tf.divide(
                d_mean,
                batch_size
            )
        )
        
        # Update the gamma parameter
        self.gamma = self.optimizer.update(
            layer = self,
            param_name = "gamma",
            params = self.gamma,
            grad_params = d_gamma
        )
        
        # Update the beta parameter
        self.beta = self.optimizer.update(
            layer = self,
            param_name = "beta",
            params = self.beta,
            grad_params = d_beta
        )
        
        # Return the gradients of the loss with respect to the input
        return dx # dL/dX_i ≡ dL/dO_{i-1}
    
    
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        
        Raises:
        - AssertionError: If the gamma and beta parameters are not initialized
        """
        
        # Assert that the gamma and beta parameters are initialized
        assert isinstance(self.gamma, tf.Tensor), "Gamma parameter is not initialized. Please call the layer with input data."
        assert isinstance(self.beta, tf.Tensor), "Beta parameter is not initialized. Please call the layer with input data."
        
        # Return the number of parameters in the layer
        return int(tf.add(tf.size(self.gamma), tf.size(self.beta)))
    
    
    def output_shape(self) -> tf.TensorShape:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tf.TensorShape: The shape of the output of the layer
        """
        
        return self.input_shape

    
    def init_params(self, x: tf.Tensor) -> None:
        """
        Method to initialize the parameters of the layer
        
        Parameters:
        - x (tf.Tensor): Input data
        """
                 
        # Store the input dimension
        self.input_shape = x.shape
        
        # Extract the shape of the parameters: all except the batch dimension
        feature_shape = self.input_shape[1:]
        
        # Initialize the scale and shift parameters
        self.gamma = tf.ones(feature_shape)
        self.beta = tf.zeros(feature_shape)
        
        # Update the initialization flag
        self.initialized = True