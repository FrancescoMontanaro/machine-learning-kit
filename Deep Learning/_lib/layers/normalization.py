import numpy as np
from typing import Optional

from .base import Layer
from ..optimizers import Optimizer


class BatchNormalization1D(Layer):
    
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
     
     
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Method to set the input shape of the layer and initialize the scale and shift parameters
        
        Parameters:
        - x (np.ndarray): Input data. Shape: (Batch size, number of features)
        
        Returns:
        - np.ndarray: Output of the layer after the forward pass
        
        Raises:
        - ValueError: If the input shape is not 2D
        """
        
        # Check if the input shape is valid
        if len(x.shape) != 2:
            raise ValueError(f"Invalid input shape. Input must be a 2D array. The shape must be (Batch size, number of features). Got shape: {x.shape}")
        
        # Unpack the shape of the input data for better readability
        batch_size, num_features = x.shape
        
        # Store the input dimension
        self.input_shape = (batch_size, num_features)
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params(num_features)
        
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
    
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the batch normalization layer.
        
        Parameters:
        - x (np.ndarray): The input tensor.
        - training (bool): Whether the model is in training mode.
        
        Returns:
        - np.ndarray: The normalized tensor.
        """
        
        # The layer is in training phase
        if self.training:
            # Calculate batch mean and variance
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean # type: ignore
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var # type: ignore

            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
           
        # The layer is in inference phase 
        else:
            # Use running statistics for normalization
            mean = self.running_mean
            var = self.running_var
            
        # Normalize the input
        self.X_centered = x - mean
        self.stddev_inv = 1 / np.sqrt(var + self.epsilon) # type: ignore
        self.X_norm = self.X_centered * self.stddev_inv
            
        return self.gamma * self.X_norm + self.beta # type: ignore
    
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of the batch normalization layer.
        
        Parameters:
        - grad_output (np.ndarray): The gradient of the loss with respect to the output of the layer.
        
        Returns:
        - np.ndarray: The gradient of the loss with respect to the input of the layer.
        """
        
        # Check if the optimizer is set
        if not isinstance(self.optimizer, Optimizer):
            raise ValueError("Optimizer is not set. Please set an optimizer before training the model.")
        
        # Number of samples in the batch
        batch_size = grad_output.shape[0]
        
        # Gradients with respect to gamma and beta
        d_beta = np.sum(grad_output, axis=0)
        d_gamma = np.sum(grad_output * self.X_norm, axis=0)

        # Gradient with respect to the normalized input
        dx_hat = grad_output * self.gamma
        
        # Gradient with respect to variance
        d_var = np.sum(dx_hat * self.X_centered, axis=0) * -0.5 * self.stddev_inv**3
        
        # Gradient with respect to mean
        d_mean = np.sum(dx_hat * -self.stddev_inv, axis=0) + d_var * np.mean(-2. * self.X_centered, axis=0)

        # Gradient with respect to input x
        dx = (dx_hat * self.stddev_inv) + (d_var * 2 * self.X_centered / batch_size) + (d_mean / batch_size)
        
        # Update the gamma parameter
        self.gamma = self.optimizer.update(
            layer = self,
            param_name = "gamma",
            params = self.gamma, # type: ignore
            grad_params = d_gamma
        )
        
        # Update the beta parameter
        self.beta = self.optimizer.update(
            layer = self,
            param_name = "beta",
            params = self.beta, # type: ignore
            grad_params = d_beta
        )
        
        return dx
    
    
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        """
        
        # Return the number of parameters in the layer
        return self.gamma.size + self.beta.size # type: ignore
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        return self.input_shape
    
    
    def init_params(self, num_features: int) -> None:
        """
        Method to initialize the parameters of the layer
        
        Parameters:
        - num_features (int): The number of features in the input data
        """
        
        # Initialize the scale and shift parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Initialize the running mean and variance
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Update the initialization flag
        self.initialized = True
 

class LayerNormalization1D(Layer):
    
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
     
     
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Method to set the input shape of the layer and initialize the scale and shift parameters
        
        Parameters:
        - x (np.ndarray): Input data. Shape: (Batch size, number of features)
        
        Returns:
        - np.ndarray: Output of the layer after the forward pass
        
        Raises:
        - ValueError: If the input shape is not 2D
        """
        
        # Check if the input shape is valid
        if len(x.shape) != 2:
            raise ValueError(f"Invalid input shape. Input must be a 2D array. The shape must be (Batch size, number of features). Got shape: {x.shape}")
                
        # Unpack the shape of the input data for better readability
        batch_size, num_features = x.shape
        
        # Store the input dimension
        self.input_shape = (batch_size, num_features)
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params(num_features)
        
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
    
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer normalization layer.
        
        Parameters:
        - x (np.ndarray): The input tensor. Shape: (Batch size, number of features)
        - training (bool): Whether the model is in training mode.
        
        Returns:
        - np.ndarray: The normalized tensor.
        """
        
        # Compute mean and variance along the feature dimension for each sample
        layer_mean = x.mean(axis=1, keepdims=True)
        layer_var = x.var(axis=1, keepdims=True)
        
        # Normalize the input
        self.X_centered = x - layer_mean
        self.stddev_inv = 1 / np.sqrt(layer_var + self.epsilon)
        self.X_norm = self.X_centered * self.stddev_inv
        
        # Scale and shift
        return self.gamma * self.X_norm + self.beta
    
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer normalization layer.
        
        Parameters:
        - grad_output (np.ndarray): The gradient of the loss with respect to the output of the layer.
        
        Returns:
        - np.ndarray: The gradient of the loss with respect to the input of the layer.
        """
        
        # Check if the optimizer is set
        if not isinstance(self.optimizer, Optimizer):
            raise ValueError("Optimizer is not set. Please set an optimizer before training the model.")
        
        # Extract the shape of the input
        batch_size = grad_output.shape[0]
        
        # Gradients of beta and gamma
        d_beta = grad_output.sum(axis=0)
        d_gamma = np.sum(grad_output * self.X_norm, axis=0)
        
        # Compute the gradient of the loss with respect to the input
        dx_norm = grad_output * self.gamma
        d_var = np.sum(dx_norm * self.X_centered, axis=0) * -0.5 * self.stddev_inv**3
        d_mean = np.sum(dx_norm * -self.stddev_inv, axis=0) + d_var * np.mean(-2 * self.X_centered, axis=0)
        d_x = (dx_norm * self.stddev_inv) + (d_var * 2 * self.X_centered / batch_size) + (d_mean / batch_size)
        
        # Update the gamma parameter
        self.gamma = self.optimizer.update(
            layer = self,
            param_name = "gamma",
            params = self.gamma, # type: ignore
            grad_params = d_gamma
        )
        
        # Update the beta parameter
        self.beta = self.optimizer.update(
            layer = self,
            param_name = "beta",
            params = self.beta, # type: ignore
            grad_params = d_beta
        )
        
        # Return the gradients of the loss with respect to the input
        return d_x
    
    
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        """
        
        # Return the number of parameters in the layer
        return self.gamma.size + self.beta.size # type: ignore
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        return self.input_shape

    
    def init_params(self, num_features: int) -> None:
        """
        Method to initialize the parameters of the layer
        
        Parameters:
        - num_features (int): The number of features in the input data
        """
        
        # Initialize the scale and shift parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Update the initialization flag
        self.initialized = True