import numpy as np
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
        
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Method to set the input shape of the layer and initialize the weights and bias
        
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
        
        # Store the input shape
        self.input_shape = (batch_size, num_features)
        
        # Check if the layer is initialized
        if not self.initialized:
            # Initialize the layer
            self.init_params(num_features)
            
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
    

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Method to compute the output of the layer
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Output of the layer
        
        Raises:
        - ValueError: If the weights and bias are not initialized
        """
        
        # Store the input for the backward pass
        self.input = x
        
        # Compute the linear combination of the weights and features
        self.linear_comb = np.dot(x, self.weights) + self.bias # type: ignore
        
        # Return the output of the neuron
        return self.activation(self.linear_comb) if self.activation is not None else self.linear_comb
    
    
    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        """
        Method to compute the gradient of the loss with respect to the input of the layer
        
        Parameters:
        - loss_gradient (np.ndarray): Gradient of the loss with respect to the output of the layer
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer
        
        Raises:
        - ValueError: If the optimizer is not set
        """
        
        # Check if the optimizer is set
        if not isinstance(self.optimizer, Optimizer):
            raise ValueError("Optimizer is not set. Please set an optimizer before training the model.")
        
        # Compute the gradient of the loss with respect to the input
        if self.activation is not None:
            # Compute derivative of the activation function
            activation_derivative = self.activation.derivative(self.linear_comb)
            
            # Multiply the incoming gradient by the activation derivative
            gradient = loss_gradient * activation_derivative
            
        else:
            # If the activation function is not defined, the gradient is the same as the incoming gradient
            gradient = loss_gradient

        # Compute gradients with respect to input, weights, and biases
        grad_input = np.dot(gradient, self.weights.T) # type: ignore
        grad_weights = np.dot(self.input.T, gradient)
        grad_bias = np.sum(gradient, axis=0)

        # Update weights
        self.weights = self.optimizer.update(
            layer = self,
            param_name = "weights",
            params = self.weights, # type: ignore
            grad_params = grad_weights
        )
        
        # Update bias
        self.bias = self.optimizer.update(
            layer = self,
            param_name = "bias",
            params = self.bias, # type: ignore
            grad_params = grad_bias
        )

        return grad_input
    
    
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        """
        
        # Return the number of parameters in the layer
        return self.weights.size + self.bias.size # type: ignore
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        return (self.input_shape[0], self.num_units)
    
        
    def init_params(self, num_features: int) -> None:
        """
        Method to initialize the weights and bias of the layer
        
        Parameters:
        - num_features (int): Number of features in the input data
        """
        
        # Initialize the weights and bias with random values
        self.weights = np.random.uniform(-np.sqrt(1 / num_features), np.sqrt(1 / num_features), (num_features, self.num_units))
        self.bias = np.zeros(self.num_units)
        
        # Update the initialization flag
        self.initialized = True