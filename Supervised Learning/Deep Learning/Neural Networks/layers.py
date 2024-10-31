import uuid
import numpy as np
from functools import wraps
from typing import Optional, Callable

from optimizers import _AbstractOptimizer
from activations import _AbstractActivationFn


class _AbstractLayer:
    
    ### Magic methods ###
    
    def __init__(self, name: Optional[str] = None) -> None:
        """
        Class constructor of the abstract class Layer
        """
        
        # Store the name of the layer
        self.name = name
        self.training = True
        
        # Generate a unique identifier for the layer
        self.__uuid = uuid.uuid4().hex
    
    
    ### Public methods ###
    
    def set_input_shape(self, input_shape: tuple[int, int]) -> None:
        """
        Abstract method to set the input shape of the layer.
        
        Parameters:
        - input_shape (tuple): Shape of the input of the layer (batch_size, num_features)
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'set_input_shape' is not implemented.")
    
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Abstract method to compute the output of the layer. 
        The implementation should be done in the child classes
        
        Parameters:
        - x (np.ndarray): Features of the dataset
        
        Returns:
        - np.ndarray: Output of the layer
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'forward' is not implemented.")
    
    
    def backward(self, grad_output: np.ndarray, optimizer: _AbstractOptimizer) -> np.ndarray:
        """
        Abstract method to compute the gradient of the loss with respect to the input of the layer. 
        The implementation should be done in the child classes
        
        Parameters:
        - grad_output (np.ndarray): Gradient of the loss with respect to the output of the layer
        - optimizer (_AbstractOptimizer): Optimizer to update the parameters of the model
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        raise NotImplementedError("The method 'backward' is not implemented.")
        
    
    def output_shape(self) -> tuple:
        """
        Abstract method to return the output shape of the layer. 
        The implementation should be done in the child classes
        
        Returns:
        - tuple: The shape of the output of the layer
        
        Raises:
        - NotImplementedError: If the method is not implemented
        """
        
        raise NotImplementedError("The method 'output_shape' is not implemented.")
    
    
    def get_uuid(self) -> str:
        """
        Method to return the unique identifier of the layer
        
        Returns:
        - str: Unique identifier of the layer
        """
        
        return self.__uuid
    
    
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        """
        
        return 0
    
 
class Dense(_AbstractLayer):
    
    ### Magic methods ###
    
    def __init__(self, num_units: int, activation: Optional[_AbstractActivationFn] = None, name: Optional[str] = None) -> None:
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
        
    
    ### Public methods ###
    
    @staticmethod
    def check_initialization(func: Callable) -> Callable:
        """
        Decorator to check if the weights and bias are initialized before calling forward and backward methods
        
        Parameters:
        - func (Callable): Function to be wrapped
        
        Returns:
        - Callable: Wrapper function
        
        Raises:
        - ValueError: If the weights and bias are not initialized
        """
        
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if the weights and bias are initialized
            if not isinstance(self.weights, np.ndarray) or not isinstance(self.bias, np.ndarray):
                raise ValueError("Weights and bias are not initialized.")
            
            # Call the function
            return func(self, *args, **kwargs)
        
        # Return the wrapper function
        return wrapper
    
    
    def set_input_shape(self, input_shape: tuple[int, int]) -> None:
        """
        Method to set the input shape of the layer and initialize the weights and bias
        
        Parameters:
        - input_shape (tuple): Shape of the input of the layer (batch_size, num_features)
        
        Raises:
        - ValueError: If the input shape is not 2D
        """
        
        # Check if the input shape is valid
        if len(input_shape) != 2:
            raise ValueError(f"Invalid input shape. Expected 2D input, but got {len(input_shape)}D input.")
        
        # Store the input dimension
        self.input_shape = input_shape # (Batch size, number of features)
        
        # Check if the weights and bias are not initialized yet
        if self.weights is None or self.bias is None:
            # Initialize the weights and bias with random values
            self.weights = np.random.uniform(-np.sqrt(1 / input_shape[-1]), np.sqrt(1 / input_shape[-1]), (input_shape[-1], self.num_units))
            self.bias = np.zeros(self.num_units)
    
    
    @check_initialization
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
        
        # Assert that the weights are a numpy array
        assert isinstance(self.weights, np.ndarray)
        assert isinstance(self.bias, np.ndarray)
        
        # Store the input for the backward pass
        self.input = x
        
        # Compute the linear combination of the weights and features
        self.linear_comb = np.dot(x, self.weights) + self.bias
        
        # Return the output of the neuron
        return self.activation(self.linear_comb) if self.activation is not None else self.linear_comb
    
    
    @check_initialization
    def backward(self, loss_gradient: np.ndarray, optimizer: _AbstractOptimizer) -> np.ndarray:
        """
        Method to compute the gradient of the loss with respect to the input of the layer
        
        Parameters:
        - loss_gradient (np.ndarray): Gradient of the loss with respect to the output of the layer
        - optimizer (_AbstractOptimizer): Optimizer to update the parameters of the model
        
        Returns:
        - np.ndarray: Gradient of the loss with respect to the input of the layer
        """
        
        # Assert that the weights are a numpy array
        assert isinstance(self.weights, np.ndarray)
        assert isinstance(self.bias, np.ndarray)
        
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
        grad_input = np.dot(gradient, self.weights.T)
        grad_weights = np.dot(self.input.T, gradient)
        grad_bias = np.sum(gradient, axis=0)
        
        # Get the current optimizer layer index
        optimizer_layer_idx = optimizer.layer_id

        # Update weights
        optimizer.layer_id = f"{optimizer_layer_idx}_weights"
        self.weights = optimizer.update(self.weights, grad_weights)
        
        # Update bias
        optimizer.layer_id = f"{optimizer_layer_idx}_bias"
        self.bias = optimizer.update(self.bias, grad_bias)
        
        # Reset the optimizer layer index
        optimizer.layer_id = optimizer_layer_idx

        return grad_input
    
    
    @check_initialization
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        """
        
        # Assert that the weights are a numpy array
        assert isinstance(self.weights, np.ndarray)
        assert isinstance(self.bias, np.ndarray)
        
        # Return the number of parameters in the layer
        return self.weights.size + self.bias.size
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        return (self.input_shape[0], self.num_units)
                
    
class BatchNormalization1D(_AbstractLayer):
    
    ### Magic methods ###
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.9, name: Optional[str] = None) -> None:
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
     
     
    ### Public methods ###
    
    @staticmethod
    def check_initialization(func: Callable) -> Callable:
        """
        Decorator to check if the gamma and beta parameters are initialized before calling forward and backward methods
        
        Parameters:
        - func (Callable): Function to be wrapped
        
        Returns:
        - Callable: Wrapper function
        
        Raises:
        - ValueError: If the gamma and beta parameters are not initialized
        """
        
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if the gamma and beta parameters are initialized
            if not isinstance(self.gamma, np.ndarray) or not isinstance(self.beta, np.ndarray) or not isinstance(self.running_mean, np.ndarray) or not isinstance(self.running_var, np.ndarray):
                raise ValueError("Gamma and beta parameters are not initialized.")
            
            # Call the function
            return func(self, *args, **kwargs)
        
        # Return the wrapper function
        return wrapper
    
    
    def set_input_shape(self, input_shape: tuple[int, int]) -> None:
        """
        Method to set the input shape of the layer and initialize the scale and shift parameters
        
        Parameters:
        - input_shape (tuple): Shape of the input of the layer (batch_size, num_features)
        
        Raises:
        - ValueError: If the input shape is not 2D
        """
        
        # Check if the input shape is valid
        if len(input_shape) != 2:
            raise ValueError(f"Invalid input shape. Expected 2D input, but got {len(input_shape)}D input.")
        
        # Store the input dimension
        self.input_shape = input_shape # (Batch size, number of features)
        
        # Initialize the scale and shift parameters
        self.gamma = np.ones(input_shape[-1])
        self.beta = np.zeros(input_shape[-1])
        
        # Initialize the moving average of the mean and variance
        self.running_mean = np.zeros((1, input_shape[-1]))
        self.running_var = np.ones((1, input_shape[-1]))
    
    
    @check_initialization
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the batch normalization layer.
        
        Parameters:
        - x (np.ndarray): The input tensor.
        - training (bool): Whether the model is in training mode.
        
        Returns:
        - np.ndarray: The normalized tensor.
        """
        
        # Assert that the gamma, beta, running mean, and running variance are numpy arrays
        assert isinstance(self.gamma, np.ndarray)
        assert isinstance(self.beta, np.ndarray)
        assert isinstance(self.running_mean, np.ndarray)
        assert isinstance(self.running_var, np.ndarray)
        
        # If the model is in training mode, compute the mean and variance and update the running mean and variance
        if self.training:
            # Compute mean and variance
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            
            # Update running mean and variance for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
        else:
            # Use running mean and variance for inference
            mean = self.running_mean
            var = self.running_var
        
        # Normalize the input
        self.X_centered = x - mean
        self.stddev_inv = 1 / np.sqrt(var + self.epsilon)
        self.X_norm = self.X_centered * self.stddev_inv
        
        # Scale and shift
        return self.gamma * self.X_norm + self.beta
    
    
    @check_initialization
    def backward(self, grad_output: np.ndarray, optimizer: _AbstractOptimizer) -> np.ndarray:
        """
        Backward pass of the batch normalization layer.
        
        Parameters:
        - grad_output (np.ndarray): The gradient of the loss with respect to the output of the layer.
        - optimizer (_AbstractOptimizer): The optimizer to update the parameters of the model.
        
        Returns:
        - np.ndarray: The gradient of the loss with respect to the input of the layer.
        """
        
        # Assert that the gamma, beta, running mean, and running variance are numpy arrays
        assert isinstance(self.gamma, np.ndarray)
        assert isinstance(self.beta, np.ndarray)
        assert isinstance(self.running_mean, np.ndarray)
        assert isinstance(self.running_var, np.ndarray)
        
        # Gradients of beta and gamma
        d_beta = grad_output.sum(axis=0)
        d_gamma = np.sum(grad_output * self.X_norm, axis=0)
        
        # Extract the shape of the input
        N, D = grad_output.shape
        
        # Compute the gradient of the loss with respect to the input
        dx_norm = grad_output * self.gamma
        d_var = np.sum(dx_norm * self.X_centered, axis=0) * -0.5 * self.stddev_inv**3
        d_mean = np.sum(dx_norm * -self.stddev_inv, axis=0) + d_var * np.mean(-2 * self.X_centered, axis=0)
        d_x = (dx_norm * self.stddev_inv) + (d_var * 2 * self.X_centered / N) + (d_mean / N)
        
        # Update the parameters of the layer
        optimizer.layer_id = f"{optimizer.layer_id}_gamma"
        self.gamma = optimizer.update(self.gamma, d_gamma)
        
        # Update the bias
        optimizer.layer_id = f"{optimizer.layer_id}_beta"
        self.beta = optimizer.update(self.beta, d_beta)
        
        # Reset the optimizer layer index
        optimizer.layer_id = optimizer.layer_id
        
        # Return the gradients of the loss with respect to the input
        return d_x
    
    
    @check_initialization
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        """
        
        # Assert that the gamma and beta are numpy arrays
        assert isinstance(self.gamma, np.ndarray)
        assert isinstance(self.beta, np.ndarray)
        
        # Return the number of parameters in the layer
        return self.gamma.size + self.beta.size
    
    
    @check_initialization
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        return self.input_shape
    

class LayerNormalization1D(_AbstractLayer):
    
    ### Magic methods ###
    
    def __init__(self, epsilon: float = 1e-5, momentum: float = 0.9, name: Optional[str] = None) -> None:
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
     
     
    ### Public methods ###
    
    @staticmethod
    def check_initialization(func: Callable) -> Callable:
        """
        Decorator to check if the gamma and beta parameters are initialized before calling forward and backward methods
        
        Parameters:
        - func (Callable): Function to be wrapped
        
        Returns:
        - Callable: Wrapper function
        
        Raises:
        - ValueError: If the gamma and beta parameters are not initialized
        """
        
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if the gamma and beta parameters are initialized
            if not isinstance(self.gamma, np.ndarray) or not isinstance(self.beta, np.ndarray) or not isinstance(self.running_mean, np.ndarray) or not isinstance(self.running_var, np.ndarray):
                raise ValueError("Gamma and beta parameters are not initialized.")
            
            # Call the function
            return func(self, *args, **kwargs)
        
        # Return the wrapper function
        return wrapper
    
    
    def set_input_shape(self, input_shape: tuple[int, int]) -> None:
        """
        Method to set the input shape of the layer and initialize the scale and shift parameters
        
        Parameters:
        - input_shape (tuple): Shape of the input of the layer (batch_size, num_features)
        
        Raises:
        - ValueError: If the input shape is not 2D
        """
        
        # Check if the input shape is valid
        if len(input_shape) != 2:
            raise ValueError(f"Invalid input shape. Expected 2D input, but got {len(input_shape)}D input.")
        
        # Store the input dimension
        self.input_shape = input_shape # (Batch size, number of features)
        
        # Initialize the scale and shift parameters
        self.gamma = np.ones(input_shape[-1])
        self.beta = np.zeros(input_shape[-1])
        
        # Initialize the moving average of the mean and variance
        self.running_mean = np.zeros((1, input_shape[-1]))
        self.running_var = np.ones((1, input_shape[-1]))
    
    
    @check_initialization
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the layer normalization layer.
        
        Parameters:
        - x (np.ndarray): The input tensor.
        - training (bool): Whether the model is in training mode.
        
        Returns:
        - np.ndarray: The normalized tensor.
        """
        
        # Assert that the gamma, beta, running mean, and running variance are numpy arrays
        assert isinstance(self.gamma, np.ndarray)
        assert isinstance(self.beta, np.ndarray)
        assert isinstance(self.running_mean, np.ndarray)
        assert isinstance(self.running_var, np.ndarray)
        
        # If the model is in training mode, compute the mean and variance and update the running mean and variance
        if self.training:
            # Compute mean and variance
            mean = x.mean(axis=1)
            var = x.var(axis=1)
            
            # Update running mean and variance for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
        else:
            # Use running mean and variance for inference
            mean = self.running_mean
            var = self.running_var
        
        # Normalize the input
        self.X_centered = x - mean
        self.stddev_inv = 1 / np.sqrt(var + self.epsilon)
        self.X_norm = self.X_centered * self.stddev_inv
        
        # Scale and shift
        return self.gamma * self.X_norm + self.beta
    
    
    @check_initialization
    def backward(self, grad_output: np.ndarray, optimizer: _AbstractOptimizer) -> np.ndarray:
        """
        Backward pass of the batch normalization layer.
        
        Parameters:
        - grad_output (np.ndarray): The gradient of the loss with respect to the output of the layer.
        - optimizer (_AbstractOptimizer): The optimizer to update the parameters of the model.
        
        Returns:
        - np.ndarray: The gradient of the loss with respect to the input of the layer.
        """
        
        # Assert that the gamma, beta, running mean, and running variance are numpy arrays
        assert isinstance(self.gamma, np.ndarray)
        assert isinstance(self.beta, np.ndarray)
        assert isinstance(self.running_mean, np.ndarray)
        assert isinstance(self.running_var, np.ndarray)
        
        # Gradients of beta and gamma
        d_beta = grad_output.sum(axis=0)
        d_gamma = np.sum(grad_output * self.X_norm, axis=0)
        
        # Extract the shape of the input
        N, D = grad_output.shape
        
        # Compute the gradient of the loss with respect to the input
        dx_norm = grad_output * self.gamma
        d_var = np.sum(dx_norm * self.X_centered, axis=0) * -0.5 * self.stddev_inv**3
        d_mean = np.sum(dx_norm * -self.stddev_inv, axis=0) + d_var * np.mean(-2 * self.X_centered, axis=0)
        d_x = (dx_norm * self.stddev_inv) + (d_var * 2 * self.X_centered / N) + (d_mean / N)
        
        # Update the parameters of the layer
        optimizer.layer_id = f"{optimizer.layer_id}_gamma"
        self.gamma = optimizer.update(self.gamma, d_gamma)
        
        # Update the bias
        optimizer.layer_id = f"{optimizer.layer_id}_beta"
        self.beta = optimizer.update(self.beta, d_beta)
        
        # Reset the optimizer layer index
        optimizer.layer_id = optimizer.layer_id
        
        # Return the gradients of the loss with respect to the input
        return d_x
    
    
    @check_initialization
    def count_params(self) -> int:
        """
        Method to count the number of parameters in the layer
        
        Returns:
        - int: Number of parameters in the layer
        """
        
        # Assert that the gamma and beta are numpy arrays
        assert isinstance(self.gamma, np.ndarray)
        assert isinstance(self.beta, np.ndarray)
        
        # Return the number of parameters in the layer
        return self.gamma.size + self.beta.size
    
    
    @check_initialization
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        return self.input_shape
    
    
class Dropout(_AbstractLayer):
    
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
        
    
    ### Public methods ###
    
    def set_input_shape(self, input_shape: tuple[int, int]) -> None:
        """
        Method to set the input shape of the layer.
        
        Parameters:
        - input_shape (tuple): Shape of the input of the layer (batch_size, num_features)
        """
        
        # Store the input dimension
        self.input_shape = input_shape # (Batch size, number of features)
    
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the dropout layer.
        
        Parameters:
        - x (np.ndarray): The input tensor.
        
        Returns:
        - np.ndarray: The output tensor.
        """
        
        # Generate the mask
        self.mask = np.random.rand(*x.shape) > self.rate
        
        if self.training:
            # Scale the output during training
            return x * self.mask / (1 - self.rate)
        else:
            # Return the output during inference
            return x
    
    
    def backward(self, grad_output: np.ndarray, optimizer: _AbstractOptimizer) -> np.ndarray:
        """
        Backward pass of the dropout layer.
        
        Parameters:
        - grad_output (np.ndarray): The gradient of the loss with respect to the output of the layer.
        - optimizer (_AbstractOptimizer): The optimizer to update the parameters of the model.
        
        Returns:
        - np.ndarray: The gradient of the loss with respect to the input of the layer.
        """
        
        # Scale the gradient
        return grad_output * self.mask / (1 - self.rate)
    
    
    def output_shape(self) -> tuple:
        """
        Method to return the output shape of the layer
        
        Returns:
        - tuple: The shape of the output of the layer
        """
        
        return self.input_shape