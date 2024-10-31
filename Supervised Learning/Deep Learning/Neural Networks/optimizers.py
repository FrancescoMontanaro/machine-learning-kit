import numpy as np


class _AbstractOptimizer:
    
    ### Magic methods ###
    
    def __init__(self) -> None:
        """
        Class constructor
        """
        
        # Initialize the layer index to keep track of the current layer being optimized
        self.layer_id = ""
    
    
    ### Public methods ###

    def update(self, params: np.ndarray, grad_params: np.ndarray) -> np.ndarray:
        """
        Abstract method to update the parameters of the model
        
        Parameters:
        - params (np.ndarray): Parameters to be updated
        - grad_params (np.ndarray): Gradient of the parameters with respect to the loss
        
        Returns:
        - np.ndarray: Updated parameters
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'update' is not implemented.")


    def init_params(self) -> None:
        """
        Method to initialize the parameters of the optimizer
        """
        
        # Initialize the optimizer
        pass


class SGD(_AbstractOptimizer):
    
    ### Magic methods ###
    
    def __init__(self, learning_rate: float, momentum: float = 0.0) -> None:
        """
        Class constructor for the Stochastic Gradient Descent (SGD) optimizer.
        Momentum accelerates updates along directions where the gradients are consistent 
        and reduces oscillations as the optimizer approaches the minimum of the loss function. 
        This allows the optimizer to converge more quickly while also reducing the risk of 
        oscillating around the minimum.
        
        Parameters:
        - learning_rate (float): Learning rate for the optimizer
        - momentum (float): Momentum for the optimizer
        """
        
        # Store the learning rate and momentum
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize the velocities
        self.velocities = {}
    
    
    ### Public methods ###
    
    def update(self, params: np.ndarray, grad_params: np.ndarray) -> np.ndarray:
        """
        Method to update the parameters of the model.
        
        Parameters:
        - params (np.ndarray): Parameters of the model
        - grad_params (np.ndarray): Gradient of the parameters
        
        Returns:
        - np.ndarray: Updated parameters
        """
        
        # Initialize the velocities
        if self.layer_id not in self.velocities:
            self.velocities[self.layer_id] = np.zeros_like(grad_params)
            
        # Update velocity of the layer
        self.velocities[self.layer_id] = self.momentum * self.velocities[self.layer_id] - self.learning_rate * grad_params
        
        # Update the parameters
        return params + self.velocities[self.layer_id]
    
    
    def init_params(self) -> None:
        """
        Method to initialize the parameters of the optimizer
        """
        
        # Initialize the velocities
        self.velocities = {}
    
    
class Adam(_AbstractOptimizer):
        
    ### Magic methods ###
        
    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.00) -> None:
        """
        Class constructor
        
        Parameters:
        - learning_rate (float): Learning rate for the optimizer
        - beta1 (float): Exponential decay rate for the first moment estimates
        - beta2 (float): Exponential decay rate for the second moment estimates
        - epsilon (float): Small value to prevent division by zero
        - weight_decay (float): Weight decay for the optimizer
        """
        
        # Store the parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # Initialize the moments and time step
        self.moments = {}
        self.t = 0
        
        
    ### Public methods ###
    
    def update(self, params: np.ndarray, grad_params: np.ndarray) -> np.ndarray:
        """
        Method to update the parameters of the model
        
        Parameters:
        - params (np.ndarray): Parameters of the model
        - grad_params (np.ndarray): Gradient of the parameters
        
        Returns:
        - np.ndarray: Updated parameters
        """
        
        # Initialize the moments
        if self.layer_id not in self.moments:
            self.moments[self.layer_id] = {
                "m": np.zeros_like(grad_params),
                "v": np.zeros_like(grad_params)
            }
            
        # Retrieve layer-specific moment vectors
        m = self.moments[self.layer_id]["m"]
        v = self.moments[self.layer_id]["v"]
        
        # Update the time step
        self.t += 1
        
        # Compute the first and second moment estimates
        m = self.beta1 * m + (1 - self.beta1) * grad_params
        v = self.beta2 * v + (1 - self.beta2) * (grad_params ** 2)
        
        # Compute the bias-corrected first and second moment estimates
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        
        # Save updated moments back to the dictionary
        self.moments[self.layer_id]["m"] = m
        self.moments[self.layer_id]["v"] = v
        
        # Update the parameters
        return params - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * params)
    
    
    def init_params(self) -> None:
        """
        Method to initialize the parameters of the optimizer
        """
        
        # Initialize the moments and time step
        self.moments = {}
        self.t = 0