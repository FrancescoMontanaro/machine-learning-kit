from typing import Any
import tensorflow as tf


class Optimizer:
    
    ### Magic methods ###
    
    def __init__(self) -> None:
        """
        Class constructor
        """
        
        # Create a dictionary to store the parameters of the model
        self.params_registry = {}
        
    
    ### Public methods ###

    def update(self, layer: Any, param_name: str, params: tf.Tensor, grad_params: tf.Tensor) -> tf.Tensor:
        """
        Abstract method to update the parameters of the model
        
        Parameters:
        - layer (Any): Instance of the Layer being optimized
        - param_name (str): Name of the parameters to be updated
        - params (tf.Tensor): Parameters to be updated
        - grad_params (tf.Tensor): Gradient of the parameters with respect to the loss
        
        Returns:
        - tf.Tensor: Updated parameters
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'update' is not implemented.")


    def init_params_registry(self) -> None:
        """
        Method to initialize the parameters of the optimizer
        """
        
        # Reset the parameters registry
        self.params_registry = {}