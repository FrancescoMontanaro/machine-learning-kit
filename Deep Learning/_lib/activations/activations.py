import numpy as np
import tensorflow as tf

from .base import Activation


class ReLU(Activation):
    
    ### Magic methods ###
    
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the output of the ReLU activation function.
        
        Parameters:
        - x (tf.Tensor): Input to the activation function
        
        Returns:
        - tf.Tensor: Output of the activation function
        """
        
        # Compute the ReLU
        return tf.maximum(0, x)


    ### Public methods ###

    def derivative(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the ReLU activation function.
        
        Parameters:
        - x (tf.Tensor): Input to the activation function
        
        Returns:
        - tf.Tensor: Derivative of the activation function
        """
        
        # Compute the derivative of the ReLU
        return tf.where(tf.greater(x, 0), 1, 0)
    
    
class Sigmoid(Activation):
    
    ### Magic methods ###
    
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the output of the sigmoid activation function.
        
        Parameters:
        - x (tf.Tensor): Input to the activation function
        
        Returns:
        - tf.Tensor: Output of the activation function
        """
        
        # Compute the sigmoid
        return tf.divide(1.0, tf.add(1.0, tf.exp(tf.negative(x))))


    ### Public methods ###

    def derivative(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the sigmoid activation function.
        
        Parameters:
        - x (tf.Tensor): Input to the activation function
        
        Returns:
        - tf.Tensor: Derivative of the activation function
        """
        
        # Compute the derivative of the sigmoid
        sigmoid_x = self(x)
        
        return tf.multiply(sigmoid_x, tf.subtract(1, sigmoid_x))
    
    
class Softmax(Activation):
        
    ### Magic methods ###
    
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the output of the softmax activation function.
        
        Parameters:
        - x (tf.Tensor): Input to the activation function
        
        Returns:
        - tf.Tensor: Output of the activation function
        """
        
        # Compute the softmax
        exp_x = tf.exp(tf.subtract(x, tf.reduce_max(x, axis=1, keepdims=True)))
        
        # Normalize the output
        return tf.divide(exp_x, tf.reduce_sum(exp_x, axis=1, keepdims=True))


    ### Public methods ###

    def derivative(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the softmax activation function.
        
        Parameters:
        - x (tf.Tensor): Input to the activation function
        
        Returns:
        - tf.Tensor: Derivative of the activation function
        """
        
        # Compute the derivative of the softmax
        softmax_x = self(x)
        
        # Compute the Jacobian matrix
        return tf.multiply(softmax_x, tf.subtract(1, softmax_x))


class Tanh(Activation):
        
    ### Magic methods ###
    
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the output of the hyperbolic tangent activation function.
        
        Parameters:
        - x (tf.Tensor): Input to the activation function
        
        Returns:
        - tf.Tensor: Output of the activation function
        """
        
        # Compute the hyperbolic tangent
        return tf.tanh(x)


    ### Public methods ###

    def derivative(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the derivative of the hyperbolic tangent activation function.
        
        Parameters:
        - x (tf.Tensor): Input to the activation function
        
        Returns:
        - tf.Tensor: Derivative of the activation function
        """
        
        # Compute the derivative of the hyperbolic tangent
        return tf.subtract(1, tf.square(tf.tanh(x)))