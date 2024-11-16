import numpy as np
import tensorflow as tf
from typing import Literal, Optional

from .base import Layer
from ..optimizers import Optimizer
from ..activations import Activation


class Conv2D(Layer):
    
    ### Magic methods ###
    
    def __init__(
        self, 
        num_filters: int, 
        kernel_size: tuple[int, int], 
        padding: Literal["valid", "same"] = "valid", 
        stride: tuple[int, int] = (1, 1), 
        activation: Optional[Activation] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Class constructor for Conv2D layer.
        
        Parameters:
        - num_filters (int): number of filters in the convolutional layer
        - kernel_size (tuple[int, int]): size of the kernel
        - padding (Union[int, Literal["valid", "same"]]): padding to be applied to the input data. If "valid", no padding is applied. If "same", padding is applied to the input data such that the output size is the same as the input size.
        - stride (tuple[int, int]): stride of the kernel. First element is the stride along the height and second element is the stride along the width.
        - activation (Optional[Activation]): activation function to be applied to the output of the layer
        - name (str): name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Check if the kernel size has a valid shape
        if len(kernel_size) != 2:
            raise ValueError("Kernel size must be a tuple of 2 integers.")
        
        # Initialize the parameters
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.activation = activation
        
        # Initializing the filters and bias
        self.filters = None
        self.bias = None
        
        # Initializing the input shape
        self.input_shape = None
    
        
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Function to compute the forward pass of the Conv2D layer.
        It initializes the filters if not initialized and computes the forward pass.
        
        Parameters:
        - x (tf.Tensor): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - tf.Tensor: output data. Shape: (Batch size, Height, Width, Number of filters)
        
        Raises:
        - ValueError: if the input shape is not a tuple of 4 integers
        """
        
        # Check if the input shape has a valid shape
        if len(x.shape) != 4:
            raise ValueError(f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {x.shape}")
        
        # Checking if the layer is initialized
        if not self.initialized:
            # Initialize the filters
            self.init_params(x)
            
        # Compute the forward pass
        return self.forward(x)


    ### Public methods ###
    
    def parameters(self) -> dict:
        """
        Function to return the parameters of the Conv2D layer.
        
        Returns:
        - dict: dictionary containing the filters of the Conv2D layer
        """
        
        return {
            "filters": self.filters,
            "bias": self.bias
        }
    
        
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Function to compute the forward pass of the Conv2D layer.
        
        Parameters:
        - x (tf.Tensor): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - tf.Tensor: output data. Shape: (Batch size, Height, Width, Number of filters)
        
        Raises:
        - AssertionError: if the filters are not initialized
        """
        
        assert isinstance(self.filters, tf.Tensor), "Filters are not initialized. Please call the layer with some input data to initialize the filters."
        assert isinstance(self.bias, tf.Tensor), "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        # Extract the required dimensions for a better interpretation
        stride_height, stride_width = self.stride # Stride of the kernel
        
        # Save the input for the backward pass
        self.x = x

        # Apply the convolution operation
        self.features_map = tf.nn.conv2d(
            input = x,
            filters = tf.transpose(self.filters, perm=[1, 2, 3, 0]),
            strides = [1, stride_height, stride_width, 1],
            padding = self.padding.upper()
        ) # Shape: (Batch size, Height, Width, Number of filters)
        
        # Add the bias
        self.features_map = tf.add(self.features_map, self.bias) # Shape: (Batch size, Height, Width, Number of filters)
        
        # Apply the activation function
        return self.features_map if self.activation is None else self.activation(self.features_map)
    
    
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
        
        # Assert that filters and bias are initialized
        assert isinstance(self.optimizer, Optimizer), "Optimizer is not set. Please set an optimizer before training the model."
        assert isinstance(self.input_shape, tf.TensorShape), "Input shape is not set. Please call the layer with some input data to set the input shape."
        assert isinstance(self.filters, tf.Tensor), "Filters are not initialized."
        assert isinstance(self.bias, tf.Tensor), "Bias is not initialized."
        
        # Apply the activation derivative if an activation function is set
        if self.activation is not None:
            # Multiply the incoming gradient by the activation derivative
            loss_gradient = tf.multiply(loss_gradient, tf.cast(self.activation.derivative(self.features_map), dtype=loss_gradient.dtype)) # dL/dO_i * dO_i/dX_i, where dO_i/dX_i is the activation derivative
        
        # Extract dimensions
        _, _, _, num_channels = self.input_shape
        _, _, _, num_filters = loss_gradient.shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        
        # Assert that the number of channels is an integer
        assert isinstance(num_channels, int), "Number of channels is not an integer."
        
        # Compute gradients with respect to the bias
        grad_bias = tf.reduce_sum(loss_gradient, axis=(0, 1, 2)) # dL/db_i
            
        # Extract patches for all pooling regions in one step
        patches = tf.image.extract_patches(
            images = self.x,
            sizes = [1, kernel_height, kernel_width, 1],
            strides = [1, stride_height, stride_width, 1],
            rates = [1, 1, 1, 1],
            padding = self.padding.upper()
        ) # Shape: (batch_size, output_height, output_width, kernel_height * kernel_width * num_channels)
        
        # Reshape patches to isolate each channel and pooling window
        patches_reshaped = tf.reshape(patches, [-1, kernel_height * kernel_width * num_channels]) # Shape: (batch_size * output_height * output_width, kernel_height * kernel_width * num_channels)
        
        # Reshape the loss gradient to compute the gradient with respect to the filters
        loss_gradient_reshaped = tf.reshape(loss_gradient, [-1, num_filters]) # Shape: (batch_size * output_height * output_width, num_channels)
        
        # Compute the gradient with respect to the filters
        grad_filters = tf.matmul(patches_reshaped, loss_gradient_reshaped, transpose_a=True) # Shape: (kernel_height * kernel_width * num_channels, num_filters)
        grad_filters = tf.reshape(grad_filters, [kernel_height, kernel_width, num_channels, num_filters]) # Shape: (kernel_height, kernel_width, num_channels, num_filters)
        
        # Apply the convolution transpose operation to compute the gradient with respect to the input
        grad_input = tf.nn.conv2d_transpose(
            input = loss_gradient,
            filters = tf.transpose(self.filters, perm=[1, 2, 3, 0]),
            output_shape = self.input_shape,
            strides = [1, stride_height, stride_width, 1],
            padding = self.padding.upper()
        )
    
        # Update the filters
        self.filters = self.optimizer.update(
            layer = self,
            param_name = "filters",
            params = self.filters,
            grad_params = tf.transpose(grad_filters, perm=[3, 0, 1, 2])
        )
        
        # Update bias
        self.bias = self.optimizer.update(
            layer = self,
            param_name = "bias",
            params = self.bias,
            grad_params = grad_bias
        )
        
        return grad_input # dL/dX_i ≡ dL/dO_{i-1}
    
    
    def output_shape(self) -> tf.TensorShape:
        """
        Function to compute the output shape of the Conv2D layer.
        
        Returns:
        - tf.TensorShape: output shape of the Conv2D layer
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Assert that the input shape is set
        assert isinstance(self.input_shape, tf.TensorShape), "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Extract the dimensions
        batch_size, input_height, input_width, _ = self.input_shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        
        # Padding is applied to the input data
        if self.padding == 'same':
            # Compute the output shape of the Conv2D layer            
            output_height = tf.math.ceil(tf.divide(input_height, stride_height))
            output_width = tf.math.ceil(tf.divide(input_width, stride_width))
            
        # No padding is applied to the input data
        else:
            # Compute the output shape of the Conv2D layer
            output_height = tf.floor(tf.add(tf.divide(tf.subtract(input_height, kernel_height), stride_height), 1))
            output_width = tf.floor(tf.add(tf.divide(tf.subtract(input_width, kernel_width), stride_width), 1))
        
        # Compute the output shape of the Conv2D layer
        return tf.TensorShape((
            batch_size, # Batch size
            tf.cast(output_height, dtype=tf.int32), # Output height
            tf.cast(output_width, dtype=tf.int32),  # Output width
            self.num_filters # Number of filters
        ))
    
    
    def count_params(self) -> int:
        """
        Function to count the number of parameters in the Conv2D layer.
        
        Returns:
        - int: number of parameters in the Conv2D layer
        
        Raises:
        - AssertionError: if the filters or bias are not initialized
        """
        
        # Assert that the filters are initialized
        assert isinstance(self.filters, tf.Tensor), "Filters are not initialized. Please call the layer with some input data to initialize the filters."
        assert isinstance(self.bias, tf.Tensor), "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        return int(tf.add(np.size(self.filters), np.size(self.bias))) # num_filters * kernel_size[0] * kernel_size[1] * num_channels + num_filters
    
    
    def init_params(self, x: tf.Tensor) -> None:
        """
        Function to initialize the filters of the Conv2D layer.
        
        Parameters:
        - x (tf.Tensor): input data. Shape: (Batch size, Height, Width, Channels)
        """
        
        # Save the input shape
        self.input_shape = x.shape
        
        # Extract the dimensions of the input data
        _, _, _, num_channels = x.shape
        
        # Extract the dimensions of the kernel
        kernel_height, kernel_width = self.kernel_size
        
        # Initializing the filters and bias with random values and normalizing them
        self.filters = tf.cast(tf.divide(tf.random.uniform((self.num_filters, kernel_height, kernel_width, num_channels)), (kernel_height * kernel_width)), dtype=x.dtype) # Shape: (num_filters, kernel_height, kernel_width, num_channels)
        self.bias = tf.cast(tf.zeros(self.num_filters), dtype=x.dtype)
        
        # Update the initialization flag
        self.initialized = True