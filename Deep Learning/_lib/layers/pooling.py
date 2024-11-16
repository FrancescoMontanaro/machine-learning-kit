import tensorflow as tf
from typing import Optional, Literal

from .base import Layer

        
class MaxPool2D(Layer):
    
    ### Magic methods ###
    
    def __init__(self, size: tuple[int, int] = (2, 2), stride: Optional[tuple[int, int]] = None, padding: Literal["valid", "same"] = "valid", name: Optional[str] = None) -> None:
        """
        Class constructor for the MaxPool2D layer
        
        Parameters:
        - size (tuple): Size of the pooling window
        - stride (tuple): Stride of the pooling window
        - name (str): Name of the layer
        """
        
        # Initialize the parent class
        super().__init__(name)
        
        # Save the pooling window size, stride and padding
        self.size = size
        self.stride = stride if stride is not None else size
        self.padding = padding
        
        
    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """
        Method to perform the forward pass through the MaxPool2D layer
        
        Parameters:
        - x (tf.Tensor): Input data. Shape: (Batch size, height, width, channels)
        
        Returns:
        - tf.Tensor: Output of the layer after the forward pass
        """
        
        # Check if the input shape has a valid shape
        if len(x.shape) != 4:
            raise ValueError(f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {(len(x.shape))}")
        
        # Initialize the layer params
        if not self.initialized:
            self.init_params(x)
        
        # Perform the forward pass
        return self.forward(x)
        
        
    ### Public methods ###
    
    def forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Method to perform the forward pass through the MaxPool2D layer
        
        Parameters:
        - x (tf.Tensor): Input data. Shape: (Batch size, height, width, channels)
        
        Returns:
        - tf.Tensor: Output of the layer after the forward pass
        """

        # Apply max pooling
        output, self.arg_max = tf.nn.max_pool_with_argmax(
            input = x,
            ksize = [1, *self.size, 1],
            strides = [1, *self.stride, 1],
            padding = self.padding.upper()
        )
        
        return output
    
    
    def backward(self, loss_gradient: tf.Tensor) -> tf.Tensor:
        """
        Backward pass for the MaxPool2D layer (Layer i)
        
        Parameters:
        - loss_gradient (tf.Tensor): Gradient of the loss with respect to the output of this layer: dL/dO_i
        
        Returns:
        - tf.Tensor: Gradient of the loss with respect to the input of this layer: dL/dX_i ≡ dL/dO_{i-1}
        """
        
        # Extract shapes
        batch_size, input_height, input_width, n_channels = self.input_shape
        
        # Asserting the types of the input arguments
        assert isinstance(batch_size, int)
        assert isinstance(input_height, int)
        assert isinstance(input_width, int)
        assert isinstance(n_channels, int)
        
        # Total number of elements per input sample
        total_number = input_height * input_width * n_channels
        
        # Flatten arg_max and loss_gradient
        arg_max_flat = tf.reshape(self.arg_max, [-1])
        loss_gradient_flat = tf.reshape(loss_gradient, [-1])
        
        # Compute batch offsets to adjust indices
        batch_offset = tf.range(batch_size, dtype=arg_max_flat.dtype) * total_number
        batch_offset = tf.reshape(batch_offset, [-1, 1, 1, 1])
        batch_offset = tf.broadcast_to(batch_offset, tf.shape(self.arg_max))
        batch_offset_flat = tf.reshape(batch_offset, [-1])
        
        # Adjust arg_max indices to account for batch offset
        arg_max_flat_adjusted = arg_max_flat + batch_offset_flat
        
        # Total number of elements in all batches
        total_elements = batch_size * total_number
        
        # Scatter the gradients back to the input tensor
        d_input_flat = tf.scatter_nd(
            indices = tf.expand_dims(arg_max_flat_adjusted, 1),
            updates = loss_gradient_flat,
            shape = [total_elements]
        )
        
        # Reshape back to input shape
        d_input = tf.reshape(d_input_flat, [batch_size, input_height, input_width, n_channels])

        return d_input

        
    def output_shape(self) -> tf.TensorShape:
        """
        Function to compute the output shape of the layer.
        
        Returns:
        - tf.TensorShape: Shape of the output data
        
        Raises:
        - AssertionError: if the input shape is not set or if the input shape is not an integer
        """
        
        # Assert that the input shape is set
        assert isinstance(self.input_shape, tf.TensorShape), "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Extract the dimensions
        batch_size, input_height, input_width, n_channels = self.input_shape
        pool_width, pool_height = self.size
        stride_height, stride_width = self.stride
        
        assert isinstance(input_height, int), "Input height must be an integer."
        assert isinstance(input_width, int), "Input width must be an integer."
        
        if self.padding.lower() == 'same':
            # Compute the output dimensions when padding is applied
            output_height = (input_height + stride_height - 1) // stride_height
            output_width = (input_width + stride_width - 1) // stride_width
        else:
            # Compute the output dimensions when padding is not applied
            output_height = (input_height - pool_height) // stride_height + 1
            output_width = (input_width - pool_width) // stride_width + 1
        
        # Return the output shape
        return tf.TensorShape((
            batch_size, # Batch size
            output_height, # Height
            output_width, # Width
            n_channels # Number of channels
        ))
        
        
    def init_params(self, x: tf.Tensor) -> None:
        """
        Method to initialize the parameters of the layer
        
        Parameters:
        - x (tf.Tensor): Input data
        """
        
        # Save the input shape
        self.input_shape = x.shape
        
        # Set the layer as initialized
        self.initialized = True