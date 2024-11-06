import numpy as np
from typing import Literal, Optional

from .base import Layer


class Conv2D(Layer):
    
    ### Magic methods ###
    
    def __init__(self, num_filters: int, kernel_size: tuple[int, int], padding: Literal["valid", "same"] = "valid", stride: tuple[int, int] = (1, 1), name: Optional[str] = None) -> None:
        """
        Class constructor for Conv2D layer.
        
        Parameters:
        - num_filters (int): number of filters in the convolutional layer
        - kernel_size (tuple[int, int]): size of the kernel
        - padding (Union[int, Literal["valid", "same"]]): padding to be applied to the input data. If "valid", no padding is applied. If "same", padding is applied to the input data such that the output size is the same as the input size.
        - stride (tuple[int, int]): stride of the kernel. First element is the stride along the height and second element is the stride along the width.
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
        
        # Initializing the filters and bias
        self.filters = None
        self.bias = None
        
        # Initializing the input shape
        self.input_shape = None
    
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Function to compute the forward pass of the Conv2D layer.
        It initializes the filters if not initialized and computes the forward pass.
        
        Parameters:
        - x (np.ndarray): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - np.ndarray: output data. Shape: (Batch size, Height, Width, Number of filters)
        
        Raises:
        - ValueError: if the input shape is not a tuple of 4 integers
        - ValueError: if the number of channels in the input data is not the same as the number of channels in the filters
        """
        
        # Check if the input shape has a valid shape
        if len(x.shape) != 4:
            raise ValueError(f"Input must be a 4D array. The shape must be (Batch size, Height, Width, Channels). Got shape: {x.shape}")
        
        # Extract the dimensions of the input data
        batch_size, input_height, input_width, num_channels = x.shape
        
        # Save the input shape
        self.input_shape = (batch_size, input_height, input_width, num_channels)
        
        # Checking if the layer is initialized
        if not self.initialized:
            # Initialize the filters
            self.init_params(num_channels)
            
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
            "filters": self.filters
        }
    
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Function to compute the forward pass of the Conv2D layer.
        
        Parameters:
        - x (np.ndarray): input data. Shape: (Batch size, Height, Width, Channels)
        
        Returns:
        - np.ndarray: output data. Shape: (Batch size, Height, Width, Number of filters)
        
        Raises:
        - AssertionError: if the filters are not initialized
        """
        
        # Assert that the filters are initialized
        assert self.filters is not None, "Filters are not initialized. Please call the layer with some input data to initialize the filters."
        assert self.bias is not None, "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        # Extract the required dimensions for a better interpretation
        input_batch_size, input_height, input_width, _ = x.shape # Shape of the input data
        output_height, output_width, num_filters = self.output_shape() # Shape of the output data
        kernel_height, kernel_width = self.kernel_size # Shape of the kernel
        stride_height, stride_width = self.stride # Stride of the kernel
        
        # Initialize the output data
        output = np.zeros((input_batch_size, output_height, output_width, num_filters))
        
        # Iterate over the height of the input data, the step size is the stride along the height
        for i in range(0, input_height - kernel_height + 1, stride_height):
            
            # Iterate over the width of the input data, the step size is the stride along the width
            for j in range(0, input_width - kernel_width + 1, stride_width):
                
                # Extract the windows of the input data
                window = x[:, i:i+kernel_height, j:j+kernel_width, :]
                
                # Iterate over the filters
                for k in range(num_filters):
                    # Compute the convolution
                    output[:, i//stride_height, j//stride_width, k] = np.sum(window * self.filters[k]) + self.bias[k]
        
        return output
    
    
    def output_shape(self) -> tuple:
        """
        Function to compute the output shape of the Conv2D layer.
        
        Returns:
        - tuple: shape of the output data
        
        Raises:
        - ValueError: if the input shape is not set
        
        Raises:
        - AssertionError: if the input shape is not set
        """
        
        # Assert that the input shape is set
        assert self.input_shape is not None, "Input shape is not set. Please call the layer with some input data to set the input shape."
        
        # Extract the dimensions
        _, input_height, input_width, _ = self.input_shape
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        
        # Compute the padding value
        padding_value_height = (kernel_height - 1) // 2 if self.padding == "same" else 0
        padding_value_width = (kernel_width - 1) // 2 if self.padding == "same" else 0
        
        # Compute the output shape of the Conv2D layer
        return (
            (input_height + 2 * padding_value_height - kernel_height) // stride_height + 1, # Height
            (input_width + 2 * padding_value_width - kernel_width) // stride_width + 1, # Width
            self.num_filters # Number of filters
        )
    
    
    def count_params(self) -> int:
        """
        Function to count the number of parameters in the Conv2D layer.
        
        Returns:
        - int: number of parameters in the Conv2D layer
        
        Raises:
        - AssertionError: if the filters or bias are not initialized
        """
        
        # Assert that the filters are initialized
        assert self.filters is not None, "Filters are not initialized. Please call the layer with some input data to initialize the filters."
        assert self.bias is not None, "Bias is not initialized. Please call the layer with some input data to initialize the bias."
        
        return int(np.prod(self.filters.shape)) + int(np.prod(self.bias.shape)) # num_filters * kernel_size[0] * kernel_size[1] * num_channels + num_filters
    
    
    def init_params(self, num_channels: int) -> None:
        """
        Function to initialize the filters of the Conv2D layer.
        
        Parameters:
        - num_channels (int): number of channels in the input data
        """
        
        # Extract the dimensions of the kernel
        kernel_height, kernel_width = self.kernel_size
        
        # Initializing the filters and bias with random values and normalizing them
        self.filters = np.random.randn(self.num_filters, kernel_height, kernel_width, num_channels) / (kernel_height * kernel_width)
        self.bias = np.zeros(self.num_filters)