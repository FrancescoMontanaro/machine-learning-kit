import torch


class LayerNormalization:
    
    ### Magic methods ###
    
    def __init__(self, dim: int, epsilon: float = 1e-5) -> None:
        """
        Initialize the layer normalization layer.
        
        Parameters:
        - dim (int): The dimensionality of the input tensor.
        - epsilon (float): The epsilon parameter for numerical stability.
        """
        
        # Initialize the epsilon parameter
        self.epsilon = epsilon
        
        # Initialize the gamma and beta parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
     
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input tensor.
        
        Parameters:
        - x (torch.Tensor): The input tensor.
        
        Returns:
        - torch.Tensor: The normalized tensor.
        """
        
        # Compute the mean and variance
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        
        # Normalize the input
        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)
        
        # Apply the gamma and beta parameters
        self.out = self.gamma * x_norm + self.beta
        
        return self.out
    
    
    ### Public methods ###
    
    def parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the gamma and beta parameters.
        
        Returns:
        - tuple[torch.Tensor, torch.Tensor]: The gamma and beta parameters.
        """
        
        # Return the gamma and beta parameters
        return self.gamma, self.beta