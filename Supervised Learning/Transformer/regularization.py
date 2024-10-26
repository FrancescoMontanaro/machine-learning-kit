import torch


class LayerNormalization:
    
    def __init__(self, dim: int, epsilon: float = 1e-5) -> None:
        # Initialize the epsilon parameter
        self.epsilon = epsilon
        
        # Initialize the gamma and beta parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
     
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the mean and variance
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        
        # Normalize the input
        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)
        
        # Apply the gamma and beta parameters
        self.out = self.gamma * x_norm + self.beta
        
        return self.out
    
    
    def parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Return the gamma and beta parameters
        return self.gamma, self.beta