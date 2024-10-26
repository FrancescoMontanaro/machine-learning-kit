import torch
import torch.nn as nn


class FeedForward(nn.Module):
    
    def __init__(self, n_embed: int, dropout: float = 0.1) -> None:
        # Initialize the superclass
        super().__init__()
        
        # Define the feed-forward layers
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # Project the embeddings back to the original size
            nn.Dropout(dropout) # Apply dropout for regularization
        )
        
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Apply the feed-forward layers
        return self.feed_forward(embeddings)