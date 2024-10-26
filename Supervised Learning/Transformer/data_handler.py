import torch
from typing import Literal

from utils import device


class DataHandler:
    
    def __init__(self, data: torch.Tensor, train_val_split: float) -> None:
        # Store the data
        self.data = data
        
        # Compute the split index
        n = int(train_val_split * len(data))

        # Split the data into training and validation sets
        self.train_data = data[:n]
        self.val_data = data[n:]
    
    
    def get_batch(self, split: Literal["train", "validation"] = "validation", batch_size: int = 4, block_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        # Select the data based on the split
        data = self.train_data if split == "train" else self.val_data
        
        # Randomly select the starting index for the sequence
        ix = torch.randint(len(data) - block_size, (batch_size,)) # (batch_size,)
        
        # Create the input and target sequences by selecting a block of text from the data
        # The target sequence is the input sequence shifted by one character
        x = torch.stack([data[i:i+block_size] for i in ix]) # (batch_size, block_size)
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # (batch_size, block_size)
        
        # Move the data to the appropriate device
        x, y = x.to(device), y.to(device)
        
        # Return the input and target sequences
        return x, y