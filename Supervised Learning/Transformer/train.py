import os
import torch

from data_loader import DataLoader
from transformer import Transformer
from utils import load_txt_file, device

# Constants
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'input.txt')

# Hyperparameters
train_val_split = 0.9 # 90% of the data will be used for training, 10% for validation
batch_size = 64 # The number of samples to use for each batch
block_size = 256 # The size of the sequence length (the context window)
learning_rate = 3e-4 # The learning rate for the optimizer
epochs = 10000 # The number of epochs to train the model for
n_embed = 384 # The size of the token embeddings (the dimensionality of the embeddings)
eval_iters = 200 # The number of iterations to evaluate the model
num_attention_heads = 6 # The number of attention heads in the multi-head attention mechanism
num_transformer_blocks = 6 # The number of transformer blocks in the model
dropout = 0.2 # The dropout rate

# Set the random seed for reproducibility
torch.manual_seed(1337)

# Load the text file
text = load_txt_file(dataset_path)

# Extract the unique characters (vocabulary)
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# Creating a simple mapping from characters to integers
char_to_int = {c: i for i, c in enumerate(vocab)}
int_to_char = {i: c for i, c in enumerate(vocab)}

# Creating the encoding and decoding functions
encode = lambda text: [char_to_int[c] for c in text]
decode = lambda tokens: ''.join([int_to_char[t] for t in tokens])

# Convert the data to a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Create the data handler
data_handler = DataLoader(
    data = data, 
    train_val_split = train_val_split
)

# Create the language model
language_model = Transformer(
    vocab_size = vocab_size,
    n_embed = n_embed,
    n_heads = num_attention_heads,
    block_size = block_size,
    n_transformer_blocks = num_transformer_blocks,
    dropout = dropout
)

# Train the model
language_model.train_model(
    data_loader = data_handler,
    epochs = epochs, 
    lr = learning_rate, 
    batch_size = batch_size,
    eval_iters = eval_iters
)

# Generate some text from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(language_model.generate(context, max_new_tokens=100).squeeze().tolist()))