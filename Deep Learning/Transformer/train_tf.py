import os
import tensorflow as tf

from lib.utils import load_txt_file

# Constants
dataset_path = os.path.join(os.getcwd(), 'dataset', 'input.txt')

# Hyperparameters
train_val_split = 0.9 # 90% of the data will be used for training, 10% for validation
batch_size = 64 # The number of samples to use for each batch
block_size = 256 # The size of the sequence length (the context window)
learning_rate = 3e-4 # The learning rate for the optimizer
epochs = 500 # The number of epochs to train the model for
n_embed = 384 # The size of the token embeddings (the dimensionality of the embeddings)
eval_iters = 200 # The number of iterations to evaluate the model
num_attention_heads = 6 # The number of attention heads in the multi-head attention mechanism
num_transformer_blocks = 6 # The number of transformer blocks in the model
dropout = 0.2 # The dropout rate

# Set the random seed for reproducibility
tf.random.set_seed(1337)