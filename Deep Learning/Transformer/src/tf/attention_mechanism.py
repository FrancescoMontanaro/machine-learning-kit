import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer # type: ignore


class SingleHeadAttention(Layer):
    
    ### Magic methods ###

    def __init__(self, head_size: int, n_embed: int, block_size: int, dropout: float = 0.1, **kwargs):
        """
        Initialize the single-head attention mechanism.

        Parameters:
        - head_size (int): The size of the attention head.
        - n_embed (int): The size of the embeddings.
        - block_size (int): The size of the block.
        - dropout (float): The dropout rate.
        """
        
        # Initialize the parent class
        super().__init__(**kwargs)
        
        # Store the parameters
        self.num_embed = n_embed
        self.head_size = head_size
        self.block_size = block_size

        # Define the key, query, and value dense layers
        self.key = Dense(head_size, use_bias=False)
        self.query = Dense(head_size, use_bias=False)
        self.value = Dense(head_size, use_bias=False)

        # Define dropout
        self.dropout = Dropout(dropout)

        # Triangular attention mask
        self.attention_mask = tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0)
        
    
    ### Public methods ###

    def call(self, embeddings: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass of the single-head attention mechanism.

        Parameters:
        - embeddings (tf.Tensor): The input embeddings.
        - training (bool): Whether the model is in training mode (for dropout).

        Returns:
        - tf.Tensor: The contextual embeddings.
        """
        # Extract the shape of the embeddings
        B, T, C = embeddings.shape  # B: batch size, T: sequence length, C: number of channels (n_embed)

        # Compute key, query, and value
        k = self.key(embeddings)  # (B, T, H)
        q = self.query(embeddings)  # (B, T, H)
        v = self.value(embeddings)  # (B, T, H)

        # Compute scaled dot-product attention
        attention_weights = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_size, tf.float32))  # (B, T, T)

        # Apply the mask
        attention_weights = tf.where(
            self.attention_mask[:T, :T] == 1,
            attention_weights,
            tf.constant(float('-inf'))
        )

        # Apply softmax
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)  # (B, T, T)

        # Apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights, training=training)

        # Compute the contextual embeddings
        contextual_embeddings = tf.matmul(attention_weights, v)  # (B, T, H)

        return contextual_embeddings


class MultiHeadAttention(Layer):
    
    ### Magic methods ###

    def __init__(self, n_heads: int, head_size: int, n_embed: int, block_size: int, dropout: float = 0.1, **kwargs):
        """
        Initialize the multi-head attention mechanism.

        Parameters:
        - n_heads (int): The number of attention heads.
        - head_size (int): The size of each attention head.
        - n_embed (int): The size of the embeddings.
        - block_size (int): The size of the block.
        - dropout (float): The dropout rate.
        """
        
        # Initialize the parent class
        super().__init__(**kwargs)
        
        # Store the parameters
        self.n_heads = n_heads
        
        # Create the attention heads
        self.heads = [SingleHeadAttention(head_size, n_embed, block_size, dropout) for _ in range(n_heads)]
        
        # Define the output linear layer
        self.output_linear = Dense(n_embed) # Project the embeddings back to the original size
        
        # Define dropout
        self.dropout = Dropout(dropout)


    ### Public methods ###

    def call(self, embeddings: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass of the multi-head attention mechanism.

        Parameters:
        - embeddings (tf.Tensor): The input embeddings.
        - training (bool): Whether the model is in training mode (for dropout).

        Returns:
        - tf.Tensor: The output embeddings.
        """
        
        # Apply each head to the embeddings and concatenate the results
        head_outputs = [head(embeddings, training=training) for head in self.heads]
        concatenated = tf.concat(head_outputs, axis=-1)  # (B, T, H * n_heads)

        # Apply the output linear layer
        output = self.dropout(self.output_linear(concatenated), training=training)  # (B, T, C)

        return output