import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout  # type: ignore


class FeedForward(Layer):

    def __init__(self, n_embed: int, dropout: float = 0.1):
        """
        Initialize the feed-forward layers of the transformer.
        
        Parameters:
        - n_embed (int): The size of the embeddings.
        - dropout (float): The dropout rate.
        """
        
        # Initialize the parent class
        super(FeedForward, self).__init__()

        # Define the feed-forward layers
        self.dense1 = Dense(4 * n_embed, activation='relu')  # First dense layer with ReLU activation
        self.dense2 = Dense(n_embed)  # Second dense layer
        self.dropout = Dropout(dropout)  # Dropout for regularization


    def call(self, embeddings: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass of the feed-forward layers.
        
        Parameters:
        - embeddings (tf.Tensor): The input embeddings.
        - training (bool): Whether the layer is in training mode (applies dropout only in training).
        
        Returns:
        - tf.Tensor: The output embeddings.
        """
        
        x = self.dense1(embeddings)  # Apply the first dense layer
        x = self.dense2(x)  # Apply the second dense layer
        x = self.dropout(x, training=training)  # Apply dropout if training
        
        return x