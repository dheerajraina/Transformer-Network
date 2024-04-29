from typing import Any
import tensorflow as tf
import keras
from transformer_block import TransformerBlock


class Encoder(tf.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = keras.layers.Embedding(
            src_vocab_size, embed_size)
        self.position_embedding = keras.layers.Embedding(
            max_length, embed_size)

        self.layers = [
            TransformerBlock(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion
            )
            for _ in range(num_layers)
        ]
        self.dropout = keras.layers.Dropout(dropout)

    def __call__(self, x, mask):
        N, seq_length = x.shape
        positions = tf.range(0, seq_length)
        positions = tf.broadcast_to(positions, [N, seq_length])

        out = self.dropout(self.word_embedding(
            x)+self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, out)

        return out
