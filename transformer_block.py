import tensorflow as tf
import keras
from self_attention import SelfAttention


class TransformerBlock(tf.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):

        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()

        self.feed_forward = keras.models.Sequential([
            keras.layers.Dense(forward_expansion*embed_size, activation="relu"),
            keras.layers.Dense(embed_size),
        ])
        self.dropout = keras.layers.Dropout(dropout)

    def __call__(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out
