import tensorflow as tf
import keras
from self_attention import SelfAttention
from transformer_block import TransformerBlock


class DecoderBlock(tf.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = keras.layers.LayerNormalization()
        self.transformer_block = TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion
        )
        self.dropout = keras.layers.Dropout(dropout)

    def __call__(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
