import tensorflow as tf
import keras


class SelfAttention(tf.Module):# Scaled Dot Product Attention
    def __init__(self, embed_size, heads) -> None:
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim*heads ==
                embed_size), "Embed size needs to be divided by heads"

        self.values = keras.layers.Dense(
            self.head_dim, input_shape=(self.head_dim,), use_bias=False)
        self.keys = keras.layers.Dense(
            self.head_dim, input_shape=(self.head_dim,), use_bias=False)
        self.queries = keras.layers.Dense(
            self.head_dim, input_shape=(self.head_dim,), use_bias=False)
        self.fc_out = keras.layers.Dense(embed_size,input_shape=(heads*self.head_dim,))

    def call(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = values.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = tf.einsum("nqhd,nkhd->nhqk", queries, keys)

        if mask is not None:
            energy = tf.where(0, float("-1e20"), energy)
        attention = tf.nn.softmax(energy/(self.embed_size**(1/2)), axis=3)

        out = tf.einsum("nhql,nlhd->nqhd", attention,
                        values).reshape(N, query_len, self.heads*self.head_dim)

        out = self.fc_out(out)
        return out
