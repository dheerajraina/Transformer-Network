import tensorflow as tf
import keras


class SelfAttention(tf.Module):  # Scaled Dot Product Attention
    def __init__(self, embed_size, heads):
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
        self.fc_out = keras.layers.Dense(
            embed_size, input_shape=(heads*self.head_dim,))

    def __call__(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = tf.reshape(values, (N, value_len, self.heads, self.head_dim))
        keys = tf.reshape(values, (N, key_len, self.heads, self.head_dim))
        queries = tf.reshape(query, (N, query_len, self.heads, self.head_dim))

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        e = tf.einsum("nqhd,nkhd->nhqk", queries, keys)  # matmul

        if mask is not None:
            mask = tf.equal(e, 0)
            e = tf.where(mask, float("-1e20"), e)
        attention = tf.nn.softmax(e/(self.embed_size**(1/2)), axis=3)

        out = tf.reshape(tf.einsum("nhql,nlhd->nqhd", attention,
                                   values), (N, query_len, self.heads*self.head_dim))  # matmul + concat

        out = self.fc_out(out)
        return out
