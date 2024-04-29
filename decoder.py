import tensorflow as tf
import keras
from decoder_block import DecoderBlock


class Decoder(tf.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 max_length
                 ):
        super(Decoder, self).__init__()
        self.word_embedding = keras.layers.Embedding(
            trg_vocab_size, embed_size)
        self.position_embedding = keras.layers.Embedding(
            max_length, embed_size)
        self.layers = [
            DecoderBlock(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ]

        self.fc_out = keras.layers.Dense(
            trg_vocab_size, input_shape=(trg_vocab_size,),)
        self.dropout = keras.layers.Dropout(dropout)

    def __call__(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = tf.range(0, seq_length)
        positions = tf.broadcast_to(positions, [N, seq_length])
        x = self.dropout((self.word_embedding(x)+self.position_embedding(x)))

        for layer in self.layers:
            print(layer)
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)

        return out
