import tensorflow as tf
from encoder import Encoder
from decoder import Decoder


class TransformerNetwork(tf.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        max_length=100
    ):
        super(TransformerNetwork, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = tf.expand_dims(tf.expand_dims(
            tf.not_equal(src, self.src_pad_idx), axis=1), axis=2)
        return src_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # return the lower triangular part of the matrix
        tril_matrix = tf.linalg.band_part(tf.ones((trg_len, trg_len)), -1, 0)
        trg_mask = tf.broadcast_to(tf.expand_dims(
            tril_matrix, axis=0), [N, 1, trg_len, trg_len])  # adding new dimension to the matrix and then broadcasting it to desired shape

        return trg_mask

    def __call__(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out
