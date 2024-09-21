import tensorflow as tf
from utils import positional_encoding

class Transformer(tf.keras.Model):
    """
    This is a class for create Transformer model.
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        """
        The constructor for Transformer class.

        Parameters:
           num_layers (int): number of stacked encoder  and decoder.
           d_model (int): representation dimension.
           num_heads (int): number of heads in self-attention of each encoder layer or each decoder layer.
           dff (int): neuron size of point wise feed forward network of each encoder layer or each decoder layer.
           input_vocab_size (int): input vocabulary size.
           target_vocab_size (int): target vocabulary size.
           pe_input (int): input position encoding maximum length of encoder
           pe_target (int): target position encoding maximum length of decoder
           rate (float): The imaginary part of complex number.
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        The function to execute operation from encoder to decoder for
        inp > N encoder > N decoder > enc_output
        tar, enc_output > N decoder > final layer(softmax)

        Parameters:
            inp (ComplexNumber): input for encoder, shape=(batch_size, input_seq_len, d_model)
            tar (ComplexNumber): target input for decoder, shape=(batch_size, target_seq_len, d_model)
            training (bool): is training or not
            enc_padding_mask (tensor): mask tensor of encoder, shape=(..., seq_len_q, seq_len_k)
            dec_padding_mask (tensor): mask tensor of decoder, shape=(..., seq_len_q, seq_len_k)
            look_ahead_mask (tensor):  block future token mask tensor only used for decoder,
                                      shape=(..., target_seq_len, target_seq_len)

        Returns:
            ComplexNumber: A complex number which contains the sum.
        """
        enc_output = self.encoder(inp, training, enc_padding_mask) # (batch_size, input_seq_len, d_model)

        # dec_output: (batch_size, tar_seq_len, d_model)
        # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


class Encoder(tf.keras.layers.Layer):
    """
    This is a class for stacked Encoder include N Encoder layer
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        """
        The constructor for Encoder class.

        Parameters:
           num_layers (int): number of stacked encoder layer
           d_model (int): token embedding size
           num_heads (int): the number of heads
           dff (int): the number of units of feed forward hidden layer
           input_vocab_size: vocabulary size
           maximum_position_encoding: maximum length of sequence
           rate (float):  dropout rate
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """
        The function to execute operation of stacked encoder for
        x > embedding layer > add positional encoding > dropout layer
        > 1th encoder layer > 2th encoder layer > ... > N-th encoder layer

        Parameters:
            x (tensor): shape=(batch_size, input_seq_len, d_model)
            training (bool): is training or not
            mask (tensor): mask tensor, shape=(..., seq_len_q, seq_len_k)
                    mask is multiplied by -1e9 (close to negative infinity).
                    this is useful for multihead attention.
        Returns:
            output tensor from one encoder layer: shape=(batch_size, input_seq_len, d_model)
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x # (batch_size, input_seq_len, d_model)



class Decoder(tf.keras.layers.Layer):
    """
    This is a class for stacked Decoder include N Decoder layer
    """

    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, maximum_position_encoding, rate=0.1):
        """
        The constructor for Decoder class.

        Parameters:
           num_layers (int): number of stacked encoder layer
           d_model (int): token embedding size
           num_heads (int): the number of heads
           dff (int): the number of units of feed forward hidden layer
           target_vocab_size (int): target vocabulary size
           maximum_position_encoding (int): maximum length of sequence
           rate (float):  dropout rate
        """
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        The function to execute operation of stacked decoder for
        x > embedding layer > add positional encoding > dropout layer
        > 1th decoder layer > 2th decoder layer > ... > N-th decoder layer

        Parameters:
            x (tensor): shape=(batch_size, target_seq_len, d_model)
            training (bool): is training or not
            look_ahead_mask (tensor): block future token mask tensor only used for decoder,
                                      shape=(..., target_seq_len, target_seq_len)
            padding_mask (tensor): shape = (batch_size, 1, 1, target_seq_len)

        Returns:
            output tensor from one decoder layer: shape=(batch_size, target_seq_len, d_model)
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    """
    This is a class for operations of multi-head attention and
    point wise feed forward networks.
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        The constructor for EncoderLayer class.

        Parameters:
           d_model (int): token embedding size
           num_heads (int): the number of heads
           dff (int): the number of units of feed forward hidden layer
           rate (float):  dropout rate
        """
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """
        The function to execute operation of one encoder layer for
            multi-head attention > dropout > add & layernorm
            > feed-forward network > dropout > add & layernorm

        Parameters:
            x (tensor): shape=(batch_size, seq_len, d_model)
            training (bool): is training or not
            mask (tensor): shape = (batch_size, 1, 1, input_seq_len)

        Returns:
            output tensor from one encoder layer: shape=(batch_size, seq_len, d_model)
        """
        attn_output, _ = self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """
    This is a class for operations of masked multi-head attention and
    encoder-decoder multi-head attention and
    point wise feed forward networks.
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        The constructor for EncoderLayer class.

        Parameters:
           d_model (int): token embedding size
           num_heads (int): the number of heads
           dff (int): the number of units of feed forward hidden layer
           rate (float):  dropout rate
        """
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        The function to execute operation of one decoder layer for
        masked multi-head attention > dropout > add & layernorm
        > encoder-decoder multi-head attention > dropout > add & layernorm
        > feed-forward network > dropout > add & layernorm

        Parameters:
            x (tensor): decoder input, shape=(batch_size, target_seq_len, d_model)
            enc_output (tensor): encoder output, shape=(batch_size, input_seq_len, d_model)
            look_ahead_mask (tensor): block future token mask tensor only used for decoder,
                                      shape=(..., target_seq_len, target_seq_len)
            padding_mask (tensor): shape = (batch_size, 1, 1, target_seq_len)

        Returns:
            output tensor from one encoder layer: shape=(batch_size, target_seq_len, d_model)
            attn_weights_block1 (tensor): shape = (batch_size, num_heads, seq_len_q, seq_len_k)
            attn_weights_block2 (tensor): shape = (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # attn1: shape = (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x) # (batch_size, target_seq_len, d_model)

        # attn2: shape = (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1) # (batch_size, target_seq_len, d_model)

        # ffn_output: shape = (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2) # (batch_size, target_seq_len, d_model)
        return out3, attn_weights_block1, attn_weights_block2


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calulate self attention tensor.

    Extended description of function.

    Parameters:
        q (tensor): query tensor, shape=(..., seq_len_q, depth)
        k (tensor): key tensor, shape=(..., seq_len_k, depth)
        v (tensor): value tensor, shape=(..., seq_len_v, depth_v)
        mask (tensor): mask tensor, shape=(..., seq_len_q, seq_len_k)
                    mask is multiplied by -1e9 (close to negative infinity).
                    this is useful for multihead attention.

    Returns:
        output: attention tensor, shape=(..., seq_len_q, depth_v)
        attention_weights: attention weights tensor, shape=(..., seq_len_q, seq_len_k)
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add mask tensor to scaled attention logits
    # scaled_attention_logits: shape=(..., seq_len, seq_len)
    # mask: shape=(..., seq_len, seq_len)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax at last index(seq_len_k)
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    This is a class for multi head attention.
    """

    def __init__(self, d_model, num_heads):
        """
        The constructor for MultiHeadAttention class.

        Parameters:
           d_model (int): token embedding size.
           num_heads (int): number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        The function to reshape x applied for multi head.
        Turn shape from  (batch_size, seq_len, d_model) to (batch_size, seq_len, num_heads, depth)
        and then reshape to (batch_size, num_heads, seq_len, depth)

        Parameters:
            x (tensor): a tentor, shape= (batch_size, seq_len, d_model)
            batch_size(int): batch size

        Returns:
            reshaped x(tensor): a tentor, shape= (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
        The built-in function for tensorflow to calculate result.

        Parameters:
            v (tensor): shape=(batch_size, seq_len, d_model)
            v (tensor): shape=(batch_size, seq_len, d_model)
            v (tensor): shape=(batch_size, seq_len, d_model)
            mask (tensor): shape=(..., seq_len_q, seq_len_k)

        Returns:
            output: multi head attention output, shape=(batch_size, seq_len_q, d_model)
            attention_weights: multi head attention weights, shape=(batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """
    Point wise feed forward network.

    Parameters:
        d_model (int): token embedding size
        dff (int): feed forward network the number of units

    Returns:
    Sequential model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model) # (batch_size, seq_len, d_model)
    ])

# sample_transformer = Transformer(
#     num_layers=2, d_model=512, num_heads=8, dff=2048,
#     input_vocab_size=8500, target_vocab_size=8000,
#     pe_input=10000, pe_target=6000)
#
# temp_input = tf.random.uniform((64, 62))
# temp_target = tf.random.uniform((64, 26))
#
# fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
#                                enc_padding_mask=None,
#                                look_ahead_mask=None,
#                                dec_padding_mask=None)

# print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
