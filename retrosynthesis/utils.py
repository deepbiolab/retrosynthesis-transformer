import tensorflow as tf
import numpy as np


def create_masks(inp, tar):
    """
    Create masks for encoder parts and decoder parts.

    Parameters:
        inp (tensor): shape = (batch size, input sequence length)
        tar (tensor): shape = (batch size, target sequence length)


    Returns:
        enc_padding_mask: padding mask for encoder
        combined_mask: look-head mask for decoder used for decoder first attention
        dec_padding_mask: padding mask for decoder used for decoder second attention
    """
    enc_padding_mask = create_padding_mask(inp)

    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def create_padding_mask(seq):
    """
    Block all pad tokens in a batch of sequences.
    This ensures that the model does not take padding as input.

    Extended description of function.
        The mask indicates the position where the filling value 0 appears:
        the mask outputs 1 at these positions, otherwise it outputs 0.

    Parameters:
    seq (tensor): a sequence of a batch, shape=batch size * length of sequence
                  such as: [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]

    Returns:
    tensor: shape=(batch_size, 1, 1, seq_len)
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # Add additional dimensions useful for broadcasting to add padding to Attention (logits) later
    # (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Look-ahead mask is used to block future tokens in a sequence.

    Extended description of function.
        This means that to predict the third word, only the first and second words will be used.
        Similarly, predict the fourth word, use only the first, second, and third words, and so on.

    Parameters:
        size (int): sequence length

    Returns:
        tensor: shape= (seq_len, seq_len)

    """
    # tf.matrix_band_part(input, 0, -1) == > Upper triangular part.
    # array([[1., 1., 1.],
    #        [0., 1., 1.],
    #        [0., 0., 1.]]
    # tf.matrix_band_part(input, -1, 0) == > Lower triangular part.
    # array([[1., 0., 0.],
    #        [1., 1., 0.],
    #        [1., 1., 1.]]
    # tf.matrix_band_part(input, 0, 0) == > Diagonal.
    # array([[1., 0., 0.],
    #        [0., 1., 0.],
    #        [0., 0., 1.]]

    mask = 1 - tf .linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def positional_encoding(position, d_model):
    """
    Given sequence length and each token embedding size and
    calculate positional matrix for each sequence.

    Extended description of function.
        P_(pos, 2i)     = sin(pos / 10000^(2i/d_model)
        P_(pos, 2i+1)   = cos(pos / 10000^(2i/d_model)

    Parameters:
        position (int): Max Length of sequence, it's not equal to length of x and greater than length of x
        d_model (int): token embedding size
    Returns:
    float32 tensor: positional matrix,shape=position * d_model
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # for each pos, 2i, calulate sin function
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # Add extra dimension used for broadcast laterly
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)



def get_angles(pos, i, d_model):
    """
    Calculate angle value used for positional encoding later.
    Extended description of function.
        P_(pos, 2i)     = sin(pos / 10000^(2i/d_model)
        P_(pos, 2i+1)   = cos(pos / 10000^(2i/d_model)
    Parameters:
        pos (int)       : position-th of token at sequence
        i (int)         : i-th dimension of certain token
        d_model (int)   : token embedding size

    Returns:
    float: the angle value for given position and dim-th
    """
    angle_rates = 1 / np.power(10000, (2*(i//2) / np.float32(d_model)))
    return pos * angle_rates


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    This is a class for Learning Rate Schedule.
    """
    def __init__(self, d_model, warmup_steps=4000):
        """
        The constructor for CustomSchedule class.

        This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
        and decreasing it thereafter proportionally to the inverse square root of the step number

        Parameters:
           d_model (int): embedding size
           warmup_steps (int): warmup steps for increasing phrase
        """
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        The function to get learning rate for each step.

        Parameters:
            step (int): decreasing it thereafter proportionally to the inverse square root of the step number.

        Returns:
            lr (float32): learning rate for this step.
        """
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """
    Calculate mean loss for target real output and target predict output of one batch.

    Parameters:
        real (tensor): target real output
        pred (tensor): target predict output

    Returns:
        mean loss for one batch

    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask

    return tf.reduce_mean(loss_)


def get_ckpt_manager(transformer_model, optimizer_, max_to_keep=5, checkpoint_path="./checkpoints/train"):
    """
    Get checkpoint manager for save weights on n loop epochs.

    Extended description of function.

    Parameters:
        transformer_model (mdoel): defined transformer model to save
        optimizer_ (optimizer): defined optimizer to save
        max_to_keep (int): how many epochs to save weights
        checkpoint_path (string): which path to save weights

    Returns:
        ckpt_manager: a manager to manage checkpoint
    """

    ckpt = tf.train.Checkpoint(transformer=transformer_model,
                               optimizer=optimizer_)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    return ckpt_manager