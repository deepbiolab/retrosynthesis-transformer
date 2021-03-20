import time
import tensorflow as tf
from transformer import Transformer
from utils import create_masks, loss_function, get_ckpt_manager, CustomSchedule, create_look_ahead_mask
from preprocess import get_dataset
import tensorflow_datasets as tfds


def train(train_dataset, transformer, epochs, ckpt_manager, optimizer):
    """
    Training transformer model for given train_dataset and optimizer.

    Parameters:
        train_dataset (tuple): (encoder input sequence of each batch, decoder input sequence of each batch)
        transformer (model): defined transformer model
        epochs (int): how many epochs for whole training process
        ckpt_manager (class): used for save weights or restore weights
        optimizer (class): defined optimizer

    Returns:
        None
    """

    # The @tf.function will track-compile train_step into TF graph for faster carried out.
    # This function is dedicated to the precise shape of the parameter tensor.
    # In order to avoid variable sequence length Re-tracking caused by batch size (the last batch is smaller),
    # use input_signature to specify More general shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        """
        The target (tar) is divided into tar_inp and tar_real
        tar_inp is passed to the decoder as input. tar_real is the same input shifted by 1:
        at each position in tar_inp, tar_real contains the next token that should be predicted.
        """

        # sentence = "SOS A lion in the jungle is sleeping EOS"
        # tar_inp = "SOS A lion in the jungle is sleeping"
        # tar_real = "A lion in the jungle is sleeping EOS"
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        data = create_look_ahead_mask(tf.shape(tar_inp)[1])

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)
    
    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def evaluate(inp_sentence):
    """
    Summary line.

    Extended description of function.

    Parameters:
    arg1 (int): Description of arg1

    Returns:
    int: Description of return value

    """
    pass


def predict(sentence):
    """
    Summary line.

    Extended description of function.

    Parameters:
    arg1 (int): Description of arg1

    Returns:
    int: Description of return value

    """
    pass


def main():
    # hyperparameters
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    epochs = 20
    pe_input, pe_target = 500, 500

    # prepare dataset
    train_dataset, val_dataset, enc_vocab_size, dec_vocab_size  = get_dataset(
        trainfile ='data/retrosynthesis-train.smi',
        validfile='data/retrosynthesis-valid.smi',
        n_read_threads=5, BUFFER_SIZE=20000, BATCH_SIZE=64)


    input_vocab_size = enc_vocab_size + 2
    target_vocab_size = dec_vocab_size + 2

    # build transformer model
    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=pe_input,
                              pe_target=pe_target,
                              rate=dropout_rate)

    # Create optimizer
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # create model checkpoint
    ckpt_manager = get_ckpt_manager(transformer, optimizer)

    # training
    train(train_dataset, transformer, epochs, ckpt_manager, optimizer)



    # evaluating

    # predicting


if __name__ == '__main__':
    main()
