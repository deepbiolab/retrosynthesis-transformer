import time
import argparse
import time
import tensorflow as tf
from transformer import Transformer
from utils import create_masks, loss_function, get_ckpt_manager, CustomSchedule, create_look_ahead_mask
from preprocess import get_dataset, token_decode, token_encode, chem_tokenizer
tokenizer = chem_tokenizer()
token_encode_dic = token_encode()
token_decode_dic = token_decode()



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


def evaluate(transformer, inp_sequence, max_length=160):
    """
    Given input sequence for encoder and predict target for decoder

    Parameters:
        inp_sequence (string): input sequence for encoder

    Returns:
        predict logits: shape = (seq_len, vocab_size)

    """
    start_token = token_encode_dic["^"]
    end_token = token_encode_dic["$"]
    # inp_sentence is product, shape = (len(inp_sentence)+2, )
    inp_sequence = tokenizer.lookup(tf.strings.bytes_split("^"+inp_sequence+"$"))
    encoder_input = tf.expand_dims(inp_sequence, 0) # (1, len(inp_sentence)+2)

    # target is reactant，first char is "^"
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # each time choose last element from seq_len dimension, because sequence is shifted
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # if predicted_id == ，就返回结果
        if predicted_id == end_token:
            return tf.squeeze(output, axis=0), attention_weights

        # each time append predicted_id to output, finaly will get whole target sequence
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def predict(transformer, inp_sequence, max_length=160):
    """
    Summary line.

    Extended description of function.

    Parameters:
    arg1 (int): Description of arg1

    Returns:
    int: Description of return value

    """

    result, attention_weights = evaluate(transformer, inp_sequence, max_length=160)

    predicted_reactants = [token_decode_dic[int(i)] for i in result.numpy() if i < len(token_decode_dic)]

    return "".join(predicted_reactants)

def main():
    parser = argparse.ArgumentParser(description="Retrosynthesis Prediction using Transformer")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='predict',
                        help='Mode to run the script: train or predict')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--train_file', type=str, default='data/retrosynthesis-train.smi',
                        help='Path to the training data file')
    parser.add_argument('--valid_file', type=str, default='data/retrosynthesis-valid.smi',
                        help='Path to the validation data file')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save/load checkpoints')
    parser.add_argument('--input_sequence', type=str, default="Ic1ccc2n(CC(=O)N3CCCCC3)c3CCN(C)Cc3c2c1",
                        help='Input SMILES string for prediction')
    args = parser.parse_args()

    # Hyperparameters
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    epochs = args.epochs
    pe_input, pe_target = 500, 500

    # Prepare dataset
    train_dataset, val_dataset, enc_vocab_size, dec_vocab_size = get_dataset(
        trainfile=args.train_file,
        validfile=args.valid_file,
        n_read_threads=5, BUFFER_SIZE=20000, BATCH_SIZE=args.batch_size)

    input_vocab_size = enc_vocab_size + 2
    target_vocab_size = dec_vocab_size + 2

    # Build transformer model
    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=pe_input,
                              pe_target=pe_target,
                              rate=dropout_rate)

    # Create optimizer
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

    # Create model checkpoint
    ckpt_manager = get_ckpt_manager(transformer, optimizer, checkpoint_dir=args.checkpoint_dir)

    # Load latest checkpoint if available
    if ckpt_manager.latest_checkpoint:
        ckpt_manager.restore_or_initialize()
        print(f'Loaded checkpoint from {ckpt_manager.latest_checkpoint}')
    else:
        print('Initializing new checkpoints.')

    if args.mode == 'train':
        print("Starting training...")
        train(train_dataset, transformer, epochs, ckpt_manager, optimizer)
        print("Training completed.")
    elif args.mode == 'predict':
        # Ensure the model is loaded
        if not ckpt_manager.latest_checkpoint:
            print("No checkpoint found. Please train the model first.")
            return
        # Perform prediction
        inp_sequence = args.input_sequence
        reactant = predict(transformer, inp_sequence, max_length=160)
        print('Input Product:       {}'.format(inp_sequence))
        print('Predicted Reactants: {}'.format(reactant))
    else:
        print("Invalid mode selected. Choose either 'train' or 'predict'.")

if __name__ == '__main__':
    main()