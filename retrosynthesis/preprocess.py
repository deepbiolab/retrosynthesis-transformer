import tensorflow as tf
import time

from transformer import Transformer

def reaction_dataset_v1(filepath='data/retrosynthesis-train.smi'):
    reactants, products = [], []
    with open(filepath) as f:
        reactions = f.readlines()

    for reaction in reactions:
        p, r = reaction.split(" >> ")
        products.append(p.strip())
        reactants.append(r.strip())
    return tf.data.Dataset.from_tensor_slices(
        (tf.constant(products), tf.constant(reactants))
    )


def preprocess(X_batch):
    Z = tf.strings.bytes_split(X_batch)
    return Z.to_tensor()


class TextVectorization(tf.keras.layers.Layer):
    """
    vocab_string = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
    text_vectorization = TextVectorization(vocab_string)
    text_vectorization.adapt(train_dataset)
    """

    def __init__(self, vocab_string, n_oov_buckets=1, dtype=tf.string, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.vocab = list(vocab_string)
        self.n_oov_buckets = n_oov_buckets

    def adapt(self, data_sample):
        chars = tf.constant(self.vocab)
        char_ids = tf.range(len(self.vocab), dtype=tf.int64)
        vocab_init = tf.lookup.KeyValueTensorInitializer(chars, char_ids)
        self.table = tf.lookup.StaticVocabularyTable(vocab_init, self.n_oov_buckets)

    def call(self, inputs):
        preprocessed_inputs = preprocess(inputs)
        return self.table.lookup(preprocessed_inputs)


def chem_tokenizer(
        vocab_string=" ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"):
    vocab = list(vocab_string)
    indices = tf.range(len(vocab), dtype=tf.int64)
    table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
    num_oov_buckets = 1
    table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
    return table

def token_decode(
        vocab_string=" ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"):
    # tokenizer = chem_tokenizer()
    #
    # index = tokenizer.lookup(tf.constant(list(vocab_string))).numpy()
    # char = list(vocab_string)
    return {i: ch for i, ch in enumerate(list(vocab_string))}

def token_encode(
        vocab_string=" ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"):
    # tokenizer = chem_tokenizer()
    #
    # index = tokenizer.lookup(tf.constant(list(vocab_string))).numpy()
    # char = list(vocab_string)
    return {ch: i for i, ch in enumerate(list(vocab_string))}


def get_dataset(trainfile ='data/retrosynthesis-train.smi', validfile='data/retrosynthesis-valid.smi',
                     n_read_threads=5, BUFFER_SIZE=20000, BATCH_SIZE=64):

    train_dataset = tf.data.TextLineDataset(trainfile, num_parallel_reads=n_read_threads)
    train_dataset = train_dataset.map(lambda r: (tf.strings.split(r, sep=" >> ")[0], tf.strings.split(r, sep=" >> ")[1]))
    train_dataset = train_dataset.map(lambda p, r: ("^" + p + "$", "^" + r + "$"))

    valid_dataset = tf.data.TextLineDataset(validfile, num_parallel_reads=n_read_threads)
    valid_dataset = valid_dataset.map(
        lambda r: (tf.strings.split(r, sep=" >> ")[0], tf.strings.split(r, sep=" >> ")[1]))
    valid_dataset = valid_dataset.map(lambda p, r: ("^" + p + "$", "^" + r))

    vocab_string = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
    tokenizer = chem_tokenizer(vocab_string)

    def encode(product, reactant):
        product = tokenizer.lookup(tf.strings.bytes_split(product))
        reactant = tokenizer.lookup(tf.strings.bytes_split(reactant))
        return product, reactant

    def tf_encode(product, reactant):
        result_prod, result_reac = tf.py_function(encode, [product, reactant], [tf.int64, tf.int64])
        result_prod.set_shape([None])
        result_reac.set_shape([None])

        return result_prod, result_reac

    train_dataset = train_dataset.map(tf_encode)
    # put dataset to RAM for speed up
    train_dataset = train_dataset.cache()
    # train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


    valid_dataset = valid_dataset.map(tf_encode)
    valid_dataset = valid_dataset.padded_batch(BATCH_SIZE)

    return train_dataset, valid_dataset, len(vocab_string)-2, len(vocab_string)-2




# sample_transformer = Transformer(
#     num_layers=2, d_model=512, num_heads=8, dff=2048,
#     input_vocab_size=66, target_vocab_size=66,
#     pe_input=10000, pe_target=6000)




# start_time = time.time()

# for i in reaction_dataset_v1().take(3):
#     print(i)

# print(time.time() - start_time)


# # start_time2 = time.time()
# train_dataset, val_dataset, s, _ = get_dataset()
#
# temp_input, temp_target = next(iter(train_dataset))
# print(s)
#
# print(temp_input.shape, temp_target.shape)
#
# fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
#                                enc_padding_mask=None,
#                                look_ahead_mask=None,
#                                dec_padding_mask=None)
#
# print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
# # print(time.time() - start_time2)