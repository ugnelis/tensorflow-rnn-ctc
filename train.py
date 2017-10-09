from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import re
import logging
import sys
import math
import unicodedata
import codecs

import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

# Logging configuration.
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

DATA_DIR = "data/LibriSpeech/"
TRAIN_DIR = DATA_DIR + "train-clean-100-wav/"
TEST_DIR = DATA_DIR + "test-clean-wav/"
DEV_DIR = DATA_DIR + "dev-clean-wav/"

# Constants.
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Number of features.
NUM_FEATURES = 13

# Accounting the 0th index + space + blank label = 28 characters
NUM_CLASSES = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters.
NUM_EPOCHS = 200
NUM_HIDDEN = 50
NUM_LAYERS = 1
BATCH_SIZE = 1

# Optimizer parameters.
INITIAL_LEARNING_RATE = 1e-2
MOMENTUM = 0.9


def read_text_file(path):
    """
    Read text from file

    Args:
        path: string.
            Path to text file.

    Returns:
        string.
            Read text.
    """
    with codecs.open(path, encoding="utf-8") as file:
        return file.read()


def make_char_array(text, space_token='<space>'):
    """
    Make text as char array. Replace spaces with space token.

    Args:
        text: string.
            Given text.
        space_token: string.
            Text which represents space char.

    Returns:
        string array.
            Split text.
    """
    result = np.hstack([space_token if x == ' ' else list(x) for x in text])
    return result


def normalize_text(text, remove_apostrophe=True):
    """
    Normalize given text.

    Args:
        text: string.
            Given text.
        remove_apostrophe: bool.
            Whether to remove apostrophe in given text.

    Returns:
        string.
            Normalized text.
    """

    # Convert unicode characters to ASCII.
    result = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()

    # Remove apostrophes.
    if remove_apostrophe:
        result = result.replace("'", "")

    return re.sub("[^a-zA-Z']+", ' ', result).strip().lower()


def sparse_tuples_from_sequences(sequences, dtype=np.int32):
    """
    Create a sparse representations of inputs.

    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indexes = []
    values = []

    for n, sequence in enumerate(sequences):
        indexes.extend(zip([n] * len(sequence), range(len(sequence))))
        values.extend(sequence)

    indexes = np.asarray(indexes, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indexes).max(0)[1] + 1], dtype=np.int64)

    return indexes, values, shape


def read_audio_files(dir, extensions=['wav']):
    """
    Read audio files.

    Args:
        dir: string.
            Data directory.
        extensions: list of strings.
            File extensions.
    Returns:
        files: array of audios.
    """
    if not os.path.isdir(dir):
        logging.error("Audio files directory %s is not found.", dir)
        return None

    if not all(isinstance(extension, str) for extension in extensions):
        logging.error("Variable 'extensions' is not a list of strings.")
        return None

    # Get files list.
    files_paths_list = []
    for extension in extensions:
        file_glob = os.path.join(dir, '*.' + extension)
        files_paths_list.extend(glob.glob(file_glob))

    # Read files.
    files = []
    for file_path in files_paths_list:
        audio_rate, audio_data = wav.read(file_path)
        file = mfcc(audio_data, samplerate=audio_rate)
        files.append(file)

    files = np.array(files)
    return files


def read_text_files(dir, extensions=['txt']):
    """
    Read text files.

    Args:
        dir: string.
            Data directory.
        extensions: list of strings.
            File extensions.
    Returns:
        files: array of texts.
    """
    if not os.path.isdir(dir):
        logging.error("Text files directory %s is not found.", dir)
        return None

    if not all(isinstance(extension, str) for extension in extensions):
        logging.error("Variable 'extensions' is not a list of strings.")
        return None

    # Get files list.
    files_paths_list = []
    for extension in extensions:
        file_glob = os.path.join(dir, '*.' + extension)
        files_paths_list.extend(glob.glob(file_glob))

    # Read files.
    files = []
    for file_path in files_paths_list:
        file = read_text_file(file_path)
        file = normalize_text(file)
        files.append(file)

    files = np.array(files)
    return files


def sequence_decoder(sequence, first_index=(ord('a') - 1)):
    """
    Read text files.

    Args:
        sequence: list of int.
            Encoded sequence
        first_index: int.
            First index (usually index of 'a').
    Returns:
        decoded_text: string.
    """
    decoded_text = ''.join([chr(x) for x in np.asarray(sequence) + first_index])
    # Replacing blank label to none.
    decoded_text = decoded_text.replace(chr(ord('z') + 1), '')
    # Replacing space label to space.
    decoded_text = decoded_text.replace(chr(ord('a') - 1), ' ')
    return decoded_text


def texts_encoder(texts, first_index=(ord('a') - 1), space_index=0, space_token='<space>'):
    """
    Encode texts to numbers.

    Args:
        texts: list of texts.
            Data directory.
        first_index: int.
            First index (usually index of 'a').
        space_index: int.
            Index of 'space'.
        space_token: string.
            'space' representation.
    Returns:
        array of encoded texts.
    """
    result = []
    for text in texts:
        item = make_char_array(text, space_token)
        item = np.asarray([space_index if x == space_token else ord(x) - first_index for x in item])
        result.append(item)

    return np.array(result)


def standardize_audios(inputs):
    """
    Standardize audio inputs.

    Args:
        inputs: array of audios.
            Audio files.
    Returns:
        decoded_text: array of audios.
    """
    result = []
    for i in range(inputs.shape[0]):
        item = np.array((inputs[i] - np.mean(inputs[i])) / np.std(inputs[i]))
        result.append(item)

    return np.array(result)


def get_sequence_lengths(inputs):
    """
    Get sequence length of each sequence.

    Args:
        inputs: list of lists where each element is a sequence.
    Returns:
        array of sequence lengths.
    """
    result = []
    for input in inputs:
        result.append(len(input))

    return np.array(result, dtype=np.int64)


def make_sequences_same_length(sequences, sequence_lengths, default_value=0.0):
    """
    Make sequences same length for avoiding value
    error: setting an array element with a sequence.

    Args:
        sequences: list of sequence arrays.
        sequence_lengths: list of int.
        default_value: float32.
            Default value of newly created array.
    Returns:
        result: array of with same dimensions [num_samples, max_length, num_features].
    """

    # Get number of sequnces.
    num_samples = len(sequences)

    max_length = np.max(sequence_lengths)

    # Get shape of the first non-zero length sequence.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    # Create same sizes array
    result = (np.ones((num_samples, max_length) + sample_shape) * default_value)

    # Put sequences into new array.
    for idx, sequence in enumerate(sequences):
        result[idx, :len(sequence)] = sequence

    return result


def main(argv):
    # Read train data files.
    train_texts = read_text_files(TRAIN_DIR)
    train_labels = texts_encoder(train_texts,
                                 first_index=FIRST_INDEX,
                                 space_index=SPACE_INDEX,
                                 space_token=SPACE_TOKEN)
    train_inputs = read_audio_files(TRAIN_DIR)
    train_inputs = standardize_audios(train_inputs)
    train_sequence_lengths = get_sequence_lengths(train_inputs)
    train_inputs = make_sequences_same_length(train_inputs, train_sequence_lengths)

    # Read validation data files.
    validation_texts = read_text_files(DEV_DIR)
    validation_labels = texts_encoder(validation_texts,
                                      first_index=FIRST_INDEX,
                                      space_index=SPACE_INDEX,
                                      space_token=SPACE_TOKEN)
    validation_labels = sparse_tuples_from_sequences(validation_labels)
    validation_inputs = read_audio_files(DEV_DIR)
    validation_inputs = standardize_audios(validation_inputs)
    validation_sequence_lengths = get_sequence_lengths(validation_inputs)
    validation_inputs = make_sequences_same_length(validation_inputs, validation_sequence_lengths)

    # Read test data files.
    test_texts = read_text_files(TEST_DIR)
    test_labels = texts_encoder(test_texts,
                                first_index=FIRST_INDEX,
                                space_index=SPACE_INDEX,
                                space_token=SPACE_TOKEN)
    test_labels = sparse_tuples_from_sequences(test_labels)
    test_inputs = read_audio_files(DEV_DIR)
    test_inputs = standardize_audios(test_inputs)
    test_sequence_lengths = get_sequence_lengths(test_inputs)
    test_inputs = make_sequences_same_length(test_inputs, test_sequence_lengths)

    with tf.device('/gpu:0'):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        graph = tf.Graph()
        with graph.as_default():
            logging.debug("Starting new TensorFlow graph.")
            inputs_placeholder = tf.placeholder(tf.float32, [None, None, NUM_FEATURES])

            # SparseTensor placeholder required by ctc_loss op.
            labels_placeholder = tf.sparse_placeholder(tf.int32)

            # 1d array of size [batch_size].
            sequence_length_placeholder = tf.placeholder(tf.int32, [None])

            # Defining the cell.
            cell = tf.contrib.rnn.LSTMCell(NUM_HIDDEN, state_is_tuple=True)

            # Stacking rnn cells.
            stack = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS,
                                                state_is_tuple=True)

            # Creates a recurrent neural network.
            outputs, _ = tf.nn.dynamic_rnn(stack, inputs_placeholder, sequence_length_placeholder, dtype=tf.float32)

            shape = tf.shape(inputs_placeholder)
            batch_s, max_time_steps = shape[0], shape[1]

            # Reshaping to apply the same weights over the time steps.
            outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])

            weigths = tf.Variable(tf.truncated_normal([NUM_HIDDEN,
                                                       NUM_CLASSES],
                                                      stddev=0.1))
            biases = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]))

            # Doing the affine projection.
            logits = tf.matmul(outputs, weigths) + biases

            # Reshaping back to the original shape.
            logits = tf.reshape(logits, [batch_s, -1, NUM_CLASSES])

            # Time is major.
            logits = tf.transpose(logits, (1, 0, 2))

            loss = tf.nn.ctc_loss(labels_placeholder, logits, sequence_length_placeholder)
            cost = tf.reduce_mean(loss)

            optimizer = tf.train.MomentumOptimizer(INITIAL_LEARNING_RATE, 0.9).minimize(cost)

            # CTC decoder.
            decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(logits, sequence_length_placeholder)

            label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                                               labels_placeholder))

    with tf.Session(graph=graph) as session:
        logging.debug("Starting TensorFlow session.")
        # Initialize the weights and biases.
        tf.global_variables_initializer().run()

        train_num = train_inputs.shape[0]
        validation_num = validation_inputs.shape[0]

        # Check if there is any example.
        if train_num <= 0:
            logging.error("There are no training examples.")
            return

        num_batches_per_epoch = math.ceil(train_num / BATCH_SIZE)

        for current_epoch in range(NUM_EPOCHS):
            train_cost = 0
            train_label_error_rate = 0
            start_time = time.time()

            for batch in range(num_batches_per_epoch):
                # Format batches.
                if int(train_num / ((batch + 1) * BATCH_SIZE)) >= 1:
                    indexes = [i % train_num for i in range(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)]
                else:
                    indexes = [i % train_num for i in range(batch * BATCH_SIZE, train_num)]

                batch_train_inputs = train_inputs[indexes]
                batch_train_sequence_lengths = train_sequence_lengths[indexes]
                batch_train_targets = sparse_tuples_from_sequences(train_labels[indexes])

                feed = {inputs_placeholder: batch_train_inputs,
                        labels_placeholder: batch_train_targets,
                        sequence_length_placeholder: batch_train_sequence_lengths}

                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost * BATCH_SIZE
                train_label_error_rate += session.run(label_error_rate, feed_dict=feed) * BATCH_SIZE

            train_cost /= train_num
            train_label_error_rate /= train_num

            validation_feed = {inputs_placeholder: validation_inputs,
                               labels_placeholder: validation_labels,
                               sequence_length_placeholder: validation_sequence_lengths}

            validation_cost, validation_label_error_rate = session.run([cost, label_error_rate],
                                                                       feed_dict=validation_feed)

            validation_cost /= validation_num
            validation_label_error_rate /= validation_num

            # Output intermediate step information.
            print("Epoch %d/%d (time: %.3f s)" %
                  (current_epoch + 1, NUM_EPOCHS, time.time() - start_time))
            print("Train cost: %.3f, train label error rate: %.3f" %
                  (train_cost, train_label_error_rate))
            print("Validation cost: %.3f, validation label error rate: %.3f" %
                  (validation_cost, validation_label_error_rate))

        test_feed = {inputs_placeholder: test_inputs,
                     sequence_length_placeholder: test_sequence_lengths}
        # Decoding.
        decoded_outputs = session.run(decoded[0], feed_dict=test_feed)
        dense_decoded = tf.sparse_tensor_to_dense(decoded_outputs, default_value=-1).eval(session=session)

        for i, sequence in enumerate(dense_decoded):
            sequence = [s for s in sequence if s != -1]
            decoded_text = sequence_decoder(sequence)

            print('Sequence %d' % i)
            logging.info("Original:\n%s", test_texts[i])
            logging.info("Decoded:\n%s", decoded_text)


if __name__ == '__main__':
    tf.app.run()
