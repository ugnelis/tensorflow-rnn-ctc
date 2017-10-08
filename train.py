from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import re
import logging
import sys
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

# Data parameters.
NUM_EXAMPLES = 1
NUM_BATCHES_PER_EPOCH = int(NUM_EXAMPLES / BATCH_SIZE)

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


def main(argv):
    # Read text file.
    text_file_path = TRAIN_DIR + "211-122425-0059.txt"
    text = read_text_file(text_file_path)
    text = normalize_text(text)

    # Read audio file.
    wav_file_path = TRAIN_DIR + "211-122425-0059.wav"
    audio_rate, audio_data = wav.read(wav_file_path)
    inputs = mfcc(audio_data, samplerate=audio_rate)

    # Make text as as char array.
    labels = make_char_array(text, SPACE_TOKEN)
    labels = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in labels])

    # Labels sparse representation for feeding the placeholder.
    train_labels = sparse_tuples_from_sequences([labels])

    # Train inputs.
    train_inputs = np.asarray(inputs[np.newaxis, :])
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    train_sequence_length = [train_inputs.shape[1]]

    # TODO(ugnelis): define different validation variables.
    validation_inputs = train_inputs
    validation_labels = train_labels
    validation_sequence_length = train_sequence_length

    with tf.device('/gpu:0'):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        graph = tf.Graph()
        with graph.as_default():

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
            decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, sequence_length_placeholder)

            label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                                               labels_placeholder))
        with tf.Session(graph=graph) as session:
            # Initialize the weights and biases.
            tf.global_variables_initializer().run()

            for current_epoch in range(NUM_EPOCHS):
                train_cost = train_label_error_rate = 0
                start_time = time.time()

                for batch in range(NUM_BATCHES_PER_EPOCH):
                    feed = {inputs_placeholder: train_inputs,
                            labels_placeholder: train_labels,
                            sequence_length_placeholder: train_sequence_length}

                    batch_cost, _ = session.run([cost, optimizer], feed)
                    train_cost += batch_cost * BATCH_SIZE
                    train_label_error_rate += session.run(label_error_rate, feed_dict=feed) * BATCH_SIZE

                train_cost /= NUM_EXAMPLES
                train_label_error_rate /= NUM_EXAMPLES

                val_feed = {inputs_placeholder: validation_inputs,
                            labels_placeholder: validation_labels,
                            sequence_length_placeholder: validation_sequence_length}

                validation_cost, validation_label_error_rate = session.run([cost, label_error_rate], feed_dict=val_feed)

                # Output intermediate step information.
                logging.info("Epoch %d/%d (time: %.3f s)",
                             current_epoch + 1,
                             NUM_EPOCHS,
                             time.time() - start_time)
                logging.info("Train cost: %.3f, train label error rate: %.3f",
                             train_cost,
                             train_label_error_rate)
                logging.info("Validation cost: %.3f, validation label error rate: %.3f",
                             validation_cost,
                             validation_label_error_rate)

            # Decoding.
            decoded_outputs = session.run(decoded[0], feed_dict=feed)
            decoded_text = sequence_decoder(decoded_outputs[1])

            logging.info("Original:\n%s", text)
            logging.info("Decoded:\n%s", decoded_text)


if __name__ == '__main__':
    tf.app.run()
