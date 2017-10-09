from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import sys
import math

import tensorflow as tf

import utils

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


def main(argv):
    # Read train data files.
    train_texts = utils.read_text_files(TRAIN_DIR)
    train_labels = utils.texts_encoder(train_texts,
                                       first_index=FIRST_INDEX,
                                       space_index=SPACE_INDEX,
                                       space_token=SPACE_TOKEN)
    train_inputs = utils.read_audio_files(TRAIN_DIR)
    train_inputs = utils.standardize_audios(train_inputs)
    train_sequence_lengths = utils.get_sequence_lengths(train_inputs)
    train_inputs = utils.make_sequences_same_length(train_inputs, train_sequence_lengths)

    # Read validation data files.
    validation_texts = utils.read_text_files(DEV_DIR)
    validation_labels = utils.texts_encoder(validation_texts,
                                            first_index=FIRST_INDEX,
                                            space_index=SPACE_INDEX,
                                            space_token=SPACE_TOKEN)
    validation_labels = utils.sparse_tuples_from_sequences(validation_labels)
    validation_inputs = utils.read_audio_files(DEV_DIR)
    validation_inputs = utils.standardize_audios(validation_inputs)
    validation_sequence_lengths = utils.get_sequence_lengths(validation_inputs)
    validation_inputs = utils.make_sequences_same_length(validation_inputs, validation_sequence_lengths)

    # Read test data files.
    test_texts = utils.read_text_files(TEST_DIR)
    test_labels = utils.texts_encoder(test_texts,
                                      first_index=FIRST_INDEX,
                                      space_index=SPACE_INDEX,
                                      space_token=SPACE_TOKEN)
    test_labels = utils.sparse_tuples_from_sequences(test_labels)
    test_inputs = utils.read_audio_files(DEV_DIR)
    test_inputs = utils.standardize_audios(test_inputs)
    test_sequence_lengths = utils.get_sequence_lengths(test_inputs)
    test_inputs = utils.make_sequences_same_length(test_inputs, test_sequence_lengths)

    with tf.device('/cpu:0'):
        config = tf.ConfigProto()

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

        with tf.Session(config=config, graph=graph) as session:
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
                    batch_train_targets = utils.sparse_tuples_from_sequences(train_labels[indexes])

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

            test_feed = {inputs_placeholder: test_inputs,
                         sequence_length_placeholder: test_sequence_lengths}
            # Decoding.
            decoded_outputs = session.run(decoded[0], feed_dict=test_feed)
            dense_decoded = tf.sparse_tensor_to_dense(decoded_outputs, default_value=-1).eval(session=session)
            test_num = test_texts.shape[0]

            for i, sequence in enumerate(dense_decoded):
                sequence = [s for s in sequence if s != -1]
                decoded_text = utils.sequence_decoder(sequence)

                logging.info("Sequence %d/%d", i + 1, test_num)
                logging.info("Original:\n%s", test_texts[i])
                logging.info("Decoded:\n%s", decoded_text)


if __name__ == '__main__':
    tf.app.run()
