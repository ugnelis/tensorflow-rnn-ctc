from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import codecs
import unicodedata
import re

import tensorflow as tf

import scipy.io.wavfile as wav

DATA_DIR = "data/LibriSpeech/"
TRAIN_DIR = DATA_DIR + "train-clean-100-wav/"
TEST_DIR = DATA_DIR + "test-clean-wav/"
DEV_DIR = DATA_DIR + "dev-clean-wav/"


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


def main(argv):
    # Read text file.
    text_file_path = TRAIN_DIR + "211-122425-0059.txt"
    text = read_text_file(text_file_path)
    text = normalize_text(text)

    # Read audio file.
    wav_file_path = TRAIN_DIR + "211-122425-0059.wav"
    audio_rate, audio_data = wav.read(wav_file_path)


if __name__ == '__main__':
    tf.app.run()
