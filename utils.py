from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import logging
import unicodedata
import codecs

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc


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
