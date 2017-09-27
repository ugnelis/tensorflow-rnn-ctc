from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import codecs
import unicodedata
import re


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
