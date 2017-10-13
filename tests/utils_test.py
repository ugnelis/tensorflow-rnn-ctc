import unittest
import utils

from tempfile import mkdtemp, mktemp
from shutil import rmtree

# Extension of created test files.
TEST_FILE_EXTENSION = 'txt'

# Directory of test suite wav audio files.
TEST_AUDIO_FILE_DIR = 'data/LibriSpeech/test-clean-wav'


class UtilsTest(unittest.TestCase):
    def setUp(self):
        self.test_file_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.test_file_dir)

    def test_read_text_file(self):
        content, file_path = self.create_test_file('test')
        self.assertEqual(utils.read_text_file(file_path), content)

    def test_make_char_array(self):
        self.assertEqual(utils.make_char_array('ab').tolist(), ['a', 'b'])

    def test_normalize_text(self):
        text = 'A\' '

        self.assertEqual(utils.normalize_text(text), 'a')
        self.assertEqual(utils.normalize_text(text, False), 'a\'')

    def test_sparse_tuples_from_sequences(self):
        result = utils.sparse_tuples_from_sequences([[1], [2, 3]])

        self.assertEqual(result[0].tolist(), [[0, 0], [1, 0], [1, 1]])
        self.assertEqual(result[1].tolist(), [1, 2, 3])
        self.assertEqual(result[2].tolist(), [2, 2])

    def test_read_audio_files(self):
        self.assertTrue(utils.read_audio_files(TEST_AUDIO_FILE_DIR).size > 0)

    def test_read_text_files(self):
        content = self.create_test_file('test')[0]
        self.assertEqual(utils.read_text_files(self.test_file_dir, [TEST_FILE_EXTENSION]), content)

    def test_sequence_decoder(self):
        self.assertEqual(utils.sequence_decoder([1, 2, 3]), 'abc')

    def test_texts_encoder(self):
        self.assertEqual(utils.texts_encoder(['abc']).tolist()[0], [1, 2, 3])

    def test_standardize_audios(self):
        files = utils.read_audio_files(TEST_AUDIO_FILE_DIR)
        self.assertEqual(utils.standardize_audios(files).size, files.size)

    def test_get_sequence_lengths(self):
        self.assertEqual(utils.get_sequence_lengths([[1], [], [1, 2]]).tolist(), [1, 0, 2])

    def test_make_sequences_same_length(self):
        self.assertEqual(utils.make_sequences_same_length([[1, 2], []], [2, 0]).tolist(), [[1.0, 2.0], [0.0, 0.0]])

    def create_test_file(self, content):
        """
        Write a string to a new temporary test file.

        Args:
            content:
                Content to write to the text file.
        Returns:
            A tuple with (file content, file path).
        """
        file_path = mktemp(suffix='.' + TEST_FILE_EXTENSION, dir=self.test_file_dir)

        with open(file_path, 'w') as f:
            f.write(content)

        return content, file_path


if __name__ == '__main__':
    unittest.main()
