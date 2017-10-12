import unittest
import utils

from tempfile import mkdtemp
from shutil import rmtree
from os import path


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.test_dir)

    def test_read_text_file(self):
        file_path = path.join(self.test_dir, 'tmp.txt')
        content = 'test'

        with open(file_path, 'w') as f:
            f.write(content)

        self.assertEqual(utils.read_text_file(file_path), content)


if __name__ == '__main__':
    unittest.main()
