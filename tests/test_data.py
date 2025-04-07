import unittest
from src.data.move_data import move_data

class TestData(unittest.TestCase):
    def test_move_data(self):
        result = move_data()
        self.assertIn(result, [True, False])

if __name__ == "__main__":
    unittest.main()