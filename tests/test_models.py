import unittest
from src.models.train import train_model

class TestModels(unittest.TestCase):
    def test_train(self):
        train_model()

if __name__ == "__main__":
    unittest.main()