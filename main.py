import argparse
import numpy as np

from model import PMF
from utils import rmse
from utils import get_data, train_test_split

np.random.seed(42)

if __name_ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="CPMF", type=str, help="Algorithm to use")
    parser.add_argument(
        "--mode", default="netflix", type=str, help="Download data mode or load data mode"
    )
    parser.add_argument("--data_url", type=str, help="Url for rating data")
    parser.add_argument("--data_path", type=str, help="Folder for existing data")
    parser.add_argument("--epoch", default=20, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=1000, type=int, help="Size of batches")
    parser.add_argument("--lamda", default=0.01, type=float, help="Regularization parameter")
    parser.add_argument("--momentum", default=0.8, type=float, help="Momentum for SGD")
    parser.add_argument("--lr", default=10, type=float, help="Learning rate parameter")
    parser.add_argument("--features", default=10, type=int, help="Number of latent features")
    parser.add_argument(
        "--test_ratio", default=0.2, type=float, help="Ratio of size of test dataset"
    )

    args = parser.parse_args()
    """https://github.com/mertyg/probabilistic-matrix-factorization/blob/master/main.py"""

