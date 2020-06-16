from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from data_loader import data_loader
from gain import gain
from utils import rmse_loss


def main(args):
    """Main function for UCI letter and spam datasets.

  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations

  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  """

    data_name = args.data_name
    miss_rate = args.miss_rate

    dataset_target_columns = {
        'spam': 'spam',
        'letter': None,
        'breast': 'diagnosis',
        'credit': 'outcome',
        'heart': 'outcome',
        'pima': 'outcome',
    }

    target_column = dataset_target_columns[data_name]

    (train_x, train_y, miss_train_x), (test_x, test_y, miss_test_x) = data_loader(data_name, miss_rate, target_column)
    gain(miss_train_x, train_y, miss_test_x, test_y, test_x, target_column)
    return


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        default='spam',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.2,
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training interations',
        default=10000,
        type=int)

    args = parser.parse_args()

    # Calls main function
    main(args)
