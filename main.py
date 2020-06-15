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

    gain_parameters = {'batch_size': args.batch_size,
                       'hint_rate': args.hint_rate,
                       'alpha': args.alpha,
                       'iterations': args.iterations}

    target_column = None
    # Load data and introduce missingness
    ori_data_x, miss_data_x, _, ori_data_x_test, miss_data_x_test, _, column_names, train_y, train_y_test = data_loader(
        data_name, miss_rate, 0.3, target_column)
    # Impute missing data
    imputed_test_frame = gain(ori_data_x, ori_data_x_test, miss_data_x, miss_data_x_test, gain_parameters, column_names, target_column, train_y, train_y_test)

    return


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        default='letter',
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
