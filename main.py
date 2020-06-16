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

    (ori_train_x, train_x, train_y, miss_train_x), (ori_test_x, test_x, test_y, miss_test_x) = data_loader(data_name, miss_rate, target_column)
    gain(ori_train_x=ori_train_x, ori_test_x=ori_test_x,miss_train_x=miss_train_x, miss_test_x=miss_test_x, train_x=train_x, test_x=test_x,
         target_column=target_column, train_y=train_y, test_y=test_y)
    return


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        default='breast',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.2,
        type=float)
    args = parser.parse_args()
    main(args)
