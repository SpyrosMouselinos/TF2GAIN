from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
from data_loader import data_loader
from gain import gain
from utils import rmse_loss


def perform_experiments(data_name, miss_rate):
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

    dataset_target_columns = {
        'spam': 'spam',
        'letter': None,
        'breast': 'diagnosis',
        'credit': 'outcome',
        'heart': 'outcome',
        'pima': 'outcome',
    }

    target_column = dataset_target_columns[data_name]

    (ori_train_x, train_x, train_y, miss_train_x), (ori_test_x, test_x, test_y, miss_test_x) = data_loader(data_name,
                                                                                                           miss_rate,
                                                                                                           target_column)
    train_frame, test_frame = gain(ori_train_x=ori_train_x, ori_test_x=ori_test_x, miss_train_x=miss_train_x,
                                   miss_test_x=miss_test_x,
                                   train_x=train_x, test_x=test_x,
                                   target_column=target_column, train_y=train_y, test_y=test_y,
                                   dataset_name=str(data_name) + '_' + str(int(100 * miss_rate)))

    train_frame.to_csv('/home/spyros/Desktop/train.csv', index=False)
    test_frame.to_csv('/home/spyros/Desktop/test.csv', index=False)
    return


experiment_names = ['spam', 'letter', 'breast', 'credit', 'heart', 'pima']
missing_rates = [0.2, 0.4, 0.6, 0.8]

# In order to run the same code without the tensorflow compiler creating a bug due to errors Use the following: See
# bug: https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function
# \-decorated-functio
tf.config.experimental_run_functions_eagerly(True)
for en in experiment_names:
    for mr in missing_rates:
        perform_experiments(data_name=en, miss_rate=mr)
        break
    break
