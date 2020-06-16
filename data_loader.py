import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import binary_sampler


def data_loader(data_name, miss_rate, target_column=None):
    """Loads datasets and introduce missingness.

    Args:
      - data_name: letter, spam, or mnist
      - miss_rate: the probability of missing components

    Returns:
      data_x: original data
      miss_data_x: data with missing values
      data_m: indicator matrix for missing components
    """
    file_name = 'data/' + data_name + '.csv'
    print(file_name)
    data_x = pd.read_csv(file_name, delimiter=',')
    train_x, test_x = train_test_split(data_x, test_size=0.4, random_state=1498, shuffle=True)
    if target_column is not None:
        train_y = train_x.pop(target_column)
    else:
        train_y = None
    if target_column is not None:
        test_y = test_x.pop(target_column)
    else:
        test_y = None

        # Parameters
    no_train, dim_train = train_x.shape
    no_test, dim_test = test_x.shape

    # Introduce missing data
    data_m_train = binary_sampler(1 - miss_rate, no_train, dim_train)
    data_m_test = binary_sampler(1 - miss_rate, no_test, dim_test)
    miss_train_x = train_x.astype('float32').copy()
    ori_train_x = train_x.astype('float32').copy()
    miss_test_x = test_x.astype('float32').copy()
    ori_test_x = test_x.astype('float32').copy()
    miss_train_x = miss_train_x.values
    miss_test_x = miss_test_x.values
    miss_train_x[data_m_train == 0] = np.nan
    miss_test_x[data_m_test == 0] = np.nan
    miss_train_x = pd.DataFrame(data=miss_train_x, columns=train_x.columns)
    miss_test_x = pd.DataFrame(data=miss_test_x, columns=test_x.columns)
    return (ori_train_x, train_x, train_y, miss_train_x), (ori_test_x, test_x, test_y, miss_test_x)
