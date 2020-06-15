import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import binary_sampler



def data_loader(data_name, miss_rate, validation_split=0.3, target_column=None):
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
    column_names = data_x.columns
    # train_x, test_x = train_test_split(data_x, test_size=validation_split, random_state=42)
    train_x = data_x.copy()
    test_x = data_x.copy()
    train_y = train_x.pop(target_column) if target_column is not None else None
    train_x = train_x.values
    train_y_test = test_x.pop(target_column) if target_column is not None else None
    test_x = test_x.values


    # Parameters
    no_train, dim_train = train_x.shape
    no_test, dim_test = test_x.shape

    # Introduce missing data
    data_m_train = binary_sampler(1 - miss_rate, no_train, dim_train)
    data_m_test = binary_sampler(1 - miss_rate, no_test, dim_test)
    miss_data_x_train = train_x.astype('float32').copy()
    miss_data_x_test = test_x.astype('float32').copy()
    miss_data_x_train[data_m_train == 0] = np.nan
    miss_data_x_test[data_m_test == 0] = np.nan
    return train_x, miss_data_x_train, data_m_train, test_x, miss_data_x_test, data_m_test, column_names, train_y, train_y_test
