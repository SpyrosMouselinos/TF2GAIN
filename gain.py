import numpy as np
from model import train
from normalizer import MinMaxNormalizer


def gain(miss_train_x, train_y, miss_test_x, test_y, test_x, target_column):
    """Impute missing values in data_x

  Args:
    - miss_train_x: original data with missing values
    - train_y: original target column
    - miss_test_x: validation data with missing values
    - test_y: validation target column
    - test_x: Non-empty Validation Data for reconstruction Purposes
    - target_column: the name of the target column in order to reconstruct imputed data into a dataframe
    - gain_parameters: The Training/Model Parameters


  Returns:
    - imputed_data: imputed data
  """
    # Define mask matrix
    data_m = 1 - np.isnan(miss_train_x)
    data_m_test = 1 - np.isnan(miss_test_x)
    normalizer = MinMaxNormalizer(excluded_columns=target_column)
    normalizer.fit(miss_train_x)
    norm_train_x = normalizer.transform(miss_train_x, fill_na=0)
    norm_test_x = normalizer.transform(miss_test_x, fill_na=0)

    train(norm_train_x=norm_train_x, data_m=data_m, norm_test_x=norm_test_x, data_m_test=data_m_test, train_x=None,
          test_x=None)
    return
