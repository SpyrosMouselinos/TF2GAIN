import numpy as np
from model import train
from normalizer import MinMaxNormalizer
from sklearn.model_selection import train_test_split


def gain(ori_train_x, ori_test_x, miss_train_x, train_y, miss_test_x, test_y, train_x, test_x, target_column,
         dataset_name):
    """Impute missing values in data_x

  Args:
    - miss_train_x: original data with missing values
    - train_y: original target column
    - miss_test_x: validation data with missing values
    - test_y: validation target column
    - test_x: Non-empty Validation Data for reconstruction Purposes
    - train_x: Non-empty Train Data for reconstruction Purposes
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

    test_frame = train(ori_train_x=ori_train_x, ori_test_x=ori_test_x, norm_train_x=norm_train_x,
                       data_m=data_m, norm_test_x=norm_test_x, data_m_test=data_m_test, train_x=train_x,
                       test_x=test_x, normalizer=normalizer, dataset_name=dataset_name)
    test_frame[target_column] = test_y.values
    train_frame, test_frame = train_test_split(test_frame, test_size=0.3, random_state=42, shuffle=True,
                                               stratify=test_frame[target_column].values)
    return train_frame, test_frame
