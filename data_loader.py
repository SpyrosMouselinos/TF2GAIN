import numpy as np
from utils import binary_sampler



def data_loader(data_name, miss_rate):
    """Loads datasets and introduce missingness.

    Args:
      - data_name: letter, spam, or mnist
      - miss_rate: the probability of missing components

    Returns:
      data_x: original data
      miss_data_x: data with missing values
      data_m: indicator matrix for missing components
    """

    try:
        file_name = 'data/' + data_name + '.csv'
        data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    except:
        print(f"No dataset with prefix {data_name} found! Check the data folder!\n")

    # Parameters
    no, dim = data_x.shape

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m
