import pandas as pd
import numpy as np


class MinMaxNormalizer:
    """
        A class for Normalizing Data in a 0-1 Fashion
        It follows the Sci-Kit Learn Design Patterns
    """

    def __init__(self, excluded_columns, bottom=0):
        self.is_fit_ = False
        self.n_feats_ = 0
        self.internal_dict = {}
        self.excluded_columns_ = excluded_columns
        self.bottom = bottom

    def fit(self, pandas_data):
        self.is_fit_ = True
        self.n_feats_ = len(pandas_data.columns)
        for column in pandas_data.columns:
            if self.excluded_columns_ is not None and column not in self.excluded_columns_  :
                self.internal_dict.update({
                    column: {
                        'min': pandas_data[column].min(),
                        'max': pandas_data[column].max(),
                        'range': pandas_data[column].max() - pandas_data[column].min(),
                    }
                })
        return

    def transform(self, pandas_data, fill_na=None):
        if self.is_fit_ is False:
            raise ValueError("Fit Data First re Bro!")
        for column in pandas_data.columns:
            if column in self.internal_dict:
                # Warning if Range is 0 (case of all constant) we need to handle it as it was 1
                if self.internal_dict[column]['range'] == 0:
                    rectified_range = 1
                else:
                    rectified_range = self.internal_dict[column]['range']

                if self.bottom == 0:
                    pandas_data[column] = (pandas_data[column] - self.internal_dict[column]['min']) / (rectified_range)
                else:
                    # [-1,1] Range
                    pandas_data[column] = 2 * (
                                (pandas_data[column] - self.internal_dict[column]['min']) / (rectified_range)) - 1

        if fill_na is not None:
            pandas_data.fillna(value=fill_na, inplace=True)
        return pandas_data

    def fit_transform(self, pandas_data, mode=None, fill_na=0):
        self.fit(pandas_data=pandas_data)
        pandas_data = self.transform(pandas_data=pandas_data, fill_na=fill_na)
        return pandas_data

    def spit_label_metrics(self, label):
        try:
            assert label in self.internal_dict.keys()
        except:
            raise AttributeError("Key not in fitted keys!.\n")
        min = self.internal_dict[label]['min']
        max = self.internal_dict[label]['max']
        range = self.internal_dict[label]['range']
        return min, max, range

    def denormalize(self, data):
        """
        Denormalize data back to original view, can work with Numpy and Pandas Data
        :param data: The data to denormalize
        :return: The denormalized data
        """
        if isinstance(data, pd.DataFrame):
            # We have a DataFrame
            for column in data.columns:
                if self.bottom == 0:
                    data[column] = data[column] * self.internal_dict[column]['range'] + self.internal_dict[column][
                        'min']
                else:
                    data[column] = ((data[column] + 1) / 2.0) * self.internal_dict[column]['range'] + \
                                   self.internal_dict[column]['min']

            return data

        elif isinstance(data, np.ndarray):
            # We have a numpy Array
            raise NotImplementedError("Ftiaxto re!")


