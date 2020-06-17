import tensorflow as tf
from data_loader import data_loader
from gain import gain
from truck import DNNNetwork
from normalizer import MinMaxNormalizer


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
        'breast': 'diagnosis',
        'credit': 'outcome',
        'spam': 'spam',
        'pima': 'outcome',
        'heart': 'outcome',
        '9': 'foVolConsumption'
    }

    dataset_task = {
        'breast': 'classification',
        'credit': 'classification',
        'spam': 'classification',
        'pima': 'classification',
        'heart': 'classification',
        '9': 'regression'
    }

    task = dataset_task[data_name]
    target_column = dataset_target_columns[data_name]

    (ori_train_x, train_x, train_y, miss_train_x), (ori_test_x, test_x, test_y, miss_test_x) = data_loader(data_name,
                                                                                                           miss_rate,
                                                                                                           target_column)
    train_frame, test_frame = gain(ori_train_x=ori_train_x, ori_test_x=ori_test_x, miss_train_x=miss_train_x,
                                   miss_test_x=miss_test_x,
                                   train_x=train_x, test_x=test_x,
                                   target_column=target_column, train_y=train_y, test_y=test_y,
                                   dataset_name=str(data_name) + '_' + str(int(100 * miss_rate)))

    normalizer = MinMaxNormalizer(excluded_columns=None)
    normalizer.fit(train_frame)
    train_frame = normalizer.transform(train_frame, fill_na=0)
    test_frame = normalizer.transform(test_frame, fill_na=0)

    y_train = train_frame.pop(target_column)
    y_train = y_train.values

    y_test = test_frame.pop(target_column)
    y_test = y_test.values

    x_train = train_frame.values
    x_test = test_frame.values

    network = DNNNetwork(task=task, experiment_name=str(data_name) + '_' + str(int(100 * miss_rate)))
    network.build_network()
    network.fit(x_data=x_train, y_data=y_train, x_data_val=x_test, y_data_val=y_test)
    return


experiment_names = ['9']
                    #'credit', 'spam', 'pima', 'heart']
missing_rates = [0.2, 0.4, 0.6, 0.8]

# In order to run the same code without the tensorflow compiler creating a bug due to errors Use the following: See
# bug: https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function
# \-decorated-functio
tf.config.experimental_run_functions_eagerly(True)
for en in experiment_names:
    for mr in missing_rates:
        perform_experiments(data_name=en, miss_rate=mr)
