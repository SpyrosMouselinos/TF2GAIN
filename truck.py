import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as kb
from sklearn.utils import class_weight
from os import path
from model import FFN


def wbce(y_true, y_pred, weight1=1, weight0=1):
    """
    Custom Implementation of Weighted Binary CrossEntropy for Imbalanced Classification
    :param y_true: The real labels
    :param y_pred: The predicted logits
    :param weight1: Weight for class 1
    :param weight0: Weight for class 0
    :return: WBCE
    """
    y_true = kb.clip(y_true, kb.epsilon(), 1 - kb.epsilon())
    y_pred = kb.clip(y_pred, kb.epsilon(), 1 - kb.epsilon())
    logloss = -(y_true * kb.log(y_pred) * weight1 + (1 - y_true) * kb.log(1 - y_pred) * weight0)
    return kb.mean(logloss, axis=-1)


class DNNNetwork:
    """
        A DNN Network capable of solving the respective tasks.
    """

    def __init__(self, task, experiment_name):
        if task.lower() not in ['regression', 'classification']:
            raise ValueError("Task must be one of the following: regression, classification")
        self.task = task.lower()
        self.epochs = 100
        self.model = None
        self.model_is_built = False
        self.experiment_name = experiment_name
        return

    def build_network(self):
        """
         Builds Network Based on using attention and target task
        """
        self.model = FFN()
        self.lr = 1e-3
        self.batch_size = 256
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.model_is_built = True
        return

    def fit(self, x_data, y_data, x_data_val=None, y_data_val=None):
        """
        Trains the Network.
        :param x_data: Corrupted Input Data
        :param y_data: Input Label/Target
        :param x_data_val: Validation Input Data
        :param y_data_val: Validation Label/Target Data
        :return:
        """

        if self.model_is_built:
            x_dataset = tf.data.Dataset.from_tensor_slices(
                (x_data.astype('float32'), y_data.astype('float32').reshape(-1, 1))).shuffle(
                600000).batch(self.batch_size)

            x_dataset_val = tf.data.Dataset.from_tensor_slices(
                (x_data_val.astype('float32'), y_data_val.astype('float32').reshape(-1, 1))).batch(
                x_data_val.shape[0])

            if self.task == 'regression':
                train_metric = tf.keras.metrics.MeanSquaredError(name='train_loss')
                val_metric = tf.keras.metrics.MeanSquaredError(name='val_loss')
                train_loss = tf.keras.losses.MSE
            elif self.task == 'classification':
                train_metric = tf.keras.metrics.AUC(name='train_loss')
                val_metric = tf.keras.metrics.AUC(name='val_loss')
                train_loss = wbce

            if self.task == 'regression':
                @tf.function
                def train_step(model, x, y):
                    with tf.GradientTape() as tape:
                        predictions = model(q=x)
                        loss = train_loss(y_true=y, y_pred=predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    train_metric(y, predictions)

                @tf.function
                def validation_step(model, x, y):
                    predictions = model(q=x)
                    val_metric(y, predictions)

                previous_val_loss = 100
                count = 0
                for epoch in range(self.epochs):
                    train_metric.reset_states()
                    val_metric.reset_states()

                    for (batch, (x, y)) in enumerate(x_dataset):
                        train_step(self.model, x, y)

                    for x, y in x_dataset_val:
                        validation_step(self.model, x, y)
                    print('Epoch {} Training Loss {:.4f} / Validation Loss {:.4f}'.format(epoch + 1,
                                                                                          train_metric.result(),
                                                                                          val_metric.result()))
                    if val_metric.result() > previous_val_loss:
                        count += 1
                    else:
                        previous_val_loss = val_metric.result()
                        self.model.save_weights('../training_checkpoints/best_model.tf')
                        count = 0
                    if count == 50:
                        self.model.load_weights('../training_checkpoints/best_model.tf')
                        break

                self.model.load_weights('../training_checkpoints/best_model.tf')
                predictions, _ = self.model(q=x_data_val.astype('float32'))
                result_loss_mse = tf.keras.metrics.MeanSquaredError()(y_data_val, predictions).numpy()
                result_loss_mae = tf.keras.metrics.MeanAbsoluteError()(y_data_val, predictions).numpy()

                result = (result_loss_mse, result_loss_mae)
            elif self.task == 'classification':
                print(f"Class 0 count: {y_data[y_data == 0].shape[0]}")
                print(f"Class 1 count: {y_data[y_data == 1].shape[0]}")
                class_weights = class_weight.compute_class_weight('balanced',
                                                                  np.unique(y_data),
                                                                  y_data)

                @tf.function
                def train_step(model, x, y):
                    with tf.GradientTape() as tape:
                        predictions = model(q=x, training=True)
                        loss = train_loss(y_true=y, y_pred=predictions, weight0=class_weights[0],
                                          weight1=class_weights[1])
                    gradients = tape.gradient(loss, model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    train_metric(y, predictions)

                @tf.function
                def validation_step(model, x, y):
                    predictions = model(q=x, training=False)
                    val_metric(y, predictions)

                previous_val_loss = 0
                count = 0
                for epoch in range(self.epochs):
                    train_metric.reset_states()
                    val_metric.reset_states()

                    for (batch, (x, y)) in enumerate(x_dataset):
                        train_step(self.model, x, y)

                    for x, y in x_dataset_val:
                        validation_step(self.model, x, y)
                    print('Epoch {} Training AUC {:.4f} / Validation AUC {:.4f}'.format(epoch + 1,
                                                                                        train_metric.result(),
                                                                                        val_metric.result()))
                    if val_metric.result() < previous_val_loss:
                        count += 1
                    else:
                        previous_val_loss = val_metric.result()
                        self.model.save_weights('../training_checkpoints/best_model.tf')
                        count = 0
                    if count == 50:
                        self.model.load_weights('../training_checkpoints/best_model.tf')
                        break

                self.model.load_weights('../training_checkpoints/best_model.tf')
                predictions = self.model(q=x_data_val.astype('float32'), training=False)
                predictions = np.squeeze(predictions)
                result_acc = tf.keras.metrics.BinaryAccuracy()(y_data_val, predictions).numpy()
                result_roc = tf.keras.metrics.AUC()(y_true=y_data_val, y_pred=predictions).numpy()
                result = (result_acc, result_roc)
            self.__store_run_results(result=result)
            return
        else:
            raise ValueError("Build model first!")

    def __store_run_results(self, result):
        file = 'results/{}_results.csv'.format(self.experiment_name)
        if self.task == 'regression':
            metric_1_name = "Val_MSE"
            metric_2_name = "Val_MSE"
        else:
            metric_1_name = "Val_Acc"
            metric_2_name = "Val_ROC"

        if path.exists(file):
            data = pd.read_csv(file)
            new_row = pd.DataFrame([[result[0], result[1]]],
                                   columns=[metric_1_name, metric_2_name])
            data = data.append(new_row)
            data.to_csv(file, index=False)
        else:
            data = pd.DataFrame([[result[0], result[1]]],
                                columns=[metric_1_name, metric_2_name])
            data.to_csv(file, index=False)
