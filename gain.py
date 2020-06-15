import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import numpy as np
import pandas as pd
from utils import rmse_loss
from utils import normalization, renormalization, rounding
from utils import binary_sampler, uniform_sampler, sample_batch_index


def gain(ori_data_x, ori_data_x_test, data_x, data_x_test, gain_parameters, column_names=None, target_column=None, train_y=None, test_y=None):
    """Impute missing values in data_x

  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations

  Returns:
    - imputed_data: imputed data
  """
    # Define mask matrix
    data_m = 1 - np.isnan(data_x)
    data_m_test = 1 - np.isnan(data_x_test)

    # System parameters
    BATCH_SIZE = gain_parameters['batch_size']
    HINT_RATE = gain_parameters['hint_rate']
    ALPHA = gain_parameters['alpha']
    ITERATIONS = gain_parameters['iterations']

    # Other parameters
    NO, DIM = data_x.shape
    NO_TEST, _ = data_x_test.shape

    # Hidden state dimensions
    H_DIM = int(DIM)

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_test = np.zeros_like(data_x_test)
    for i in range(DIM):
        norm_data_test[:, i] = data_x_test[:, i] - norm_parameters['min_val'][i]
        norm_data_test[:, i] = data_x_test[:, i] / norm_parameters['max_val'][i]
    norm_data_x = np.nan_to_num(norm_data, 0)
    norm_data_x_test = np.nan_to_num(norm_data_test, 0)

    # Generator
    # Data + Mask as inputs (Random noise is in missing components)
    def create_generator():
        model = Sequential(name='Generator')
        model.add(Input(shape=(DIM * 2,), name='Generator_Input_Layer', dtype='float32'))
        model.add(Dense(units=H_DIM, activation='relu', name="Generator_Dense_Layer_1"))
        model.add(Dense(units=H_DIM, activation='relu', name="Generator_Dense_Layer_2"))
        model.add(Dense(units=DIM, activation='sigmoid', name="Generator_Output_Layer"))
        model.summary()
        return model

    # Discriminator
    # Concatenate Data and Hint
    def create_discriminator():
        model = Sequential(name='Discriminator')
        model.add(Input(shape=(DIM * 2,), name='Discriminator_Input_Layer', dtype='float32'))
        model.add(Dense(units=H_DIM, activation='relu', name="Discriminator_Dense_Layer_1"))
        model.add(Dense(units=H_DIM, activation='relu', name="Discriminator_Dense_Layer_2"))
        model.add(Dense(units=DIM, activation='sigmoid', name="Discriminator_Output_Layer"))
        model.summary()
        return model

    generator = create_generator()
    discriminator = create_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-3, name='GenOpt')
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-3, name='DiscOpt')

    @tf.function
    def train_step(X, M, H):
        """
            The training schema as defined in the original paper implementation
            We will use for convenience the same Variable Names in order to Point
            out the calculations we perform
        """
        X = tf.cast(X, dtype='float32')
        M = tf.cast(M, dtype='float32')
        H = tf.cast(H, dtype='float32')

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Lets create an Imputation
            G_sample = generator(tf.concat([X, M], axis=1))

            # Combine with observed data
            Hat_X = X * M + G_sample * (1 - M)

            # Discriminate Between Them
            D_prob = discriminator(tf.concat([Hat_X, H], axis=1))

            # Calculate Adversarial Losses(NegLogLik)
            D_loss_temp = -tf.reduce_mean(M * kb.log(D_prob + 1e-8) + (1 - M) * kb.log(1. - D_prob + 1e-8))
            G_loss_temp = -tf.reduce_mean((1 - M) * kb.log(D_prob + 1e-8))

            # Add Extra Reconstruction MSE loss in the generator
            MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

            # Finalize Losses
            D_loss = D_loss_temp
            G_loss = G_loss_temp + ALPHA * MSE_loss

        gradients_of_generator = gen_tape.gradient(G_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(D_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return D_loss, G_loss

    def validation_step():
        """
            The validation schema is absent in the original paper implementation
            We will use for convenience the RMSE Value that is used as a Metric
            to perform Early Stopping and monitor the During-Training Performance of the Model
        """
        Z_mb = uniform_sampler(0, 0.01, NO_TEST, DIM)
        Z_mb = Z_mb.astype('float32')
        M_mb = data_m_test
        M_mb = M_mb.astype('float32')
        X_mb = norm_data_x_test
        X_mb = X_mb.astype('float32')
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        imputed_data = generator.predict(tf.concat([X_mb, M_mb], axis=1))[0]
        imputed_data = data_m_test * norm_data_x_test + (1 - data_m_test) * imputed_data

        # Renormalization
        imputed_data = renormalization(imputed_data, norm_parameters)

        # Rounding
        imputed_data = rounding(imputed_data, data_x_test)

        rmse = rmse_loss(ori_data_x_test, imputed_data, data_m_test)
        print(f"Validation RMSE:{rmse}")
        return rmse, imputed_data

    def train():
        counter = 0
        rmse_old = 500
        for idx in range(ITERATIONS):
            # Sample batch
            batch_idx = sample_batch_index(NO, BATCH_SIZE)
            X_mb = norm_data_x[batch_idx, :]
            M_mb = data_m[batch_idx, :]
            # Sample random vectors
            Z_mb = uniform_sampler(0, 0.01, BATCH_SIZE, DIM)
            # Sample hint vectors
            H_mb_temp = binary_sampler(HINT_RATE, BATCH_SIZE, DIM)
            H_mb = M_mb * H_mb_temp
            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
            # Feed N Run
            dl, gl = train_step(X=X_mb, M=M_mb, H=H_mb)
            if idx % 500 == 0:
                print('Generator Loss: ' + str(gl.numpy()) + ' Discriminator Loss: ' + str(dl.numpy()))
                rmse_new, imputed_data = validation_step()
                if rmse_new < rmse_old:
                    rmse_old = rmse_new
                    df = pd.DataFrame(data=imputed_data, columns=list(column_names)[:-1])
                    df[target_column] = test_y
                    df.to_csv('Imputed_Data.csv', index=False)
                else:
                    counter += 1
            if counter > 5:
                break
    train()
    return
