import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index


def gain(data_x, gain_parameters):
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

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # Other parameters
    no, dim = data_x.shape

    # Hidden state dimensions
    h_dim = int(dim)

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    # Generator
    # Data + Mask as inputs (Random noise is in missing components)
    def create_generator():
        model = Sequential(name='Generator')
        model.add(Input(shape=(dim * 2,), name='Generator_Input_Layer', dtype='float32'))
        model.add(Dense(units=h_dim, activation='relu', name="Generator_Dense_Layer_1"))
        model.add(Dense(units=h_dim, activation='relu', name="Generator_Dense_Layer_2"))
        model.add(Dense(units=dim, activation='sigmoid', name="Generator_Output_Layer"))
        model.summary()
        return model

    # Discriminator
    # Concatenate Data and Hint
    def create_discriminator():
        model = Sequential(name='Discriminator')
        model.add(Input(shape=(dim * 2,), name='Discriminator_Input_Layer', dtype='float32'))
        model.add(Dense(units=h_dim, activation='relu', name="Discriminator_Dense_Layer_1"))
        model.add(Dense(units=h_dim, activation='relu', name="Discriminator_Dense_Layer_2"))
        model.add(Dense(units=dim, activation='sigmoid', name="Discriminator_Output_Layer"))
        model.summary()
        return model

    ## GAIN structure
    # Create Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                  + (1 - M) * tf.log(1. - D_prob + 1e-8))

    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

    MSE_loss = \
        tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start Iterations
    for it in tqdm(range(iterations)):
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                  feed_dict={M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = \
            sess.run([G_solver, G_loss_temp, MSE_loss],
                     feed_dict={X: X_mb, M: M_mb, H: H_mb})

    ## Return imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data
