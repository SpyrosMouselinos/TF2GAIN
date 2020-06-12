import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import numpy as np
import os
import time
from IPython import display
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

    generator = create_generator()
    discriminator = create_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, name='GenOpt')
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, name='DiscOpt')

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    @tf.function
    def train_step(X, M, H):
        """
            The training schema as defined in the original paper implementation
            We will use for convenience the same Variable Names in order to Point
            out the calculations we perform
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Lets create an Imputation
            G_sample = generator(X, M)

            # Combine with observed data
            Hat_X = X * M + G_sample * (1 - M)

            # Discriminate Between Them
            D_prob = discriminator(Hat_X, H)

            # Calculate Adversarial Losses(NegLogLik)
            D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
            G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

            # Add Extra Reconstruction MSE loss in the generator
            MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

            # Finalize Losses
            D_loss = D_loss_temp
            G_loss = G_loss_temp + alpha * MSE_loss

        gradients_of_generator = gen_tape.gradient(G_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(D_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def train(ITERATIONS, NO, BATCH_SIZE):
        # Start Iterations
        for it in tqdm(range(ITERATIONS)):
            # Sample batch
            batch_idx = sample_batch_index(NO, BATCH_SIZE)
            X_mb = norm_data_x[batch_idx, :]
            M_mb = data_m[batch_idx, :]
            # Sample random vectors
            Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
            # Sample hint vectors
            H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
            H_mb = M_mb * H_mb_temp

            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            # Feed N Run
            train_step(X=X_mb,M=M_mb,H=H_mb)



