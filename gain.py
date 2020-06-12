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

    def generator_loss(fake_output):
        return

    def discriminator_loss(real_output, fake_output):
        return

    generator_optimizer = tf.keras.optimizers.Adam(1e-4, name='GenOpt')
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, name='DiscOpt')

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    EPOCHS = 50
    BATCH_SIZE = 32
    noise_dim = 100
    num_examples_to_generate = 16

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)

            display.clear_output(wait=True)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
