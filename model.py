import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as kb
from utils import rmse_loss
from utils import renormalization, rounding
from utils import uniform_sampler, binary_sampler
import numpy as np
import pandas as pd


class DDBlock(tf.keras.layers.Layer):
    """
        A simple Dense - Activation - Dropout Building Block
    """

    def __init__(self, units, rate, activation=None, name='DDBlock'):
        super(DDBlock, self).__init__(name=name)
        self.dense = Dense(units=units, kernel_initializer="he_normal")
        if activation:
            self.act = Activation(activation)
        else:
            self.act = PReLU()
        self.dropout = Dropout(rate=rate)

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.act(x)
        x = self.dropout(x, training=training)
        return x


class FFN(tf.keras.Model):
    """
        A  Feed Forward Network used in VSP Paper
    """

    def __init__(self, name='FFN'):
        super(FFN, self).__init__(name=name)
        self.block_1 = DDBlock(units=256, activation='relu', rate=0.5)
        self.block_2 = DDBlock(units=256, activation='relu', rate=0.5)
        self.block_3 = DDBlock(units=256, activation='relu', rate=0.5)
        self.block_4 = DDBlock(units=256, activation='relu', rate=0.0)
        self.linear_output = Dense(units=1, activation='linear')
        return

    def __call__(self, q, training=None):
        signal = q
        signal = self.block_1(signal, training=training)
        signal = self.block_2(signal, training=training)
        signal = self.block_3(signal, training=training)
        signal = self.block_4(signal, training=training)
        x = self.linear_output(signal)
        return x


def create_truck_model():
    model = FFN()
    model.summary()
    return model

# Generator
# Data + Mask as inputs (Random noise is in missing components)
def create_generator(DIM):
    model = Sequential(name='Generator')
    model.add(Input(shape=(DIM * 2,), name='Generator_Input_Layer', dtype='float32'))
    model.add(Dense(units=DIM, activation='relu', name="Generator_Dense_Layer_1"))
    model.add(Dense(units=DIM, activation='relu', name="Generator_Dense_Layer_2"))
    model.add(Dense(units=DIM, activation='sigmoid', name="Generator_Output_Layer"))
    model.summary()
    return model


# Discriminator
# Concatenate Data and Hint
def create_discriminator(DIM):
    model = Sequential(name='Discriminator')
    model.add(Input(shape=(DIM * 2,), name='Discriminator_Input_Layer', dtype='float32'))
    model.add(Dense(units=DIM, activation='relu', name="Discriminator_Dense_Layer_1"))
    model.add(Dense(units=DIM, activation='relu', name="Discriminator_Dense_Layer_2"))
    model.add(Dense(units=DIM, activation='sigmoid', name="Discriminator_Output_Layer"))
    model.summary()
    return model


@tf.function
def train_step(generator, gen_opt, discriminator, disc_opt, X, M, H):
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
        G_loss = G_loss_temp + 100 * MSE_loss

    gradients_of_generator = gen_tape.gradient(G_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(D_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return D_loss, G_loss


def evaluation_step(generator, data_m, norm_data_x, data_x,
                    ori_data_x, normalizer):
    """
        The validation schema is absent in the original paper implementation
        We will use for convenience the RMSE Value that is used as a Metric
        to perform Early Stopping and monitor the During-Training Performance of the Model
    """
    Z_mb = uniform_sampler(0, 0.01, data_m.shape[0], data_m.shape[1])
    Z_mb = Z_mb.astype('float32')
    M_mb = data_m
    M_mb = M_mb.astype('float32')
    X_mb = norm_data_x
    X_mb = X_mb.astype('float32')
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data = generator.predict(tf.concat([X_mb.values, M_mb.values], axis=1))[0]
    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    imputed_data = normalizer.denormalize(imputed_data)

    # Rounding
    imputed_data_values = rounding(imputed_data.values, data_x.values)

    rmse = rmse_loss(ori_data_x.values, imputed_data_values, data_m.values)
    imputed_and_rounded_df_to_use_for_downstream_task = pd.DataFrame(data=imputed_data_values,
                                                                     columns=imputed_data.columns)
    return rmse, imputed_and_rounded_df_to_use_for_downstream_task


def prepare_train_pipeline(norm_train_x, data_m):
    """
    Prepares training Pipeline into a TF Dataset Format
    :param norm_train_x:
    :param data_m:
    :return:
    """
    # Perform all the data augmentation BEFORE the training Loop (ffs)
    rows, columns = norm_train_x.shape
    X_mb = norm_train_x.values
    M_mb = data_m.values

    # Sample random vectors
    Z_mb = uniform_sampler(0, 0.01, rows, columns)
    # Sample hint vectors
    H_mb_temp = binary_sampler(0.9, rows, columns)

    H_mb = M_mb * H_mb_temp

    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    tf_data = tf.data.Dataset.from_tensor_slices(
        (X_mb.astype('float32'), M_mb.astype('float32'),
         H_mb.astype('float32'))).shuffle(100000).batch(256)

    return tf_data


def train(ori_train_x, ori_test_x, norm_train_x, data_m, norm_test_x, data_m_test, train_x, test_x, normalizer,
          dataset_name):
    # Create Models
    generator = create_generator(DIM=norm_train_x.shape[1])
    discriminator = create_discriminator(DIM=norm_train_x.shape[1])

    # Create Optimizers
    gen_opt = tf.keras.optimizers.Adam(1e-3, name='GenOpt')
    disc_opt = tf.keras.optimizers.Adam(1e-3, name='DiscOpt')

    # Iteration Counter
    counter = 0

    # Dummy Initialization of the evaluation metric
    rmse_old = 500
    train_frame = None
    test_frame = None

    tf_data = prepare_train_pipeline(norm_train_x, data_m)
    for epoch in range(500):
        gen_running_avg = []
        disc_running_avg = []
        for X, M, H in tf_data:
            dl, gl = train_step(generator=generator, discriminator=discriminator, gen_opt=gen_opt, disc_opt=disc_opt,
                                X=X,
                                M=M, H=H)
            gen_running_avg.append(gl.numpy())
            disc_running_avg.append(dl.numpy())

        # On Epoch End
        #print('Generator Epoch Loss: ' + str(np.array(gen_running_avg).mean()) + ' Discriminator Epoch Loss: ' + str(
        #    np.array(disc_running_avg).mean()))

        #print(f"TRAIN RMSE:{train_rmse}")
        # On Epoch End
        rmse_new, test_frame_new = evaluation_step(generator=generator, data_m=data_m_test, norm_data_x=norm_test_x,
                                                   data_x=test_x,
                                                   ori_data_x=ori_test_x, normalizer=normalizer)
        #print(f"TEST RMSE:{rmse_new}")
        if rmse_new < rmse_old:
            rmse_old = rmse_new
            test_frame = test_frame_new
            # generator.save(f'./models/{dataset_name}.h5')
        else:
            counter += 1
        if counter > 5:
            return test_frame
