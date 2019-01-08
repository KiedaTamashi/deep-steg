from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.engine.topology import Network
from keras.layers import *
from keras.models import Model
from utils import *
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.misc
from tqdm import *
# %matplotlib inline

# ==========global variable==============
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
IMG_SHAPE = (64, 64)
# Variable used to weight the losses of the secret and cover images (See paper for more details)
beta = 1.0




# ----------------model-----------------------

# Loss for reveal network
def rev_loss(s_true, s_pred):
    # Loss for reveal network is: beta * |S-S'|
    return beta * K.sum(K.square(s_true - s_pred))


# Loss for the full model, used for preparation and hidding networks
def full_loss(y_true, y_pred):
    # Loss for the full model is: |C-C'| + beta * |S-S'|
    s_true, c_true = y_true[..., 0:3], y_true[..., 3:6]
    s_pred, c_pred = y_pred[..., 0:3], y_pred[..., 3:6]

    s_loss = rev_loss(s_true, s_pred)
    c_loss = K.sum(K.square(c_true - c_pred))

    return s_loss + c_loss


# Returns the encoder as a Keras model, composed by Preparation and Hiding Networks.
def make_encoder(input_size):
    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))

    # Preparation Network
    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_prep0_3x3')(input_S)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_prep0_4x4')(input_S)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_prep0_5x5')(input_S)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_prep1_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_prep1_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_prep1_5x5')(x)
    x = concatenate([x3, x4, x5])

    x = concatenate([input_C, x])

    # Hiding network
    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_hid0_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_hid0_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_hid0_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_hid1_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_hid1_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_hid1_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_hid2_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_hid2_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_hid2_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_hid3_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_hid3_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_hid3_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_hid4_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_hid4_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_hid5_5x5')(x)
    x = concatenate([x3, x4, x5])

    output_Cprime = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='relu', name='output_C')(x)

    return Model(inputs=[input_S, input_C],
                 outputs=output_Cprime,
                 name='Encoder')


# Returns the decoder as a Keras model, composed by the Reveal Network
def make_decoder(input_size, fixed=False):
    # Reveal network
    reveal_input = Input(shape=(input_size))

    # Adding Gaussian noise with 0.01 standard deviation.
    input_with_noise = GaussianNoise(0.01, name='output_C_noise')(reveal_input)

    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_rev0_3x3')(input_with_noise)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_rev0_4x4')(input_with_noise)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_rev0_5x5')(input_with_noise)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_rev1_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_rev1_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_rev1_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_rev2_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_rev2_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_rev2_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_rev3_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_rev3_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_rev3_5x5')(x)
    x = concatenate([x3, x4, x5])

    x3 = Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', name='conv_rev4_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides=(1, 1), padding='same', activation='relu', name='conv_rev4_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides=(1, 1), padding='same', activation='relu', name='conv_rev5_5x5')(x)
    x = concatenate([x3, x4, x5])

    output_Sprime = Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='relu', name='output_S')(x)

    if not fixed:
        return Model(inputs=reveal_input,
                     outputs=output_Sprime,
                     name='Decoder')
    else:
        return Network(inputs=reveal_input,
                         outputs=output_Sprime,
                         name='DecoderFixed')


# Full model.
def make_model(input_size):
    input_S = Input(shape=(input_size))
    input_C = Input(shape=(input_size))

    encoder = make_encoder(input_size)

    decoder = make_decoder(input_size)
    decoder.compile(optimizer='adam', loss=rev_loss)
    decoder.trainable = False

    output_Cprime = encoder([input_S, input_C])
    output_Sprime = decoder(output_Cprime)

    autoencoder = Model(inputs=[input_S, input_C],
                        outputs=concatenate([output_Sprime, output_Cprime]))
    autoencoder.compile(optimizer='adam', loss=full_loss)

    return encoder, decoder, autoencoder

# ----------------model-----------------------

# ----------------train-----------------------
def train(input_S,input_C,NB_EPOCHS = 1000,BATCH_SIZE = 32, save_model = 'models/model.hdf5'):
    def lr_schedule(epoch_idx):
        if epoch_idx < 200:
            return 0.001
        elif epoch_idx < 400:
            return 0.0003
        elif epoch_idx < 600:
            return 0.0001
        else:
            return 0.00003

    encoder_model, reveal_model, autoencoder_model = make_model(input_S.shape[1:])
    m = input_S.shape[0]
    loss_history = []
    for epoch in range(NB_EPOCHS):
        np.random.shuffle(input_S)
        np.random.shuffle(input_C)

        t = tqdm(range(0, input_S.shape[0], BATCH_SIZE), mininterval=0)
        ae_loss = []
        rev_loss = []
        for idx in t:
            batch_S = input_S[idx:min(idx + BATCH_SIZE, m)]
            batch_C = input_C[idx:min(idx + BATCH_SIZE, m)]

            C_prime = encoder_model.predict([batch_S, batch_C])

            ae_loss.append(autoencoder_model.train_on_batch(x=[batch_S, batch_C],
                                                            y=np.concatenate((batch_S, batch_C), axis=3)))
            rev_loss.append(reveal_model.train_on_batch(x=C_prime,
                                                        y=batch_S))

            # Update learning rate
            K.set_value(autoencoder_model.optimizer.lr, lr_schedule(epoch))
            K.set_value(reveal_model.optimizer.lr, lr_schedule(epoch))

            t.set_description(
                'Epoch {} | Batch: {:3} of {}. Loss AE {:10.2f} | Loss Rev {:10.2f}'.format(epoch + 1, idx, m,
                                                                                            np.mean(ae_loss),
                                                                                            np.mean(rev_loss)))
        loss_history.append(np.mean(ae_loss))

    # Plot loss through epochs
    plt.plot(loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    # Save model
    autoencoder_model.save_weights(save_model)
# ----------------train-----------------------

def validation(input_S,input_C):

    encoder_model, reveal_model, autoencoder_model = make_model(input_S.shape[1:])

    # Load model
    model_test = 'models/weights_final.hdf5'
    autoencoder_model.load_weights(model_test)
    print("Load model: ",model_test)

    # Retrieve decoded predictions.
    decoded = autoencoder_model.predict([input_S, input_C])
    decoded_S, decoded_C = decoded[...,0:3], decoded[...,3:6]


    result_display(input_S,input_C,decoded_S,decoded_C,n=decoded_C.shape[0])
    iamge_save(decoded_S,decoded_C)
# Load dataset.
def load_data_preprocess(num_images_per_class_train, num_images_test, train_set_range = 200, option="train"):
    X_train_orig, X_test_orig = load_dataset_small(num_images_per_class_train, num_images_test, train_set_range)

    # Normalize image vectors.
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Print statistics.
    print("Number of training examples = " + str(X_train.shape[0]))
    print("Number of test examples = " + str(X_test.shape[0]))


    # We split training set into two halfs.
    # First half is used for training as secret images, second half for cover images.
    if option == "train":
        # S: secret image
        print("X_train shape: " + str(X_train.shape))  # Should be (train_size, 64, 64, 3).
        input_S = X_train[0:X_train.shape[0] // 2]
        # C: cover image
        input_C = X_train[X_train.shape[0] // 2:]
    else:
        print("X_test shape: " + str(X_test.shape))  # Should be (test_size, 64, 64, 3).
        input_S = X_test[0:X_test.shape[0] // 2]
        # C: cover image
        input_C = X_test[X_test.shape[0] // 2:]


    return input_S,input_C

def main():

    option = 'validation'

    # train_num_per_class  test   train_class
    # required: n%2==0, first 1/2 be secret, second 1/2 be cover
    if option == 'validation':
        input_S, input_C = load_data_preprocess(4, 2, 4, option='validation')
        validation(input_S,input_C)
    elif option == 'train':
        input_S, input_C = load_data_preprocess(100, 10, 100)
        train(input_S, input_C)

if __name__ =="__main__":
    main()