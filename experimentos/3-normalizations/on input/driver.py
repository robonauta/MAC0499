import os
import sys
import cv2
import glob
import time
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.backend import clear_session
from models import UNet
from utils import *
from tensorflow import set_random_seed

RANDOM_SEED = 42


def main():
    # Gets currect directory path
    cdir = os.getcwd()

    # Gets all files .jpg
    inputs_train = glob.glob(
        str(cdir)+"../../subconjuntos/D1_ds0/inputs/*.jpg")
    # Gets all files .png
    targets_train = glob.glob(
        str(cdir)+"../../subconjuntos/D1_ds0/target/*.png")

    inputs_val = glob.glob(str(cdir)+"../../subconjuntos/TT_ds0/input/*.jpg")
    # Gets all files .png
    targets_val = glob.glob(str(cdir)+"../../subconjuntos/TT_ds0/target/*.png")

    # Sort paths
    inputs_train.sort()
    targets_train.sort()
    inputs_val.sort()
    targets_val.sort()

    # Parameters
    epochs = 100
    batch_size = 1
    depths = [1,2,3,4,5]
    loss_func = CategoricalCrossentropy()
    learning_rate = 1e-4
    opt = Adam(lr=learning_rate)

    df = pd.DataFrame(columns=['depth', 'loss_func', 'time elapsed during training', 'epochs',
                               'loss', 'val_loss', 'test_loss', 'test acc', 'test precision', 'test_recall'])

    fig, ax = plt.subplots(2, 1, figsize=(15, 15))

    for depth in depths:
        opt = Adam(lr=learning_rate)

        # Fixes a initial seed for randomness
        np.random.seed(RANDOM_SEED)
        set_random_seed(RANDOM_SEED)

        X_train = []
        Y_train = []
        X_val = []
        Y_val = []
        # Iterates through files and extract the patches for training, validation and testing

        for i, _ in enumerate(inputs_train): 
            x = plt.imread(inputs_train[i])
            if len(x.shape) == 3:
                x = x[:,:,0]
            X_train.append(fix_size(x, depth))
            Y_train.append(fix_size(plt.imread(targets_train[i]),depth))
        for i, _ in enumerate(inputs_val): 
            x = plt.imread(inputs_val[i])
            if len(x.shape) == 3:
                x = x[:,:,0]
            X_val.append(fix_size(x, depth))
            Y_val.append(fix_size(plt.imread(targets_val[i]),depth))

        X_train = img_to_normal(np.array(X_train)[..., np.newaxis])
        Y_train = img_to_ohe(np.array(Y_train))

        X_val = img_to_normal(np.array(X_val)[..., np.newaxis])
        Y_val = img_to_ohe(np.array(Y_val))
        
        # Shuffles both the inputs and targets set
        indexes = list(range(0, len(inputs_val)))
        np.random.shuffle(indexes)
        X_val = X_val[indexes]
        Y_val = Y_val[indexes]

        X_val1 = X_val[:5]
        Y_val1 = Y_val[:5]
        X_val2 = X_val[5:10]
        Y_val2 = Y_val[5:10]

        mc = ModelCheckpoint("unet_{0}.hdf5".format(depth), monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        # Initializes model
        model = UNet(depth)
        model.compile(loss=loss_func,
                      optimizer=opt)

        # Trains models
        start = time.time()
        history = model.fit(X_train, Y_train, validation_data=(X_val1, Y_val1), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[mc])
        end = time.time()

        # Plots some performance graphs
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        np.save('history_{0}.npy'.format(depth), history.history)

        clear_session()
        
        epochs_range = list(range(0, len(loss)))

        ax[0].plot(epochs_range[1:], loss[1:], label='Depth {0}'.format(depth))
        ax[0].xaxis.set_ticks(np.arange(0, 101, 5))
        ax[0].yaxis.set_ticks(np.arange(0, 1, 0.1))
        ax[0].set_title('Training loss - UNet')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        ax[1].plot(epochs_range[1:], val_loss[1:], label='Depth {0}'.format(depth))
        ax[1].xaxis.set_ticks(np.arange(0, 101, 5))
        ax[1].yaxis.set_ticks(np.arange(0, 1, 0.1))
        ax[1].set_title('Validation loss - UNet')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].legend()

        fig.savefig('learning_curve.png')
              
        model = UNet(depth)

        model.load_weights("unet_{0}.hdf5".format(depth))

        Y_pred = model.predict(X_val2)

        Y_pred = ohe_to_img(Y_pred)
        Y_val2 = ohe_to_img(Y_val2)

        metrics = calc_metrics(Y_val2, Y_pred)

        test_loss = loss_func(K.constant(Y_val2), K.constant(Y_pred)).numpy()

        df2 = pd.DataFrame(data={'depth': [depth], 'loss_func': ["CE"],
                                 'time elapsed during training': [end-start], 'epochs': [len(loss)],
                                 'loss': [np.amin(loss)], 'val_loss': [np.amin(val_loss)],
                                 'test_loss': [test_loss], 'test acc': [metrics['accuracy']],
                                 'test precision': [metrics['precision']],
                                 'test_recall': [metrics['recall']]})
        df = df.append(df2)

        df.to_csv('results.csv', index=False)

if __name__ == '__main__':
    main()
