from skimage.color import rgb2gray
from tensorflow import set_random_seed
from utils import *
from losses import *
from models import UNet
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import tensorflow.keras.backend as K
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

RANDOM_SEED = 42


def main():
    # Gets currect directory path
    cdir = os.getcwd()

    # Parameters
    epochs = 100
    batch_size = 1
    depth = 3
    loss_label = 'GDL'
    loss_func = generalized_dice_loss

    learning_rate = 1e-4

    df = pd.DataFrame(columns=['dataset', 'depth', 'loss_func', 'time elapsed during training', 'epochs',
                               'loss', 'val_loss', 'test_loss', 'test_acc', 'test_precision', 'test_recall'])

    datasets = ["ds0", "ds1", "ds2", "ds3"]
    datasets_label = ["EvaLady", "AosugiruHaru",
                      "JijiBabaFight", "MariaSamaNihaNaisyo"]

    for d, ds in enumerate(datasets):
        # Gets all files .jpg
        inputs_train = glob.glob(
            str(cdir)+"/../datasets/D1_"+ds+"/input/*.jpg")
        # Gets all files .png
        targets_train = glob.glob(
            str(cdir)+"/../datasets/D1_"+ds+"/target/*.png")

        inputs_val = glob.glob(str(cdir)+"/../datasets/TT_"+ds+"/input/*.jpg")
        # Gets all files .png
        targets_val = glob.glob(
            str(cdir)+"/../datasets/TT_"+ds+"/target/*.png")

        # Sort paths
        inputs_train.sort()
        targets_train.sort()
        inputs_val.sort()
        targets_val.sort()

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
                x = x[:, :, 0]
            X_train.append(fix_size(x, depth))
            Y_train.append(fix_size(plt.imread(targets_train[i]), depth))
        for i, _ in enumerate(inputs_val):
            x = plt.imread(inputs_val[i])
            if len(x.shape) == 3:
                x = x[:, :, 0]
            X_val.append(fix_size(x, depth))
            Y_val.append(fix_size(plt.imread(targets_val[i]), depth))

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

        mc = ModelCheckpoint(
            "unet_{0}.hdf5".format(ds), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        # Initializes model
        model = UNet(depth)
        model.compile(loss=loss_func, optimizer=opt)

        # Trains model
        start = time.time()
        history = model.fit(X_train, Y_train, validation_data=(
            X_val1, Y_val1), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[mc])
        end = time.time()

        # Plots some performance graphs
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        np.save('history_{0}.npy'.format(ds), history.history)

        clear_session()

        epochs_range = list(range(0, len(loss)))

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        plt.subplots_adjust(hspace=0.4)
        '''
        Loss 
        '''
        ax.plot(epochs_range[1:], loss[1:], label='Training loss')
        ax.plot(epochs_range[1:], val_loss[1:], label='Validation loss')
        ax.xaxis.set_ticks(np.arange(0, 101, 10))
        ax.yaxis.set_ticks(np.arange(0, 1, 0.1))
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Learning curve - {0}'.format(datasets_label[d]))
        ax.legend()

        fig.savefig('learning_curve_{0}.png'.format(ds))

        try:
            model = UNet(depth)

            model.load_weights("unet_{0}.hdf5".format(ds))

            Y_pred = model.predict(X_val2)

            test_loss = loss_func(K.constant(
                Y_val2), K.constant(Y_pred)).numpy()

            Y_pred = ohe_to_img(Y_pred)
            Y_val2 = ohe_to_img(Y_val2)

            metrics = calc_metrics(Y_val2, Y_pred)

            df2 = pd.DataFrame(data={'dataset': [datasets_label[d]], 'depth': [depth], 'loss_func': [loss_label],
                                     'time elapsed during training': [end-start], 'epochs': [len(loss)],
                                     'train_loss': [np.amin(loss)], 'val_loss': [np.amin(val_loss)],
                                     'test_loss': [test_loss], 'test_acc': [metrics['accuracy']],
                                     'test_precision': [metrics['precision']],
                                     'test_recall': [metrics['recall']]})
            df = df.append(df2)

            df.to_csv('results.csv', index=False)

        except:
            print('not possible to load model for {0}'.format(
                datasets_label[d]))
            pass


if __name__ == '__main__':
    main()
