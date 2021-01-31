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
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models import UNet
from utils import *
from tensorflow import set_random_seed

RANDOM_SEED = 42


def main():
    # Fixes a initial seed for randomness
    np.random.seed(RANDOM_SEED)
    set_random_seed(RANDOM_SEED)

    epochs = 100
    batch_size = 1
    depths = [1, 2, 3, 4, 5]
    loss_func = CategoricalCrossentropy()
    learning_rate = 1e-4
    opt = Adam(lr=learning_rate)
    depth = 3

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

    X_train = np.array(X_train)[..., np.newaxis]
    Y_train = img_to_ohe(np.array(Y_train))

    X_val = np.array(X_val)[..., np.newaxis]
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
        "unet.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    # Initializes model
    model = UNet(depth)
    model.compile(loss=loss_func,
                  optimizer=opt,
                  metrics=['accuracy', Precision(), Recall()])

    # Trains model
    start = time.time()
    history = model.fit(X_train, Y_train,
                        validation_data=(X_val, Y_val),
                        epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[mc, es])
    end = time.time()

    # Plots some performance graphs
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    precision = history.history['precision']
    val_precision = history.history['val_precision']
    recall = history.history['recall']
    val_recall = history.history['val_recall']

    df = pd.DataFrame(data={'acc': [np.amax(acc)], 'val_acc': [np.amax(val_acc)], 'loss': [np.amin(loss)], 'val_loss': [np.amin(
        val_loss)], 'precision': [np.amax(precision)], 'val_precision': [np.amax(val_precision)], 'recall': [np.amax(recall)], 'val_recall': [np.amax(val_recall)]})

    epochs_range = list(range(0, len(acc)))

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0][0].plot(epochs_range, loss, 'bo', label='Training loss')
    ax[0][0].plot(epochs_range, val_loss, 'b', label='Validation loss')
    ax[0][0].set_title('Training and validation loss - UNet')
    ax[0][0].set_xlabel('Epochs')
    ax[0][0].set_ylabel('Loss')
    ax[0][0].legend()

    ax[0][1].plot(epochs_range, acc, 'bo', label='Training acc')
    ax[0][1].plot(epochs_range, val_acc, 'b', label='Validation acc')
    ax[0][1].set_title('Training and validation accuracy - UNet')
    ax[0][1].set_xlabel('Epochs')
    ax[0][1].set_ylabel('Accuracy')
    ax[0][1].legend()

    ax[1][0].plot(epochs_range, precision, 'bo', label='Training precision')
    ax[1][0].plot(epochs_range, val_precision, 'b',
                  label='Validation precision')
    ax[1][0].set_title('Training and validation precision - UNet')
    ax[1][0].set_xlabel('Epochs')
    ax[1][0].set_ylabel('Precision')
    ax[1][0].legend()

    ax[1][1].plot(epochs_range, recall, 'bo', label='Training recall')
    ax[1][1].plot(epochs_range, val_recall, 'b', label='Validation recall')
    ax[1][1].set_title('Training and validation recall - UNet')
    ax[1][1].set_xlabel('Epochs')
    ax[1][1].set_ylabel('Recall')
    ax[1][1].legend()
    plt.subplots_adjust(hspace=0.5)
    fig.savefig('learning_curve.png')
    plt.clf()

    df.to_csv('results.csv')

if __name__ == '__main__':
    main()
