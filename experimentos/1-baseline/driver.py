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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models import UNet
from utils import *
from tensorflow import set_random_seed

RANDOM_SEED = 42


def main():
    # Fixes a initial seed for randomness
    np.random.seed(RANDOM_SEED)
    set_random_seed(RANDOM_SEED)

    # Gets currect directory path
    cdir = os.getcwd()

    # Gets all files .jpg
    inputs = glob.glob(str(cdir)+"/../filtered_dataset/input/*.jpg")
    # Gets all files .png
    targets = glob.glob(str(cdir)+"/../filtered_dataset/target/*.png")

    # Sort paths
    inputs.sort()
    targets.sort()

    # Parameters
    epochs = 60
    batch_size = 1
    depth = 3
    loss_func = 'categorical_crossentropy'
    learning_rate = 1e-4
    opt = Adam(lr=learning_rate)

    X = []
    Y = []

    # Iterates through files and extract the patches for training, validation and testing
    # Training
    for i in range(0, len(inputs)):
        X.append(fix_size(check_input_rgb(plt.imread(inputs[i])), depth))
        Y.append(fix_size(plt.imread(targets[i]), depth))

    # Converts it to a numpy array
    X = np.array(X)
    Y = np.array(Y)

    # Converts targets to one-hot enconding
    Y = img_to_ohe(Y)

    # Shuffles both the inputs and targets set
    indexes = list(range(0, 20))
    np.random.shuffle(indexes)
    X = X[indexes]
    Y = Y[indexes]

    # Splits the dataset into training, validation and testing sets.
    X_train = X[:10]
    Y_train = Y[:10]
    X_val = X[10:15]
    Y_val = Y[10:15]

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
