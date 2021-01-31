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

    datasets = ["ds0", "ds1", "ds2", "ds3"]
    datasets_label = ["EvaLady", "AosugiruHaru",
                      "JijiBabaFight", "MariaSamaNihaNaisyo"]

    df = pd.DataFrame(
        columns=['Trained on', 'Tested on', 'Loss', 'Accuracy', 'Precision', 'Recall'])

    for m, ml in enumerate(datasets):
        model = UNet(depth)

        model.load_weights("unet_{0}.hdf5".format(ml))

        for d, ds in enumerate(datasets):
            if ds == m:
                pass

            # Gets all files .jpg
            inputs_train = glob.glob(
                str(cdir)+"/../../datasets/D1_"+ds+"/input/*.jpg")
            # Gets all files .png
            targets_train = glob.glob(
                str(cdir)+"/../../datasets/D1_"+ds+"/target/*.png")

            inputs_val = glob.glob(
                str(cdir)+"/../../datasets/TT_"+ds+"/input/*.jpg")
            # Gets all files .png
            targets_val = glob.glob(
                str(cdir)+"/../../datasets/TT_"+ds+"/target/*.png")

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
            for i, _ in enumerate(inputs_val):
                x = plt.imread(inputs_val[i])
                if len(x.shape) == 3:
                    x = x[:, :, 0]
                X_val.append(fix_size(x, depth))
                Y_val.append(fix_size(plt.imread(targets_val[i]), depth))

            X_val = img_to_normal(np.array(X_val)[..., np.newaxis])
            Y_val = img_to_ohe(np.array(Y_val))

            # Shuffles both the inputs and targets set
            indexes = list(range(0, len(inputs_val)))
            np.random.shuffle(indexes)
            X_val = X_val[indexes]
            Y_val = Y_val[indexes]
            inputs_val = np.array(inputs_val)[indexes]

            X_val2 = X_val[5:]
            Y_val2 = Y_val[5:]

            Y_pred = model.predict(X_val2)

            test_loss = loss_func(K.constant(
                Y_val2), K.constant(Y_pred)).numpy()

            Y_pred = ohe_to_img(Y_pred)
            Y_val2 = ohe_to_img(Y_val2)

            metrics = calc_metrics(Y_val2, Y_pred)
            df = df.append(pd.DataFrame(data={
                'Trained on': [datasets_label[m]], 'Tested on': [datasets_label[d]], 'Loss': [test_loss], 'Accuracy': [metrics['accuracy']], 'Precision': [metrics['precision']], 'Recall': [metrics['recall']]}))

            df.to_csv('results.csv', index=False)

    clear_session()


if __name__ == '__main__':
    main()
