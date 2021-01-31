import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

def UNet(depth=5):
    inputs=Input(shape=(None, None, 3))
    conv = []
    pool = []

    # Contracting
    conv.append(inputs)
    pool.append(inputs)
    for i in range(1, depth):
        conv.append(Conv2D(64*(2**(i-1)), (3, 3), activation='relu',
                           data_format='channels_last', padding='same')(pool[i-1]))
        conv[i] = Conv2D(64*(2**(i-1)), (3, 3), activation='relu',
                         data_format='channels_last', padding='same')(conv[i])
        pool.append(MaxPooling2D(pool_size=(2, 2))(conv[i]))

    conv.append(Conv2D(64*(2**(depth-1)), (3, 3), activation='relu',
                       data_format='channels_last', padding='same')(pool[depth-1]))
    conv[depth] = Conv2D(64*(2**(depth-1)), (3, 3), activation='relu',
                         data_format='channels_last', padding='same')(conv[depth])

    # Expansive
    for i in range(depth-1, 0, -1):
        conv.append(UpSampling2D(size=(2, 2))(conv[-1]))
        conv[-1] = Conv2D(64*(2**(i-1)), (2, 2), activation='relu',
                          data_format='channels_last', padding='same')(conv[-1])
        conv[-1] = concatenate([conv[i], conv[-1]], axis=3)
        conv[-1] = Conv2D(64*(2**(i-1)), (3, 3), activation='relu',
                          data_format='channels_last', padding='same')(conv[-1])
        conv[-1] = Conv2D(64*(2**(i-1)), (3, 3), activation='relu',
                          data_format='channels_last', padding='same')(conv[-1])

    conv.append(Conv2D(2, (1, 1), activation='softmax',
                       data_format='channels_last', padding='same')(conv[-1]))

    model = Model(inputs=inputs, outputs=conv[-1])

    return model