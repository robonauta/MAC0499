import tensorflow as tf
import tensorflow.keras.backend as K


def dice_loss(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    num = y_true*y_pred
    dem = y_true+y_pred

    dl = 1-2*(K.sum(num))/(K.sum(dem))

    return dl


def generalized_dice_loss(y_true, y_pred):

    wc = 1/(K.pow(K.sum(y_true, axis=(0, 1, 2)), 2))

    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    product = y_true*y_pred
    addition = y_true+y_pred

    num = wc[0]*(product)[:,:,:,0] + wc[1]*(product)[:,:,:,1]
    dem = wc[0]*(addition)[:,:,:,0] + wc[1]*(addition)[:,:,:,1]
    gdl = 1-2*(K.sum(num))/(K.sum(dem))

    return gdl

def weighted_cross_entropy(y_true, y_pred):
    wc = []

    epsilon = K.epsilon()
    y_pred = y_pred + epsilon
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)

    '''
    freqs = K.sum(y_true, axis=(0, 1, 2))
    wc = tf.reduce_max(freqs)/freqs
    '''
    wc = [0.45, 0.55]
    wc = K.cast(wc, 'float32')

    cross_entropy = -y_true* K.log(y_pred)

    mean_per_class = K.mean(cross_entropy, axis=(0,1,2))

    weighted_loss = wc*mean_per_class

    return K.sum(weighted_loss)

def categorical_focal_loss(gamma=2.0, alpha=0.5):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(loss)
        return loss
    return focal_loss

