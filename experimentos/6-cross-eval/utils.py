import numpy as np
import cv2


def extract_inputs(img, dx=150, dy=150, x_offset=2, y_offset=2):
    """
    Extracts patches and also generates translated versions of them. 

    Parameters:
        img: image to be extracted the patches
        dx: patch width
        dy: patch height
        x_offset: amount of pixels to be translated on the horizontal
        y_offset: amount of pixels to be translated on the vertical

    Returns: 
        array of patches with dimensions (#patches, 3, dx, dy, 3)
    """
    # Image dimensions (height, width, dimensions)
    height, width = img.shape[:2]

    # Pads the images so that its size is multiple of the patch size
    if(height % dy != 0):
        img = np.pad(img, ((0, (height - height % dy)), (0, 0),
                           (0, 0)), 'constant', constant_values=(0))
        # Updates height with padding
        height += dy - height % dy
    if(width % dx != 0):
        img = np.pad(img, ((0, 0), (0, width - width % dx),
                           (0, 0)), 'constant', constant_values=(0))
        # Updates width with padding
        width += dx - width % dx

    # Defines the affine transformation matrices
    T1 = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    T2 = np.float32([[1, 0, -x_offset], [0, 1, -y_offset]])

    # Translates the image to the southeast and to the nortwest
    img_translated1 = cv2.warpAffine(img, T1, (width, height))
    img_translated2 = cv2.warpAffine(img, T2, (width, height))

    # Cuts the image in a grid-fashion
    patches = []
    for y in range(0, height, dy):
        for x in range(0, width, dx):
            p = img[y:y+dy, x:x+dx, :]
            p_t1 = img_translated1[y:y+dy, x:x+dx, :]
            p_t2 = img_translated1[y:y+dy, x:x+dx, :]
            patches.append([p, p_t1, p_t2])
    patches = np.array(patches)
    return patches


def extract_targets(img, dx=150, dy=150):
    """
    Extracts patches from the target images. 

    Parameters:
        img: image to be extracted the patches
        dx: patch width
        dy: patch height

    Returns: 
        array of patches with dimensions (#patches, dx, dy)
    """
    # Image dimensions (height, width, dimensions)
    height, width = img.shape[:2]

    # Pads the images so that its size is multiple of the patch size
    if(height % dy != 0):
        img = np.pad(img, ((0, (height - height % dy)), (0, 0)),
                     'constant', constant_values=(0))
        # Updates height with padding
        height += dy - height % dy
    if(width % dx != 0):
        img = np.pad(img, ((0, 0), (0, width - width % dx)),
                     'constant', constant_values=(0))
        # Updates width with padding
        width += dx - width % dx

    # Cuts the image in a grid-fashion
    patches = []
    for y in range(0, height, dy):
        for x in range(0, width, dx):
            p = img[y:y+dy, x:x+dx]
            patches.append(p)
    patches = np.array(patches)
    return patches


def img_to_ohe(batch_img):
    """
    Converts image batch to one-hot encoding batch.

    Parameters:
        batch_img: batch of images to be converted

    Returns: 
        batch of one-hot encoded images (height, width, 2)
    """
    ohe = np.zeros(
        (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], 2))
    ohe[:, :, :, 0] = batch_img == 0
    ohe[:, :, :, 1] = batch_img > 0
    ohe.astype(np.uint8)
    return ohe


def ohe_to_img(batch_ohe):
    """
    Converts one-hot encoding batch to a batch of binary image.

    Parameters:
        ohe: one-hot encoded batch of images to be converted

    Returns: 
        batch of binary image (height, width)
    """
    return np.argmax(batch_ohe, axis=3)


def img_to_normal(batch_img):
    """
    Converts image batch to normalized image batch. Each pixel will have values between 0 and 1. 

    Parameters:
        batch: batch of images to be converted

    Returns: 
        batch of one-hot encoded images (height, width, 2)
    """
    return (batch_img/255.0).astype(np.float64)


def normal_to_img(batch_normal):
    """
    Converts normalized image batch to RGB image batch. Each pixel will have values between 0 and 255. 

    Parameters:
        batch: batch of images to be converted

    Returns: 
        batch of one-hot encoded images (height, width, 2)
    """
    return (batch_normal*255.0).astype(np.uint8)


def calc_metrics(y_true, y_pred):
    """
    Calculates metrics based on the predicted outputs and the ground-truth.

    Parameters:
        y_true: ground-truth
        y_pred: predicted ouputs

    Returns: 
        Tuple with two arrays: the first with elementary measuresements
        and the second with composite measurements. 
    """
    tp = np.sum((y_true > 0)*(y_pred > 0))
    tn = np.sum((y_true == 0)*(y_pred == 0))
    fp = np.sum((y_true == 0)*(y_pred > 0))
    fn = np.sum((y_true > 0)*(y_pred == 0))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    metrics = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }
    return metrics


def fix_size(img, depth):
    """
    Converts image sizes to the next number divisible by 2^depth. This is to avoid
    the problems occured when scaling the image down and then up.

    Parameters:
        img: image read
        depth: depth of the U-Net model

    Returns: 
        image with the size fixed. 
    """
    den = 2**depth
    shape = np.array(img.shape[:2])
    new_shape = den * np.ceil(shape/den).astype(np.int)

    if new_shape[0] != shape[0] or new_shape[1] != shape[1]:
        if len(img.shape) == 3:
            return np.pad(img, ((0, new_shape[0]-shape[0]), (0, new_shape[1]-shape[1]), (0, 0)), 'constant', constant_values=(0))
        else:
            return np.pad(img, ((0, new_shape[0]-shape[0]), (0, new_shape[1]-shape[1])), 'constant', constant_values=(0))
    else:
        return img


def check_input_rgb(img):
    """
    Checks if the input has three channels. If not, creates them. This is a model requirement. 

    Parameters:
        img: image read

    Returns: 
        image with 3 color channels
    """
    if len(img.shape) == 2:
        return np.stack((img, img, img), axis=2)
    else:
        return img
