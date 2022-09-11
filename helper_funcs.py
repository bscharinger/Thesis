import keras
import tensorflow as tf
import cc3d
import numpy as np
from scipy.ndimage import zoom

def jaccard_distance(y_true, y_pred):

    """ Calculates mean of Jaccard distance as a loss function """
    smooth = 100
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd = (1 - jac) * smooth
    return tf.reduce_mean(jd)


def dice_coef(y_true, y_pred, smooth=1):

    """
    Calculate dice coefficient
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = keras.backend.sum(keras.backend.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (keras.backend.sum(keras.backend.square(y_true),-1) + keras.backend.sum(keras.backend.square(y_pred),-1) + smooth)


def dice_coef_loss(y_true, y_pred):
    """
    Calculates the dice loss
    :param y_true: True label
    :param y_pred: Predicted Label
    :return: Dice loss
    """
    return 1-dice_coef(y_true, y_pred)

def neg_dice_coef(y_true, y_pred):
    return -dice_coef(y_true, y_pred)*100000

def aorta_cc3d(input):
    """
    Does a connected component analysis in an input image and outputs the second biggest
    connected component (Biggest one is background)
    :param input: Segmented input image
    :return: Second biggest connected component
    """
    conn_comps = cc3d.connected_components(input)
    labels_out = conn_comps.reshape((1, -1))
    labels_out = labels_out[0, :]
    label = np.unique(labels_out)
    hist, bin_edges = np.histogram(labels_out, bins=label)
    hist = np.ndarray.tolist(hist)
    hist_ = hist
    hist_ = np.array(hist_)
    hist.sort(reverse=True)
    idx = (hist_ == hist[1])
    idx = idx + 1 - 1
    idx_ = np.sum(idx * label[0:len(idx)])
    output = labels_out * (labels_out == idx_)
    return output

def resizing(data):
    """
    Resize a 3d Image to 256x256x128 voxels
    :param label:
    :return: resized data
    """
    a,b,c=data.shape
    resized_data = zoom(data,(256/a,256/b,128/c),order=2, mode='constant')
    return resized_data

def padding(array, shape):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    xx, yy, zz = shape
    he, wi, de = array.shape
    print(array.shape)
    a = (xx - he) // 2
    aa = xx - a - he
    print(a, aa)
    b = (yy - wi) // 2
    bb = yy - b - wi
    print(b, bb)

    return np.pad(array, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')

def z_padding(input):
    if input.shape[:2] == (256,256):
        pass
    else:
        input=input[:256,:256,:]
    z_size = 128-input.shape[2]%128
    pad = np.zeros(input.shape[0:2]+ (z_size,))
    return np.append(input, pad, axis=-1)

def aorta_id(labels_out):
	labels_out=labels_out.reshape((1,-1))
	labels_out=labels_out[0,:]
	label=np.unique(labels_out)
	hist, bin_edges=np.histogram(labels_out,bins=label)
	hist=np.ndarray.tolist(hist)
	hist_=hist
	hist_=np.array(hist_)
	hist.sort(reverse = True)
	idx=(hist_==hist[1])
	idx=idx+1-1
	idx_=np.sum(idx*label[0:len(idx)])
	print('idx',idx_)
	return idx_

