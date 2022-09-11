import math
import numpy as np
import nrrd
from tensorflow import keras

class CTA(keras.utils.Sequence):

    def __init__(self, x_path, y_path, batch_size):
        self.x, self.y = x_path, y_path
        self.batch_size = batch_size
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_img_paths = self.x[i:i + self.batch_size]
        batch_lab_paths = self.y[i:i + self.batch_size]
        im = np.zeros((self.batch_size,) + (256, 256, 128) + (1,), dtype="float32")
        la = np.zeros((self.batch_size,) + (256, 256, 128) + (1,), dtype="float32")
        j = 0
        for j, path in enumerate(batch_img_paths):
            data, data_h = nrrd.read(path)
            data = np.expand_dims(data, axis=3)
            im[j] = data
            im = np.array(im)
        for j, path in enumerate(batch_lab_paths):
            label, label_h = nrrd.read(path)
            label = np.expand_dims(label, axis=3)
            la[j] = label
            la = np.array(la)
        return im, la