import keras
import numpy as np
import cv2
import os
import pickle
from PIL import Image
import tensorflow as tf
from imgaug import augmenters as iaa
import time

class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 data_type='training',
                 label_type='labels',
                 batch_size=1,
                 dim=(512, 640),
                 n_channels=3,
                 n_classes=67,
                 shuffle=True,
                 single=False):
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.single = single

        assert (data_type == 'training' or data_type == 'validation'), \
            "Parameter: data_type should be \'training\' or \'validation\'"
        assert (label_type == 'labels' or data_type == 'edges'), \
            "Parameter: label_type should be \'labels\' or \'edges\'"

        self.mapillary_dict = "D:\\Mapillary\\512\\" + data_type

        self.image_path = os.path.join(self.mapillary_dict, "images")
        assert (os.path.exists(self.image_path)), \
            "Image path: %s not exists" % self.image_path

        self. labels_path = os.path.join(self.mapillary_dict, label_type)
        assert (os.path.exists(self.labels_path)), \
            "%s path: %s not exists" % (label_type.title(), self.labels_path)

        self.seq = iaa.Sequential([
             iaa.Fliplr(0.5),
             iaa.Superpixels(p_replace=(0.0, 0.005), n_segments=(4, 8)),
             iaa.Grayscale(alpha=(0.0, 0.5)),
             iaa.Add((-10, 10), per_channel=0.2),
             iaa.AddElementwise((-5, 5), per_channel=0.2),
             iaa.ContrastNormalization((0.7, 1.3), per_channel=0.2),
             iaa.GaussianBlur(sigma=(0, 0.7)),
             # iaa.PiecewiseAffine(scale=(0.0, 0.004))
        ])

        self.IDs = self.load_IDs()

        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.IDs)) / self.batch_size)

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        if self.single:
            list_IDs_temp = [self.IDs[0] for k in range(self.batch_size)]
        else:
            list_IDs_temp = [self.IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_IDs(self):
        # load the names if exist, if not read the names and save them
        mapillary_names = os.path.join(self.mapillary_dict, "images.dll")
        if os._exists(mapillary_names):
            with open(mapillary_names, 'rb') as f:
                imagenames = pickle.load(f)
        else:
            imagenames = os.listdir(self.image_path)
            imagenames = [imagename.split('.')[0] for imagename in imagenames]
            with open(mapillary_names, 'wb') as f:
                pickle.dump(imagenames, f)

        return imagenames

    def __data_generation(self, list_IDs_temp):
        # Generate the data
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_classes))

        # start = time.time()

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i, ] = np.array(Image.open(os.path.join(self.image_path, ID) + '.jpg'))
            Ytemp = np.array(Image.open(os.path.join(self.labels_path, ID) + '.png'))

            # make one_hot from Y
            Ytemp = Ytemp.astype(np.uint8)
            n, m = Ytemp.shape
            k = 67
            Lhot = np.zeros((n * m, k))  # empty, flat array
            Lhot[np.arange(n * m), Ytemp.flatten()] = 1  # one-hot encoding for 1D

            Y[i, ] = Lhot.reshape(n, m, k)  # reshaping back to 3D tensor

        X = X.astype(np.uint8)
        X = (self.seq.augment_images(X).astype(np.float32)-128.)/128.

        # Y = tf.one_hot(Y, depth=self.n_classes)

        # cv2.imshow('asd', cv2.cvtColor(X[0], cv2.COLOR_RGB2BGR))
        # cv2.imshow('asd', Y[0, :, :, 27])
        # cv2.waitKey(0)

        # end = time.time()
        # print(end-start)

        return X, Y

if __name__=="__main__":
    datagen = DataGenerator()
    datagen.__getitem__(0)