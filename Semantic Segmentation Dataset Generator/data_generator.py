import keras
import numpy as np
import cv2
import os
import pickle
from PIL import Image
from imgaug import augmenters as iaa
import time
from config import DataDictionaries, Mapillary


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 data_type='training',
                 label_type='labels',
                 batch_size=1,
                 dim=(512, 640, 3),
                 n_classes=67,
                 shuffle=True,
                 single=False):
        assert (data_type == 'training' or data_type == 'validation'), \
            "Parameter: data_type should be \'training\' or \'validation\'"
        assert (label_type == 'labels' or label_type == 'edges'), \
            "Parameter: label_type should be \'labels\' or \'edges\'"

        self.batch_size = batch_size
        self.dim = (dim[0], dim[1])
        self.n_channels = dim[2]
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.single = single
        self.data_dictionaries = Mapillary(data_type)
        self.label_type = label_type

        seq_flip = iaa.Sequential([
            iaa.Fliplr(0.5)
        ], deterministic=True)

        self.seq_flip_deterministic = seq_flip.to_deterministic()

        self.seq = iaa.Sequential([
            iaa.Superpixels(p_replace=(0.0, 0.005), n_segments=(4, 8)),
            iaa.Grayscale(alpha=(0.0, 0.5)),
            iaa.Add((-10, 10), per_channel=0.2),
            iaa.AddElementwise((-5, 5), per_channel=0.2),
            iaa.ContrastNormalization((0.7, 1.3), per_channel=0.2),
            iaa.GaussianBlur(sigma=(0, 0.7)),
            iaa.PiecewiseAffine(scale=(0.0, 0.005))
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
            list_IDs_temp = ['__CRyFzoDOXn6unQ6a3DnQ' for k in range(self.batch_size)]
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
        return self.data_dictionaries.imagenames

    def __data_generation(self, list_IDs_temp):
        # Generate the data
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_classes))

        # start = time.time()

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.array(Image.open(os.path.join(self.data_dictionaries._512.images_dict, ID) + '.jpg'))
            if self.label_type == 'labels':
                Ytemp = np.array(Image.open(os.path.join(self.data_dictionaries._512.labels_dict, ID) + '.png'))

                # make one_hot from Y
                Ytemp = Ytemp.astype(np.uint8)

                n, m = Ytemp.shape
                k = self.n_classes
                Lhot = np.zeros((n * m, k))  # empty, flat array
                Lhot[np.arange(n * m), Ytemp.flatten()] = 1  # one-hot encoding for 1D

                Y[i,] = Lhot.reshape(n, m, k)  # reshaping back to 3D tensor
            elif self.label_type == 'edges':
                Ytemp = np.array(Image.open(os.path.join(self.data_dictionaries._512.edges_dict, ID) + '.png')). \
                            astype(np.float32) / 255.

                Y[i,] = np.expand_dims(Ytemp, axis=-1)

        # augment y
        Y = self.seq_flip_deterministic.augment_images(Y).astype(np.float32)

        # augment x
        X = X.astype(np.uint8)
        X = (self.seq_flip_deterministic.augment_images(X))
        X = (self.seq.augment_images(X).astype(np.float32) - 128.) / 128.

        # cv2.imshow('Image', cv2.cvtColor((X[0]+1.)/2., cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # Y_draw = (np.floor(np.squeeze(Y[0], axis=-1).astype(np.float32))*255.).astype(np.uint8)
        # # Y_draw = (np.floor(np.argmax(Y[0], axis=-1).astype(np.float32)) * 3.7).astype(np.uint8)
        # # Y_draw = cv2.applyColorMap(Y_draw, cv2.COLORMAP_HSV)
        # cv2.imshow('Label', Y_draw)
        # cv2.waitKey(0)

        # end = time.time()
        # print(end-start)

        return X, Y


if __name__ == "__main__":
    datagen = DataGenerator(label_type='edges',
                            n_classes=1,
                            single=True)
    datagen.__getitem__(0)
