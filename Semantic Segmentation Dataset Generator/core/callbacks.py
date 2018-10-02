import random
import numpy as np
import cv2
import keras as K
from PIL import Image
import shutil
from config import DataDictionaries
import os
import datetime


class ImageCheckCallback(K.callbacks.Callback):
    def __init__(self,
                 n_batch_log=1000,
                 image_log_dir='./log/image_check/',
                 save_name='experiment',
                 single=False,
                 validation=True):
        super(ImageCheckCallback, self).__init__()
        self.data_dictionary = DataDictionaries('mapillary')
        self.save_dir = image_log_dir
        self.save_name = save_name
        self.n_batch_log = n_batch_log
        self.image_names = {'training': self.data_dictionary.train.imagenames,
                            'validation': self.data_dictionary.validation.imagenames}
        self.make_imcheck_dict()
        self.current_save_dict = self.make_current_save_dict()
        self.single = single
        self.validation = validation
        self.epoch = 1

    def make_imcheck_dict(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def make_current_save_dict(self):
        current_save_dict = self.save_dir + self.save_name + '_' + \
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "_").replace(":", "-")
        os.makedirs(current_save_dict)
        return current_save_dict

    def load_image(self, image_from, batch):
        if self.single:
            image_name = '__CRyFzoDOXn6unQ6a3DnQ'
        else:
            image_name = self.image_names[image_from][random.randint(0, len(self.image_names[image_from])-1)]

        new_file_name = str(self.epoch) + '-' + str(batch) + '-' + image_from

        if image_from == 'training':
            image_source = os.path.join(self.data_dictionary.train._512.images_dict, image_name) + '.jpg'
            label_source = os.path.join(self.data_dictionary.train._512.labels_dict, image_name) + '.png'
        elif image_from == 'validation':
            image_source = os.path.join(self.data_dictionary.validation._512.images_dict, image_name) + '.jpg'
            label_source = os.path.join(self.data_dictionary.validation._512.labels_dict, image_name) + '.png'

        image = (np.array(Image.open(image_source)).astype(np.float32) - 128.) / 128.

        label = np.floor(np.array(Image.open(label_source)).astype(np.float16) * 3.7).astype(np.uint8)
        label = cv2.applyColorMap(label, cv2.COLORMAP_HSV)
        cv2.imwrite(os.path.join(self.current_save_dict, new_file_name + '.png'), label)

        shutil.copy(image_source, self.current_save_dict)

        dst_file = os.path.join(self.current_save_dict, image_name + '.jpg')
        new_dst_file_name = os.path.join(self.current_save_dict, new_file_name + '.jpg')
        os.rename(dst_file, new_dst_file_name)

        return image

    def on_batch_end(self, batch, logs=None):
        if batch % self.n_batch_log == 0:
            train_image = np.expand_dims(self.load_image(image_from='training', batch=batch), axis=0)
            train_predict = np.argmax(np.array(self.model.predict(train_image)).astype(np.float16)[0],
                                      axis=-1)
            predicted_train_file_name = str(self.epoch) + '-' + str(batch) + '-train-predict'
            label = np.floor(train_predict.astype(np.float16) * 3.7).astype(np.uint8)
            label = cv2.applyColorMap(label, cv2.COLORMAP_HSV)
            cv2.imwrite(os.path.join(self.current_save_dict, predicted_train_file_name + '.png'), label)

            if self.validation:
                validation_image = np.expand_dims(self.load_image(image_from='validation', batch=batch), axis=0)
                validation_predict = np.argmax(np.array(self.model.predict(validation_image)).astype(np.float16)[0],
                                               axis=-1)
                predicted_valid_file_name = str(self.epoch) + '-' + str(batch) + '-valid-predict'
                label = np.floor(validation_predict.astype(np.float16) * 3.7).astype(np.uint8)
                label = cv2.applyColorMap(label, cv2.COLORMAP_HSV)
                cv2.imwrite(os.path.join(self.current_save_dict, predicted_valid_file_name + '.png'), label)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = self.epoch+1


if __name__ == "__main__":
    image_check_callback = ImageCheckCallback()
    image_check_callback.on_batch_end(100, 100)
