import random
import numpy as np
import cv2
import keras as K
from PIL import Image
import shutil
import os
import datetime


class ImageCheckCallback(K.callbacks.Callback):
    def __init__(self,
                 data_dictionary,
                 n_batch_log=1000,
                 image_log_dir='./log/image_check/',
                 save_name='experiment',
                 label_type='labels',
                 single=False,
                 validation=True):
        super(ImageCheckCallback, self).__init__()
        self.data_dictionary = data_dictionary
        self.save_dir = image_log_dir
        self.label_type = label_type
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
            image_name = self.image_names[image_from][random.randint(0, len(self.image_names[image_from]) - 1)]

        new_file_name = str(self.epoch) + '-' + str(batch) + '-' + image_from

        # load the train/validation image and the label/edge
        if image_from == 'training':
            image_source = os.path.join(self.data_dictionary.train._512.images_dict, image_name) + '.jpg'
            if self.label_type == 'labels':
                label_source = os.path.join(self.data_dictionary.train._512.labels_dict, image_name) + '.png'
            elif self.label_type == 'edges':
                edge_source = os.path.join(self.data_dictionary.train._512.edges_dict, image_name) + '.png'
        elif image_from == 'validation':
            image_source = os.path.join(self.data_dictionary.validation._512.images_dict, image_name) + '.jpg'
            if self.label_type == 'labels':
                label_source = os.path.join(self.data_dictionary.validation._512.labels_dict, image_name) + '.png'
            elif self.label_type == 'edges':
                edge_source = os.path.join(self.data_dictionary.validation._512.edges_dict, image_name) + '.png'

        # preprocess the image
        image = (np.array(Image.open(image_source)).astype(np.float32) - 128.) / 128.

        # color and copy the labels / copy the edges
        if self.label_type == 'labels':
            label = np.floor(np.array(Image.open(label_source)).astype(np.float16) * 3.7).astype(np.uint8)
            label = cv2.applyColorMap(label, cv2.COLORMAP_HSV)
            cv2.imwrite(os.path.join(self.current_save_dict, new_file_name + '.png'), label)
        elif self.label_type == 'edges':
            shutil.copy(edge_source, self.current_save_dict)
            dst_file = os.path.join(self.current_save_dict, image_name + '.png')
            new_dst_file_name = os.path.join(self.current_save_dict, new_file_name + '.png')
            os.rename(dst_file, new_dst_file_name)

        # copy the image
        shutil.copy(image_source, self.current_save_dict)

        # rename the image
        dst_file = os.path.join(self.current_save_dict, image_name + '.jpg')
        new_dst_file_name = os.path.join(self.current_save_dict, new_file_name + '.jpg')
        os.rename(dst_file, new_dst_file_name)

        return image

    def on_batch_end(self, batch, logs=None):
        if batch % self.n_batch_log == 0:
            train_image = np.expand_dims(self.load_image(image_from='training', batch=batch), axis=0)
            train_predict = np.array(self.model.predict(train_image)).astype(np.float32)[0]
            if self.label_type == 'labels':
                train_predict = np.argmax(train_predict, axis=-1)
                train_predict = np.floor(train_predict.astype(np.float32) * 3.7).astype(np.uint8)
                train_predict = cv2.applyColorMap(train_predict, cv2.COLORMAP_HSV)
            elif self.label_type == 'edges':
                train_predict = np.floor(train_predict.astype(np.float32) * 255.).astype(np.uint8)

            predicted_train_file_name = str(self.epoch) + '-' + str(batch) + '-train-predict'
            cv2.imwrite(os.path.join(self.current_save_dict, predicted_train_file_name + '.png'), train_predict)

            if self.validation:
                validation_image = np.expand_dims(self.load_image(image_from='validation', batch=batch), axis=0)
                validation_predict = np.array(self.model.predict(validation_image)).astype(np.float16)[0]
                if self.label_type == 'labels':
                    validation_predict = np.argmax(validation_predict, axis=-1)
                    validation_predict = np.floor(validation_predict.astype(np.float32) * 3.7).astype(np.uint8)
                    validation_predict = cv2.applyColorMap(validation_predict, cv2.COLORMAP_HSV)
                elif self.label_type == 'edges':
                    validation_predict = np.floor(validation_predict.astype(np.float32) * 255.).astype(np.uint8)

                predicted_valid_file_name = str(self.epoch) + '-' + str(batch) + '-valid-predict'
                cv2.imwrite(os.path.join(self.current_save_dict, predicted_valid_file_name + '.png'),
                            validation_predict)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = self.epoch + 1


if __name__ == "__main__":
    image_check_callback = ImageCheckCallback()
    image_check_callback.on_batch_end(100, 100)
