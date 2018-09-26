import os
import pickle
import keras as K


class ImageProperties():
    def __init__(self):
        self.image_shape = (512, 640, 3)
        self.n_classes = 67


class TrainProperties():
    def __init__(self):
        self.experiment_number = 1

        self.name = 'FCN_VGG16'
        self.epochs = 10
        self.train_batch_size = 1
        self.validation_batch_size = 1

        self.learning_rate = 1e-3
        self.learning_rate_decay = 1e-5

        self.loss = K.losses.categorical_crossentropy

        self.model_file = './log/model/' + self.name + str(self.experiment_number) + '.h5'
        self.csv_file ='./log/csv/log.csv'
        self.tensorboard_file = './log/tensorboard'

        self.train_sample_image_dir = ...


class DataDictionaries():
    def __init__(self):
        self.mapillary = Mapillary()


class Mapillary():
    def __init__(self):
        self.dict = 'D:\\Mapillary'

        self.raw_dict = os.path.join(self.dict, 'raw')
        self._512_dict = os.path.join(self.dict, '512')

        self.raw_training_images_dict = os.path.join(self.raw_dict, 'training\\images')
        self.raw_training_labels_dict = os.path.join(self.raw_dict, 'training\\labels')
        self.raw_training_edges_dict = os.path.join(self.raw_dict, 'training\\edges')
        self._512_training_images_dict = os.path.join(self._512_dict, 'training\\images')
        self._512_training_labels_dict = os.path.join(self._512_dict, 'training\\labels')
        self._512_training_edges_dict = os.path.join(self._512_dict, 'training\\edges')

        self.raw_validation_images_dict = os.path.join(self.raw_dict, 'validation\\images')
        self.raw_validation_labels_dict = os.path.join(self.raw_dict, 'validation\\labels')
        self.raw_validation_edges_dict = os.path.join(self.raw_dict, 'validation\\edges')
        self._512_validation_images_dict = os.path.join(self._512_dict, 'validation\\images')
        self._512_validation_labels_dict = os.path.join(self._512_dict, 'validation\\labels')
        self._512_validation_edges_dict = os.path.join(self._512_dict, 'validation\\edges')

        # load the training names if exist, if not read the names and save them
        self.training_names_file = os.path.join(self._512_dict, "images.dll")
        if os._exists(self.training_names_file):
            with open(self.training_names_file, 'rb') as f:
                self.training_imagenames = pickle.load(f)
        else:
            self.training_imagenames = os.listdir(self._512_training_images_dict)
            self.training_imagenames = [imagename.split('.')[0] for imagename in
                                        self.training_imagenames]
            with open(self.training_names_file, 'wb') as f:
                pickle.dump(self.training_imagenames, f)

        # load the validation names if exist, if not read the names and save them
        self.validation_names_file = os.path.join(self._512_dict, "images.dll")
        if os._exists(self.validation_names_file):
            with open(self.validation_names_file, 'rb') as f:
                self.validation_imagenames = pickle.load(f)
        else:
            self.validation_imagenames = os.listdir(self._512_validation_images_dict)
            self.validation_imagenames = [imagename.split('.')[0] for imagename in
                                          self.validation_imagenames]
            with open(self.validation_names_file, 'wb') as f:
                pickle.dump(self.validation_imagenames, f)
