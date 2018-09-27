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
        self.csv_file = './log/csv/log.csv'
        self.tensorboard_file = './log/tensorboard'

        self.train_sample_image_dir = ...


class DataDictionaries():
    def __init__(self, dataset):
        if dataset == 'mapillary':
            self.train = Mapillary('training')
            self.validation = Mapillary('validation')


class Mapillary():
    def __init__(self, data_type):

        assert (data_type == 'training' or data_type == 'validation'), \
            "Parameter: data_type should be \'training\' or \'validation\'"

        self.base_dict = 'D:\\Mapillary'

        self.raw = self.Raw(self.base_dict, data_type)
        self._512 = self.C_512(self.base_dict, data_type)

        # load the validation names if exist, if not read the names and save them
        self.names_file = os.path.join(self._512.dict, data_type+"/images.dll")
        if os.path.exists(self.names_file):
            with open(self.names_file, 'rb') as f:
                self.imagenames = pickle.load(f)
        else:
            self.imagenames = os.listdir(self._512.images_dict)
            self.imagenames = [imagename.split('.')[0] for imagename in
                               self.imagenames]
            with open(self.names_file, 'wb') as f:
                pickle.dump(self.imagenames, f)

    class Raw():
        def __init__(self, base_dict, data_type):
            self.dict = os.path.join(base_dict, 'raw')
            self.images_dict = os.path.join(self.dict, data_type + '\\images')
            self.labels_dict = os.path.join(self.dict, data_type + '\\labels')
            self.edges_dict = os.path.join(self.dict, data_type + '\\edges')

    class C_512():
        def __init__(self, base_dict, data_type):
            self.dict = os.path.join(base_dict, '512')
            self.images_dict = os.path.join(self.dict, data_type + '\\images')
            self.labels_dict = os.path.join(self.dict, data_type + '\\labels')
            self.edges_dict = os.path.join(self.dict, data_type + '\\edges')
