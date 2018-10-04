import os
import pickle
import keras as K
import core.losses


class ImageProperties():
    def __init__(self):
        self.image_shape = (512, 640, 3)
        self.n_classes = 1


class TrainProperties():
    def __init__(self):
        self.experiment_name = 1
        self.is_new = False
        self.single_image = True
        self.label_type = 'edges'

        self.name = 'RCF'
        self.epochs = 10
        self.train_batch_size = 1
        self.validation_batch_size = 1

        self.learning_rate = 7e-3
        self.learning_rate_decay = 3e-5

        self.workers = 1
        self.image_batch_log = 100

        self.loss = core.losses.edge_detection_loss.edge_detection_loss
        self.optimizer = K.optimizers.Adam(lr=self.learning_rate,
                                           decay=self.learning_rate_decay)
        self.model_file = './log/model/' + self.name + '-' + \
                          str(self.experiment_name) + '.h5'
        self.csv_file = './log/csv/' + self.name + '-' + \
                        str(self.experiment_name) + '.csv'
        self.tensorboard_file = './log/tensorboard'


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
        self.names_file = os.path.join(self._512.dict, data_type + "/images.dll")
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
