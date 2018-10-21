import data_generator
import network
import keras as K
import os
from config import ImageProperties, TrainProperties, DataDictionaries
import core

image_properties = ImageProperties()
train_properties = TrainProperties()

if __name__ == "__main__":
    training_generator = data_generator.DataGenerator(data_type='training',
                                                      label_type=train_properties.label_type,
                                                      batch_size=train_properties.train_batch_size,
                                                      dim=image_properties.image_shape,
                                                      n_classes=image_properties.n_classes,
                                                      single=train_properties.single_image)
    validation_generator = data_generator.DataGenerator(data_type='validation',
                                                        label_type=train_properties.label_type,
                                                        batch_size=train_properties.validation_batch_size,
                                                        dim=image_properties.image_shape,
                                                        n_classes=image_properties.n_classes,
                                                        single=train_properties.single_image)

    if os.path.exists(train_properties.vgg16_model_file) and not train_properties.is_new:
        vgg16_model = K.models.load_model(train_properties.vgg16_model_file, compile=False)
        print('Model %s saved at %s loaded' % (vgg16_model.name, train_properties.vgg16_model_file))
    elif train_properties.network_name == 'RCF':
        # model = network.fcn_vgg16(input_shape=image_properties.image_shape)
        rfc_model = network.rcf(input_shape=image_properties.image_shape)
    elif train_properties.network_name == 'FCN-VGG16':
        vgg16_model = network.fcn_vgg16(input_shape=image_properties.image_shape)
    elif train_properties.network_name == 'FCN-VGG16_Conv-CRFRNN':
        vgg16_model = network.fcn_vgg16(input_shape=image_properties.image_shape)
    else:
        raise AssertionError('There is no model file in %s or the network called \'%s\' network is not available' \
                             % (train_properties.model_file, train_properties.network_name))

    # vgg16_model.compile(optimizer=train_properties.optimizer,
    #                     loss=train_properties.loss)

    conv_crf_rnn_model = network.conv_crf_rnn([vgg16_model.output_shape, vgg16_model.input_shape])

    full_model = network.full_network(vgg16_model, conv_crf_rnn_model)

    full_model.compile(optimizer=train_properties.optimizer,
                        loss=train_properties.loss)

    if not os.path.exists(train_properties.model_dict):
        os.mkdir(train_properties.model_dict)

    save_callback = K.callbacks.ModelCheckpoint(filepath=train_properties.model_file,
                                                save_best_only=True)
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=train_properties.tensorboard_file,
                                                   histogram_freq=0,
                                                   write_graph=True,
                                                   write_images=True)
    image_check_callback = core.callbacks.ImageCheckCallback(data_dictionary=DataDictionaries('mapillary'),
                                                             save_name=train_properties.network_name,
                                                             n_batch_log=train_properties.image_batch_log,
                                                             label_type=train_properties.label_type,
                                                             single=train_properties.single_image,
                                                             validation=train_properties.use_validation)
    if not os.path.exists(train_properties.csv_dict):
        os.mkdir(train_properties.csv_dict)

    csv_callback = K.callbacks.CSVLogger(filename=train_properties.csv_file,
                                         append=True)

    full_model.fit_generator(epochs=train_properties.epochs,
                             generator=training_generator,
                             validation_data=validation_generator,
                             validation_steps=3,
                             use_multiprocessing=True,
                             callbacks=[save_callback,
                                        csv_callback,
                                        image_check_callback],
                             workers=train_properties.workers)
