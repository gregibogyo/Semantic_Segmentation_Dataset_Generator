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

    config = K.backend.tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = K.backend.tf.Session(config=config)
    config.gpu_options.visible_device_list = "0"
    K.backend.set_session(sess)

    if os.path.exists(train_properties.vgg16_model_file) and not train_properties.is_new:
        vgg16_model = K.models.load_model(train_properties.vgg16_model_file, compile=False)
        print('Model %s saved at %s loaded' % (vgg16_model.name, train_properties.vgg16_model_file))
    else:
        vgg16_model = network.fcn_vgg16(input_shape=image_properties.image_shape)

    if os.path.exists(train_properties.rfc_model_file) and not train_properties.is_new:
        rfc_model = K.models.load_model(train_properties.rfc_model_file, compile=False)
        print('Model %s saved at %s loaded' % (rfc_model.name, train_properties.rfc_model_file))
    else:
        rfc_model = network.rcf(input_shape=image_properties.image_shape)

    conv_crf_model = network.conv_crf_rnn(input_shape=[vgg16_model.input_shape,
                                                       vgg16_model.output_shape,
                                                       rfc_model.output_shape])

    if os.path.exists(train_properties.model_file) and not train_properties.is_new:
        full_model = K.models.load_model(train_properties.model_file, compile=False)
        print('Model %s saved at %s loaded' % (full_model.name, train_properties.model_file))
    else:
        full_model = network.full_network(label_model=vgg16_model,
                                          edge_model=rfc_model,
                                          crf_model=conv_crf_model)

    full_model.compile(optimizer=train_properties.optimizer,
                       loss=train_properties.loss,
                       metrics=['acc'])

    save_callback = core.callbacks.ModelBatchCheckpoint(N=500,
                                                        filepath=train_properties.model_file,
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
