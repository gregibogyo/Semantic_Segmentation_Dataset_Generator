import data_generator
import network
import keras as K
import os
from config import ImageProperties, TrainProperties
import core

image_properties = ImageProperties()
train_properties = TrainProperties()

if __name__ == "__main__":
    training_generator = data_generator.DataGenerator(data_type='training',
                                                      batch_size=train_properties.train_batch_size,
                                                      single=False)
    validation_generator = data_generator.DataGenerator(data_type='validation',
                                                        batch_size=train_properties.validation_batch_size)

    loss = train_properties.loss

    optimizer = K.optimizers.Adam(lr=train_properties.learning_rate,
                                  decay=train_properties.learning_rate_decay)

    if os.path.exists(train_properties.model_file) and not train_properties.is_new:
        model = K.models.load_model(train_properties.model_file, compile=False)
        print('Model %s saved at %s loaded' % (model.name, train_properties.model_file))
    else:
        model = network.fcn_vgg16(input_shape=image_properties.image_shape)

    model.compile(optimizer=optimizer,
                  loss=loss)

    save_callback = K.callbacks.ModelCheckpoint(filepath=train_properties.model_file,
                                                save_best_only=True)
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=train_properties.tensorboard_file,
                                                   histogram_freq=0,
                                                   write_graph=True,
                                                   write_images=True)
    image_check_callback = core.callbacks.ImageCheckCallback(save_name=model.name,
                                                             n_batch_log=1000,
                                                             single=False,
                                                             validation=True)
    csv_callback = K.callbacks.CSVLogger(filename=train_properties.csv_file,
                                         append=True)

    model.fit_generator(epochs=train_properties.epochs,
                        generator=training_generator,
                        validation_data=validation_generator,
                        validation_steps=3,
                        use_multiprocessing=True,
                        callbacks=[save_callback,
                                   csv_callback,
                                   image_check_callback],
                        workers=train_properties.workers)
