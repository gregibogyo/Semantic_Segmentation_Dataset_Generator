import data_generator
import network
import keras as K
import os
from config import ImageProperties, TrainProperties, DataDictionaries
import core

image_properties = ImageProperties()
train_properties = TrainProperties()
# data_dictionaries = DataDictionaries('mapillary')

if __name__ == "__main__":
    training_generator = data_generator.DataGenerator(data_type='training',
                                                      batch_size=train_properties.train_batch_size)
    validation_generator = data_generator.DataGenerator(data_type='validation',
                                                        batch_size=train_properties.validation_batch_size)

    model = network.fcn_vgg16(input_shape=image_properties.image_shape)
    loss = train_properties.loss

    optimizer = K.optimizers.Adam(lr=train_properties.learning_rate,
                                      decay=train_properties.learning_rate_decay)

    model.compile(optimizer=optimizer,
                  loss=loss)

    save_callback = K.callbacks.ModelCheckpoint(filepath=train_properties.model_file,
                                                    save_best_only=True)
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=train_properties.tensorboard_file,
                                                       histogram_freq=0,
                                                       write_graph=True,
                                                       write_images=True)
    # TODO: ImageSampleMakerCallback
    image_check_callback = core.callbacks.ImageCheckCallback()

    csv_callback = K.callbacks.CSVLogger(filename=train_properties.csv_file,
                                         append=True)

    if os.path.exists(train_properties.model_file):
        model = K.models.load_model(train_properties.model_file)

    model.fit_generator(epochs=train_properties.epochs,
                        generator=training_generator,
                        validation_data=validation_generator,
                        validation_steps=3,
                        use_multiprocessing=True,
                        callbacks=[save_callback,
                                   csv_callback,
                                   image_check_callback],
                        workers=1)
