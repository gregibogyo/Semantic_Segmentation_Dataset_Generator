import data_generator
import network
import keras

if __name__ == "__main__":
    training_generator = data_generator.DataGenerator(data_type='training', batch_size=2)
    validation_generator = data_generator.DataGenerator(data_type='validation', batch_size=1)

    model = network.FCN_VGG16(input_shape=(512, 640, 3))
    loss = keras.losses.categorical_crossentropy

    optimizer = keras.optimizers.Adam(lr=1e-5, decay=1e-7)

    model.compile(optimizer=optimizer,
                  loss=loss)

    save_callback = keras.callbacks.ModelCheckpoint(filepath='./log/model.h5',
                                                    save_best_only=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./tensorboard',
                                                       histogram_freq=0,
                                                       write_graph=True,
                                                       write_images=True)

    model.fit_generator(epochs=20,
                        generator=training_generator,
                        validation_data=validation_generator,
                        validation_steps=1,
                        use_multiprocessing=True,
                        callbacks=[save_callback],
                        workers=1)
