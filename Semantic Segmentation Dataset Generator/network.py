import keras as K


def fcn_vgg16(input_shape):
    VGG16_input = K.layers.Input(input_shape,
                                 name='VGG16_input')
    VGG16_conv01 = K.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv01')(VGG16_input)
    VGG16_conv02 = K.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv02')(VGG16_conv01)

    VGG16_pool11 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool11')(VGG16_conv02)
    VGG16_conv11 = K.layers.Conv2D(filters=128,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv11')(VGG16_pool11)
    VGG16_conv12 = K.layers.Conv2D(filters=128,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv12')(VGG16_conv11)

    VGG16_pool21 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool21')(VGG16_conv12)
    VGG16_conv21 = K.layers.Conv2D(filters=256,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv21')(VGG16_pool21)
    VGG16_conv22 = K.layers.Conv2D(filters=256,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv22')(VGG16_conv21)
    VGG16_conv23 = K.layers.Conv2D(filters=256,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv23')(VGG16_conv22)
    VGG16_conv24 = K.layers.Conv2D(filters=256,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv24')(VGG16_conv23)

    VGG16_pool31 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool31')(VGG16_conv24)
    VGG16_conv31 = K.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv31')(VGG16_pool31)
    VGG16_conv32 = K.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv32')(VGG16_conv31)
    VGG16_conv33 = K.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv33')(VGG16_conv32)
    VGG16_conv34 = K.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv34')(VGG16_conv33)

    VGG16_pool41 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool41')(VGG16_conv34)
    VGG16_conv41 = K.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv41')(VGG16_pool41)
    VGG16_conv42 = K.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv42')(VGG16_conv41)
    VGG16_conv43 = K.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv43')(VGG16_conv42)
    VGG16_conv44 = K.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv44')(VGG16_conv43)

    VGG16_pool51 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool51')(VGG16_conv44)
    VGG16_conv51 = K.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv51')(VGG16_pool51)
    VGG16_conv52 = K.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv52')(VGG16_conv51)

    decoder_trconv01 = K.layers.Conv2DTranspose(filters=512,
                                                kernel_size=4,
                                                strides=2,
                                                padding='same',
                                                activation='relu',
                                                name='decoder_trconv01'
                                                )(VGG16_conv52)
    decoder_trconv01_concated = K.layers.Add(name='decoder_trconv01_concated')([VGG16_pool41,
                                                                                decoder_trconv01])

    decoder_trconv11 = K.layers.Conv2DTranspose(filters=256,
                                                kernel_size=4,
                                                strides=2,
                                                padding='same',
                                                activation='relu',
                                                name='decoder_trconv11'
                                                )(decoder_trconv01_concated)
    decoder_trconv11_concated = K.layers.Add(name='decoder_trconv11_concated')([VGG16_pool31,
                                                                                decoder_trconv11])

    decoder_trconv21 = K.layers.Conv2DTranspose(filters=128,
                                                kernel_size=4,
                                                strides=2,
                                                padding='same',
                                                activation='relu',
                                                name='decoder_trconv21'
                                                )(decoder_trconv11_concated)
    decoder_trconv21_concated = K.layers.Add(name='decoder_trconv21_concated')([VGG16_pool21,
                                                                                decoder_trconv21])

    decoder_trconv31 = K.layers.Conv2DTranspose(filters=64,
                                                kernel_size=4,
                                                strides=2,
                                                padding='same',
                                                activation='relu',
                                                name='decoder_trconv31'
                                                )(decoder_trconv21_concated)
    decoder_trconv31_concated = K.layers.Add(name='decoder_trconv31_concated')([VGG16_pool11,
                                                                                decoder_trconv31])

    decoder_trconv41 = K.layers.Conv2DTranspose(filters=67,
                                              kernel_size=4,
                                              strides=2,
                                              padding='same',
                                              name='decoder_trconv41'
                                              )(decoder_trconv31_concated)

    decoder_trconv41_softmax = K.layers.Softmax(axis=-1)(decoder_trconv41)

    fcn_model = K.Model(VGG16_input, decoder_trconv41_softmax)
    fcn_model.name = 'FCN_model'

    return fcn_model


def fcn_resnet():
    ...
    # TODO


def unet():
    ...
    # TODO


if __name__ == "__main__":
    fcn_vgg16((1024, 768, 3))
