import keras as K
from core import layers


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
    VGG16_dropout01 = K.layers.Dropout(0.5,
                                       name='VGG16_dropout01')(VGG16_conv02)

    VGG16_pool11 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool11')(VGG16_dropout01)
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
    VGG16_dropout11 = K.layers.Dropout(0.5,
                                       name='VGG16_dropout11')(VGG16_conv12)

    VGG16_pool21 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool21')(VGG16_dropout11)
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
    VGG16_dropout21 = K.layers.Dropout(0.5,
                                       name='VGG16_dropout21')(VGG16_conv24)

    VGG16_pool31 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool31')(VGG16_dropout21)
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
    VGG16_dropout31 = K.layers.Dropout(0.5,
                                       name='VGG16_dropout31')(VGG16_conv34)

    VGG16_pool41 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool41')(VGG16_dropout31)
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
    VGG16_dropout41 = K.layers.Dropout(0.5,
                                       name='VGG16_dropout41')(VGG16_conv44)

    VGG16_pool51 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool51')(VGG16_dropout41)
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
    VGG16_dropout51 = K.layers.Dropout(0.5,
                                       name='VGG16_dropout51')(VGG16_conv52)

    decoder_trconv01 = K.layers.Conv2DTranspose(filters=512,
                                                kernel_size=4,
                                                strides=2,
                                                padding='same',
                                                activation='relu',
                                                name='decoder_trconv01'
                                                )(VGG16_dropout51)
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

    fcn_model.summary()

    return fcn_model


def conv_crf_rnn(input_shape):
    conv_crf_rnn_image_input = K.layers.Input(batch_shape=input_shape[0],
                                              name='conv_crf_rnn_image_input')
    conv_crf_rnn_unaries_input = K.layers.Input(batch_shape=input_shape[1],
                                                name='conv_crf_rnn_unaries_input')
    conv_crf_rnn_edges_input = K.layers.Input(batch_shape=input_shape[2],
                                              name='conv_crf_rnn_edges_input')

    conv_crf_rnn_layer = layers.ConvCrfRnnLayer(name='conv_crf_rnn_layer') \
        ([conv_crf_rnn_image_input,
          conv_crf_rnn_unaries_input,
          conv_crf_rnn_edges_input])

    conv_crf_rnn_model = K.Model([conv_crf_rnn_image_input,
                                  conv_crf_rnn_unaries_input,
                                  conv_crf_rnn_edges_input],
                                 conv_crf_rnn_layer)

    conv_crf_rnn_model.name = 'Conv_CRF_RNN_model'

    conv_crf_rnn_model.summary()

    return conv_crf_rnn_model


def full_network(label_model, edge_model, crf_model):
    full_model_input = K.layers.Input(batch_shape=label_model.input_shape,
                                      name='full_model_input_layer')
    label_model_output = label_model(full_model_input)
    edge_model_output = edge_model(full_model_input)
    crf_model_output = crf_model([full_model_input, label_model_output, edge_model_output])

    full_model = K.models.Model(full_model_input, crf_model_output)
    full_model.name = 'full_model'
    full_model.summary()

    return full_model


def fcn_resnet():
    ...
    # TODO


def unet():
    ...
    # TODO


def rcf(input_shape):
    RCF_input = K.layers.Input(input_shape,
                               name='RCF_input')

    VGG16_conv01 = K.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv01')(RCF_input)
    VGG16_conv02 = K.layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   name='VGG16_conv02')(VGG16_conv01)
    RCF_conv01 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv01')(VGG16_conv01)
    RCF_conv02 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv02')(VGG16_conv02)
    RCF_conv0_sum = K.layers.Add(name='RCF_conv0_sum')([RCF_conv01, RCF_conv02])
    RCF_conv0 = K.layers.Conv2D(filters=16,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                name='RCF_conv0')(RCF_conv0_sum)

    VGG16_dropout01 = K.layers.Dropout(0.5,
                                       name='VGG16_dropout01')(VGG16_conv02)

    VGG16_pool11 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool11')(VGG16_dropout01)
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
    RCF_conv11 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv11')(VGG16_conv11)
    RCF_conv12 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv12')(VGG16_conv12)
    RCF_conv1_sum = K.layers.Add(name='RCF_conv1_sum')([RCF_conv11, RCF_conv12])
    RCF_conv1 = K.layers.Conv2D(filters=16,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                name='RCF_conv1')(RCF_conv1_sum)
    RCF_trconv11 = K.layers.Conv2DTranspose(filters=16,
                                            kernel_size=4,
                                            strides=2,
                                            padding='same',
                                            activation='relu',
                                            name='RCF_trconv11'
                                            )(RCF_conv1)

    VGG16_dropout11 = K.layers.Dropout(0.5,
                                       name='VGG16_dropout11')(VGG16_conv12)

    VGG16_pool21 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool21')(VGG16_dropout11)
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
    RCF_conv21 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv21')(VGG16_conv21)
    RCF_conv22 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv22')(VGG16_conv22)
    RCF_conv23 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv23')(VGG16_conv23)
    RCF_conv2_sum = K.layers.Add(name='RCF_conv2_sum')([RCF_conv21, RCF_conv22, RCF_conv23])
    RCF_conv2 = K.layers.Conv2D(filters=16,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                name='RCF_conv2')(RCF_conv2_sum)
    RCF_trconv21 = K.layers.Conv2DTranspose(filters=16,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu',
                                            name='RCF_trconv21'
                                            )(RCF_conv2)
    RCF_trconv22 = K.layers.Conv2DTranspose(filters=16,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu',
                                            name='RCF_trconv22'
                                            )(RCF_trconv21)

    VGG16_dropout21 = K.layers.Dropout(0.5,
                                       name='VGG16_dropout21')(VGG16_conv23)

    VGG16_pool31 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool31')(VGG16_dropout21)
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
    RCF_conv31 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv31')(VGG16_conv31)
    RCF_conv32 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv32')(VGG16_conv32)
    RCF_conv33 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv33')(VGG16_conv33)
    RCF_conv3_sum = K.layers.Add(name='RCF_conv3_sum')([RCF_conv31, RCF_conv32, RCF_conv33])
    RCF_conv3 = K.layers.Conv2D(filters=16,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                name='RCF_conv3')(RCF_conv3_sum)
    RCF_trconv31 = K.layers.Conv2DTranspose(filters=16,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu',
                                            name='RCF_trconv31'
                                            )(RCF_conv3)
    RCF_trconv32 = K.layers.Conv2DTranspose(filters=16,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu',
                                            name='RCF_trconv32'
                                            )(RCF_trconv31)
    RCF_trconv33 = K.layers.Conv2DTranspose(filters=16,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu',
                                            name='RCF_trconv33'
                                            )(RCF_trconv32)

    VGG16_dropout31 = K.layers.Dropout(0.5,
                                       name='VGG16_dropout31')(VGG16_conv33)

    VGG16_pool41 = K.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      name='VGG16_pool41')(VGG16_dropout31)
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
    RCF_conv41 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv41')(VGG16_conv41)
    RCF_conv42 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv42')(VGG16_conv42)
    RCF_conv43 = K.layers.Conv2D(filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 name='RCF_conv43')(VGG16_conv43)
    RCF_conv4_sum = K.layers.Add(name='RCF_conv4_sum')([RCF_conv41, RCF_conv42, RCF_conv43])
    RCF_conv4 = K.layers.Conv2D(filters=16,
                                kernel_size=1,
                                padding='same',
                                activation='relu',
                                name='RCF_conv4')(RCF_conv4_sum)
    RCF_trconv41 = K.layers.Conv2DTranspose(filters=16,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu',
                                            name='RCF_trconv41'
                                            )(RCF_conv4)
    RCF_trconv42 = K.layers.Conv2DTranspose(filters=16,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu',
                                            name='RCF_trconv42'
                                            )(RCF_trconv41)
    RCF_trconv43 = K.layers.Conv2DTranspose(filters=16,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu',
                                            name='RCF_trconv43'
                                            )(RCF_trconv42)
    RCF_trconv44 = K.layers.Conv2DTranspose(filters=16,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            activation='relu',
                                            name='RCF_trconv44'
                                            )(RCF_trconv43)

    RCF_fusion = K.layers.Concatenate(name='RCF_fusion')([RCF_conv0,
                                                          RCF_trconv11,
                                                          RCF_trconv22,
                                                          RCF_trconv33,
                                                          RCF_trconv44])
    RCF_fusion_conv = K.layers.Conv2D(filters=1,
                                      kernel_size=3,
                                      padding='same',
                                      activation='sigmoid',
                                      name='RCF_fusion_conv')(RCF_fusion)

    rcf_model = K.models.Model(RCF_input, RCF_fusion_conv)
    rcf_model.name = ('RCF_net')
    rcf_model.summary()

    return rcf_model


if __name__ == "__main__":
    rcf((512, 640, 3))
