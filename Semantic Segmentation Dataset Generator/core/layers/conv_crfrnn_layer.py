import keras as K
import numpy as np


def compability_matrix_initalizer(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)


class ConvCrfRnnLayer(K.layers.Layer):
    def __init__(self, num_iterations=5, **kwargs):
        self.theta_alpha = 1
        self.theta_beta = 2
        self.theta_gamma = 3
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None

        self.input_masks = [None, None]

        super(ConvCrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.image_dims = input_shape[1][-1]
        self.num_classes = input_shape[0][-1]

        self.contrast_weight_matrix = self.add_weight(name='contrast_weight_matrix',
                                                      shape=(1, 1, 1, self.num_classes, self.num_classes),
                                                      initializer=K.initializers.RandomUniform(),
                                                      trainable=True)

        self.compability_matrix = self.add_weight(name='compability_matrix',
                                                  shape=(self.num_classes, self.num_classes),
                                                  initializer=K.initializers.Identity(),
                                                  trainable=True)
        self.compability_matrix = K.backend.expand_dims(
            K.backend.expand_dims(
                K.backend.expand_dims(self.compability_matrix, 0), 0), 0)

        self.contrast_difference_kernel = self.contrast_difference_kernel_initalizer(kernel_size=3,
                                                                                     depth=self.image_dims,
                                                                                     middle_item=-1)
        self.contrast_theta = self.add_weight(name='contrast_theta',
                                              shape=[1],
                                              initializer=K.initializers.RandomUniform(),
                                              trainable=True)

        self.unary_kernel = self.contrast_difference_kernel_initalizer(kernel_size=3,
                                                                       depth=self.num_classes,
                                                                       middle_item=0)

        super(ConvCrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        unaries, image = inputs
        Q = unaries

        contrast = K.backend.depthwise_conv2d(image,
                                              depthwise_kernel=self.contrast_difference_kernel,
                                              padding='same')

        contrast = K.backend.reshape(contrast,
                                     shape=([K.backend.shape(contrast)[0],
                                             contrast.get_shape().as_list()[1],
                                             contrast.get_shape().as_list()[2],
                                             int(contrast.get_shape().as_list()[3] / self.image_dims),
                                             self.image_dims]))

        contrast = contrast ** 2
        contrast = K.backend.sum(contrast, axis=-1)
        k = K.backend.exp(- contrast / (2 * (self.contrast_theta ** 2)))

        for i in range(self.num_iterations):
            Q = K.backend.depthwise_conv2d(Q,
                                           depthwise_kernel=self.unary_kernel,
                                           padding='same')
            Q = K.backend.reshape(Q,
                                  shape=([K.backend.shape(Q)[0],
                                          K.backend.shape(Q)[1],
                                          K.backend.shape(Q)[2],
                                          int(Q.get_shape().as_list()[3] / self.num_classes),
                                          self.num_classes]))

            Q = K.backend.sum(K.backend.expand_dims(k, axis=-1) * Q, axis=-2)
            Q = K.backend.sum(self.contrast_weight_matrix * K.backend.expand_dims(Q, -1), axis=-2)
            Q = K.backend.sum(self.compability_matrix * K.backend.expand_dims(Q, -1), axis=-2)
            Q = K.backend.exp(-unaries - Q)
            Q = K.backend.softmax(Q, axis=-1)

        return Q

    def contrast_difference_kernel_initalizer(self, kernel_size=3, depth=1, middle_item=-1):
        contrast_difference_kernel = np.zeros([kernel_size, kernel_size, depth, kernel_size * kernel_size])
        middle = np.floor(kernel_size / 2).astype(np.uint8)
        for i in range(kernel_size):
            for j in range(kernel_size):
                contrast_difference_kernel[i, j, :, i * kernel_size + j] = 1.0
        contrast_difference_kernel[middle, middle, :] = middle_item
        contrast_difference_kernel = contrast_difference_kernel.astype(np.float32)

        return contrast_difference_kernel

    def compute_output_shape(self, input_shape):
        return input_shape[0]
