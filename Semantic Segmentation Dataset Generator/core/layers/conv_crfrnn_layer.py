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
        self.image_dims = input_shape[1][-1:-3]
        self.num_classes = input_shape[0][-1]

        self.contrast_weight_matrix = self.add_weight(name='contrast_weight_matrix',
                                                      shape=(self.num_classes, self.num_classes),
                                                      initializer=K.initializers.RandomUniform(),
                                                      trainable=True)

        self.compability_matrix = self.add_weight(name='compability_matrix',
                                                  shape=(self.num_classes, self.num_classes),
                                                  initializer=K.initializers.Identity(),
                                                  trainable=True)

        self.contrast_difference_kernel = self.contrast_difference_kernel_initalizer(kernel_size=3,
                                                                                     depth=3,
                                                                                     middle_item=-1)

        super(ConvCrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        unaries, image = inputs
        Q = unaries

        for i in range(self.num_iterations):
            contrast = K.backend.depthwise_conv2d(image,
                                                  depthwise_kernel=self.contrast_difference_kernel)

            Q = K.backend.dot(self.contrast_weight_matrix, Q)
            Q = K.backend.dot(self.compability_matrix, Q)
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
