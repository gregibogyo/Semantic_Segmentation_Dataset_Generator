import keras as K
import numpy as np


def compability_matrix_initalizer(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)


class ConvCrfRnnLayer(K.layers.Layer):
    def __init__(self, num_iterations=1, **kwargs):
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None

        self.input_masks = [None, None]

        super(ConvCrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.image_dims = input_shape[1][-1]
        self.num_classes = input_shape[0][-1]

        # self.contrast_weight_matrix = self.add_weight(name='contrast_weight_matrix',
        #                                               shape=(self.num_classes, self.num_classes),
        #                                               initializer=K.initializers.Identity(),
        #                                               trainable=True)
        #
        # self.compability_matrix = self.add_weight(name='compability_matrix',
        #                                           shape=(self.num_classes, self.num_classes),
        #                                           initializer=K.initializers.Identity(),
        #                                           trainable=True)

        self.unary_weight = self.add_weight(name='unary_weight',
                                            shape=(1, 1),
                                            initializer=K.initializers.uniform(),
                                            trainable=True)
        self.contrast_weight = self.add_weight(name='contrast_weight',
                                               shape=(1, 1),
                                               initializer=K.initializers.random_normal(),
                                               trainable=True)

        # self.compability_matrix = K.backend.expand_dims(
        #     K.backend.expand_dims(
        #         K.backend.expand_dims(self.compability_matrix, 0), 0), 0)

        self.contrast_difference_kernel = self.contrast_difference_kernel_initalizer(kernel_size=3,
                                                                                     depth=self.image_dims,
                                                                                     middle_item=-1)
        # self.contrast_theta = self.add_weight(name='contrast_theta',
        #                                       shape=[1],
        #                                       initializer=K.initializers.RandomUniform(),
        #                                       trainable=True)

        self.unary_kernel = self.contrast_difference_kernel_initalizer(kernel_size=3,
                                                                       depth=1,
                                                                       middle_item=0)

        super(ConvCrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        unaries, image = inputs
        Q_all = unaries

        contrast = K.backend.depthwise_conv2d(image,
                                              depthwise_kernel=self.contrast_difference_kernel,
                                              padding='same')

        contrast = K.backend.reshape(contrast,
                                     shape=([K.backend.shape(contrast)[0],
                                             contrast.get_shape().as_list()[1],
                                             contrast.get_shape().as_list()[2],
                                             int(contrast.get_shape().as_list()[3] / self.image_dims),
                                             self.image_dims]))

        contrast = K.backend.sum(contrast ** 2, axis=-1)
        k = K.backend.exp(- contrast)

        num_batch = K.backend.tf.shape(image)[0]
        num_height = K.backend.tf.shape(image)[1]
        num_width = K.backend.tf.shape(image)[2]
        batch_range = K.backend.tf.range(num_batch)
        batch_height = K.backend.tf.range(num_height)
        batch_width = K.backend.tf.range(num_width)
        row_tensor = K.backend.tf.tile(batch_range[:, None], (1, 1))

        for i in range(self.num_iterations):
            Q, indices = K.backend.tf.nn.top_k(Q_all, k=1)
            # Q = K.backend.depthwise_conv2d(Q,
            #                                depthwise_kernel=self.unary_kernel,
            #                                padding='same')
            #
            # Q = K.backend.reshape(Q,
            #                       shape=([K.backend.shape(Q)[0],
            #                               K.backend.shape(Q)[1],
            #                               K.backend.shape(Q)[2],
            #                               int(Q.get_shape().as_list()[3] / self.top_class_number),
            #                               self.top_class_number]))
            #
            # # Q = K.backend.sum(Q, axis=-2)
            # # Q = K.backend.sum(K.backend.expand_dims(k, axis=-1) * Q, axis=-2)
            # Q = K.backend.batch_dot(k, Q, axes=[-1, -2])
            # Q = K.backend.sum(self.contrast_weight_matrix * K.backend.expand_dims(Q, -1), axis=-2)
            # Q = K.backend.sum(self.compability_matrix * K.backend.expand_dims(Q, -1), axis=-2)



            # stack along the final dimension, as this is what
            # scatter_nd uses as the indices
            top_k_row_col_indices = K.backend.tf.stack([row_tensor, indices], axis=2)

            # to mask off everything, we will multiply the top_k by
            # 1. so all the updates are just 1
            updates = K.backend.tf.ones([num_batch, k], dtype=K.backend.tf.float32)

            # build the mask
            zero_mask = K.backend.tf.scatter_nd(top_k_row_col_indices, updates, [num_batch, 4])
            shape = K.backend.tf.constant([1, 512, 640, 67])
            Q_all = K.backend.tf.scatter_nd_add(ref=Q_all, indices=indices, updates=Q)
            potencial = (self.unary_weight * unaries + self.contrast_weight * Q_all) / \
                        (self.unary_weight + self.contrast_weight)
            Q_all = K.backend.exp(potencial)
            Q_all = K.backend.softmax(Q_all, axis=-1)

        return Q_all

    def contrast_difference_kernel_initalizer(self, kernel_size=3, depth=1, middle_item=-1):
        contrast_difference_kernel = np.zeros([kernel_size, kernel_size, depth, kernel_size * kernel_size - 1])
        middle_index = np.floor(kernel_size / 2).astype(np.uint8)
        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i + j) < middle_index:
                    contrast_difference_kernel[i, j, :, i * kernel_size + j] = 1.0
                elif (i + j) > middle_index:
                    contrast_difference_kernel[i, j, :, i * kernel_size + j - 1] = 1.0
        contrast_difference_kernel[middle_index, middle_index, :] = middle_item
        contrast_difference_kernel = contrast_difference_kernel.astype(np.float32)

        return contrast_difference_kernel

    def compute_output_shape(self, input_shape):
        return input_shape[0]
