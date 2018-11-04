from keras.backend import tf as tf
import keras as K
import numpy as np


def compability_matrix_initalizer(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)


class ConvCrfRnnLayer(K.layers.Layer):
    def __init__(self, num_iterations=10, kernel_size=3, **kwargs):
        self.num_iterations = num_iterations
        self.kernel_size = kernel_size
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None

        self.input_masks = [None, None]

        super(ConvCrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.image_channels = input_shape[0][-1]
        self.num_classes = input_shape[1][-1]

        self.unary_weight = self.add_weight(name='unary_weight',
                                            shape=(1, 1),
                                            initializer=K.initializers.uniform(0, 1),
                                            trainable=True)
        self.color_weight = self.add_weight(name='color_weight',
                                            shape=(1, 1),
                                            initializer=K.initializers.uniform(0, 1),
                                            trainable=True)
        self.edge_weight = self.add_weight(name='color_weight',
                                           shape=(1, 1),
                                           initializer=K.initializers.uniform(0, 1),
                                           trainable=True)
        self.color_theta = self.add_weight(name='color_theta',
                                           shape=(1, 1),
                                           initializer=K.initializers.uniform(0, 1),
                                           trainable=True)
        self.edge_theta = self.add_weight(name='edge_theta',
                                          shape=(1, 1),
                                          initializer=K.initializers.uniform(0, 1),
                                          trainable=True)

        self.color_difference_kernel = self.color_difference_kernel_initalizer(kernel_size=self.kernel_size,
                                                                               depth=self.image_channels,
                                                                               middle_item=-1)

        self.edge_probability_kernel = self.color_difference_kernel_initalizer(kernel_size=3,
                                                                               depth=1,
                                                                               middle_item=0)

        self.color_indice_diff_kernel = self.color_difference_kernel_initalizer(kernel_size=self.kernel_size,
                                                                                depth=1,
                                                                                middle_item=-1)

        self.edge_indice_diff_kernel = self.edge_difference_kernel_initalizer(kernel_size=5,
                                                                              depth=1,
                                                                              middle_item=-1)

        super(ConvCrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        image, unaries, edges = inputs
        Q_all = unaries

        color_difference = K.backend.depthwise_conv2d(image,
                                                      depthwise_kernel=self.color_difference_kernel,
                                                      padding='same')

        color_difference = K.backend.reshape(color_difference,
                                             shape=([K.backend.shape(color_difference)[0],
                                                     color_difference.get_shape().as_list()[1],
                                                     color_difference.get_shape().as_list()[2],
                                                     int(color_difference.get_shape().as_list()[
                                                             3] / self.image_channels),
                                                     self.image_channels]))

        color_difference = K.backend.sum((color_difference ** 2), axis=-1)
        color_probability = K.backend.exp(-color_difference / self.color_theta)

        edge_difference = K.backend.depthwise_conv2d(edges,
                                                     depthwise_kernel=self.edge_probability_kernel,
                                                     padding='same')

        edge_probability = K.backend.exp(-edge_difference ** 2 / self.edge_theta)

        num_batch = K.backend.shape(image)[0]
        num_height = K.backend.shape(image)[1]
        num_width = K.backend.shape(image)[2]

        ones = tf.ones(shape=[num_batch, num_height, num_width, self.kernel_size ** 2 - 1])
        minus_ones = -ones

        batch_range = tf.range(num_batch)
        height_range = tf.range(num_height)
        width_range = tf.range(num_width)
        batch_tensor = tf.tile(batch_range[:, None, None, None], (1, num_height, num_width, 1))
        height_tensor = tf.tile(height_range[None, :, None, None], (num_batch, 1, num_width, 1))
        width_tensor = tf.tile(width_range[None, None, :, None], (num_batch, num_height, 1, 1))

        for i in range(self.num_iterations):
            Q, indices = tf.nn.top_k(Q_all, k=1)

            # shape the indices into scatter_nd form
            batch_height_width_stack = tf.stack([batch_tensor, height_tensor, width_tensor, indices], axis=4)

            float_indices = tf.cast(indices, dtype=tf.float32)

            color_indice_diff = K.backend.depthwise_conv2d(float_indices,
                                                           depthwise_kernel=self.color_indice_diff_kernel,
                                                           padding='same')

            color_upgrape = tf.where(tf.greater(color_indice_diff, 0.1),
                                     x=minus_ones,
                                     y=ones)

            color_upgrape = K.backend.sum(color_probability * color_upgrape, axis=-1, keepdims=True)

            edge_indice_diff = K.backend.depthwise_conv2d(float_indices,
                                                          depthwise_kernel=self.edge_indice_diff_kernel,
                                                          padding='same')

            edge_upgrape = tf.where(tf.greater(edge_indice_diff, 0.1),
                                    x=minus_ones,
                                    y=ones)

            edge_upgrape = K.backend.sum(edge_probability * edge_upgrape, axis=-1, keepdims=True)

            pairwise = (self.color_weight * color_upgrape) + (self.edge_weight * edge_upgrape)

            # build the mask where there are the pairwise values are in the indices places
            pairwise_augmented = tf.scatter_nd(indices=batch_height_width_stack,
                                               updates=pairwise,
                                               shape=[num_batch, num_height, num_width, self.num_classes])

            potencial = (self.unary_weight * unaries) + pairwise_augmented

            Q_all = K.backend.softmax(potencial, axis=-1)

        return Q_all

    def color_difference_kernel_initalizer(self, kernel_size=3, depth=1, middle_item=-1):
        color_difference_kernel = np.zeros([kernel_size, kernel_size, depth, kernel_size * kernel_size - 1])
        middle_index = np.floor(kernel_size / 2).astype(np.uint8)
        k = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i != middle_index or j != middle_index:
                    color_difference_kernel[i, j, :, k] = 1.0
                    k = k + 1
        color_difference_kernel[middle_index, middle_index, :] = middle_item
        color_difference_kernel = color_difference_kernel.astype(np.float32)

        return color_difference_kernel

    def edge_difference_kernel_initalizer(self, kernel_size=5, depth=1, middle_item=-1):
        color_difference_kernel = np.zeros([kernel_size, kernel_size, depth, 3 * 3 - 1])
        middle_index = np.floor(kernel_size / 2).astype(np.uint8)
        k = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i != middle_index or j != middle_index) and j % 2 == 0 and i % 2 == 0:
                    color_difference_kernel[i, j, :, k] = 1.0
                    k = k + 1

        color_difference_kernel[middle_index, middle_index, :] = middle_item
        color_difference_kernel = color_difference_kernel.astype(np.float32)

        return color_difference_kernel

    def compute_output_shape(self, input_shape):
        return input_shape[1]
