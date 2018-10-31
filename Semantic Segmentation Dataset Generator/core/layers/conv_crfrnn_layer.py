from keras.backend import tf as tf
import keras as K
import numpy as np


def compability_matrix_initalizer(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)


class ConvCrfRnnLayer(K.layers.Layer):
    def __init__(self, num_iterations=5, **kwargs):
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None

        self.input_masks = [None, None]

        super(ConvCrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.image_channels = input_shape[1][-1]
        self.num_classes = input_shape[0][-1]

        self.unary_weight = self.add_weight(name='unary_weight',
                                            shape=(1, 1),
                                            initializer=K.initializers.uniform(),
                                            trainable=True)
        self.contrast_weight = self.add_weight(name='contrast_weight',
                                               shape=(1, 1),
                                               initializer=K.initializers.random_normal(),
                                               trainable=True)

        self.contrast_difference_kernel = self.contrast_difference_kernel_initalizer(kernel_size=3,
                                                                                     depth=self.image_channels,
                                                                                     middle_item=-1)

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
                                             int(contrast.get_shape().as_list()[3] / self.image_channels),
                                             self.image_channels]))

        contrast = K.backend.sum(contrast ** 2, axis=-1)
        k = K.backend.exp(- contrast)

        num_batch = 1
        num_height = 512
        num_width = 640
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

            # build the mask where there are ones in the indices places
            Q_augmented = tf.scatter_nd(indices=batch_height_width_stack,
                                        updates=Q,
                                        shape=[num_batch, num_height, num_width, self.num_classes])

            Q_all = Q_all + Q_augmented
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
