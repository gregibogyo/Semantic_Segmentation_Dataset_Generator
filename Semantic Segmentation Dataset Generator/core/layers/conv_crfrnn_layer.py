import keras as K


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
        super(ConvCrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        unaries, image = inputs

        for i in range(self.num_iterations):
            a = 0

        return unaries

    def compute_output_shape(self, input_shape):
        return input_shape[0]
