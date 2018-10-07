import keras as K
import tensorflow as tf


def edge_detection_loss(y_true, y_pred):
    y_pred_ones = y_true * y_pred
    # edge_pixel_number = K.backend.mean(y_true, axis=[1, 2, 3])
    # all_pixel_number = y_pred.get_shape().as_list()[1] * \
    #                    y_pred.get_shape().as_list()[2]

    ones_loss = K.backend.binary_crossentropy(y_true, y_pred_ones)
    zeros_loss = K.backend.binary_crossentropy(y_true, y_pred)

    mean_ones_loss = K.backend.mean(ones_loss, axis=[1, 2, 3])
    mean_zeros_loss = K.backend.mean(zeros_loss, axis=[1, 2, 3])

    # mean_ones_loss = tf.Print(mean_ones_loss, [mean_ones_loss], message="\nmean_ones_loss ")
    # mean_zeros_loss = tf.Print(mean_zeros_loss, [mean_zeros_loss], message="\nsum_zeros_loss ")

    normed_mean_ones_loss = mean_ones_loss / (mean_ones_loss + mean_zeros_loss)
    normed_mean_zeros_loss = mean_zeros_loss / (mean_ones_loss + mean_zeros_loss)

    ones_loss = ones_loss / normed_mean_ones_loss
    zeros_loss = zeros_loss / normed_mean_zeros_loss

    return K.backend.mean(zeros_loss + ones_loss, axis=-1)
