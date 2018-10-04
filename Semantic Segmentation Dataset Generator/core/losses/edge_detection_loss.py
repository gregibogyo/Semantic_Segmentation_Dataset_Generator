import keras as K


def edge_detection_loss(y_true, y_pred):
    y_pred_ones = y_true * y_pred
    ones_loss = K.backend.binary_crossentropy(y_true, y_pred_ones)*1000

    y_pred_zeros = (1. - y_true) * y_pred
    zeros_loss = K.backend.binary_crossentropy(y_true, y_pred_zeros)

    return K.backend.mean(ones_loss+zeros_loss, axis=-1)
