import keras as K

class ImageCheckCallback(K.callbacks.Callback):
    def __init__(self):
        super(ImageCheckCallback, self).__init__()