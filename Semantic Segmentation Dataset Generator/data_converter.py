import numpy as np
import cv2
import os
import pickle
from PIL import Image

# This data converter's aim is to:
# resize the images to 512 x width size
# cut the images to 512 x 683 size
# delete the pixels in the labels out from 0 to 66 (the object indexes)
# make the edge ground truth from the labels

deleteall = True

mapillary_dict = "D:\\Mapillary"
imageset = 'training'

mapillary_raw_dict = os.path.join(mapillary_dict, 'raw')
mapillary_512_dict = os.path.join(mapillary_dict, '512')

raw_mapillary_images_dict = os.path.join(mapillary_raw_dict, imageset + "\\images")
raw_mapillary_labels_dict = os.path.join(mapillary_raw_dict, imageset + '\\labels')

_512_mapillary_images_dict = os.path.join(mapillary_512_dict, imageset + "\\images")
if not os.path.exists(_512_mapillary_images_dict):
    os.makedirs(_512_mapillary_images_dict)

_512_mapillary_labels_dict = os.path.join(mapillary_512_dict, imageset + "\\labels")
if not os.path.exists(_512_mapillary_labels_dict):
    os.makedirs(_512_mapillary_labels_dict)

_512_mapillary_edges_dict = os.path.join(mapillary_512_dict, imageset + "\\edges")
if not os.path.exists(_512_mapillary_edges_dict):
    os.makedirs(_512_mapillary_edges_dict)

# load the names if exist, if not read the names and save them
mapillary_names = os.path.join(mapillary_512_dict, "images.dll")
if os._exists(mapillary_names) and not deleteall:
    with open(mapillary_names, 'rb') as f:
        imagenames = pickle.load(f)
else:
    imagenames = os.listdir(raw_mapillary_images_dict)
    imagenames = [imagename.split('.')[0] for imagename in imagenames]
    with open(mapillary_names, 'wb') as f:
        pickle.dump(imagenames, f)

print(imagenames)
_512_imagenames = []

for i in range(len(imagenames)):
    imagename = imagenames[i] + ".jpg"
    raw_mapillary_image_path = os.path.join(raw_mapillary_images_dict, imagename)
    raw_image = Image.open(raw_mapillary_image_path)
    raw_image = np.array(raw_image)

    labelname = imagenames[i] + ".png"
    raw_mapillary_label_path = os.path.join(raw_mapillary_labels_dict, labelname)
    raw_label = Image.open(raw_mapillary_label_path)
    raw_label = np.array(raw_label)

    _512_label_path = os.path.join(_512_mapillary_labels_dict, labelname)
    _512_edge_path = os.path.join(_512_mapillary_edges_dict, labelname)
    _512_image_path = os.path.join(_512_mapillary_images_dict, imagename)

    if not os.path.exists(_512_label_path) or deleteall:
        raw_labelheight = raw_label.shape[0]
        label_rate = 512 / raw_labelheight
        _512_label = cv2.resize(raw_label, dsize=(0, 0), fx=label_rate, fy=label_rate, interpolation=cv2.INTER_CUBIC)
        _512_label = _512_label[:, 0:640]
        _512_label = np.clip(_512_label, a_min=0, a_max=66)
        cv2.imwrite(filename=_512_label_path, img=_512_label)

    if not os.path.exists(_512_edge_path) or deleteall:
        _512_edge = cv2.Canny(_512_label, threshold1=0, threshold2=1)
        _512_edge = _512_edge[:, 0:640]
        cv2.imwrite(filename=_512_edge_path, img=_512_edge)

    if not os.path.exists(_512_image_path) or deleteall:
        raw_imageheight = raw_image.shape[0]
        image_rate = 512 / raw_imageheight
        _512_image = cv2.resize(raw_image, dsize=(0, 0), fx=image_rate, fy=image_rate, interpolation=cv2.INTER_CUBIC)
        _512_image = _512_image[:, 0:640]
        _512_image = cv2.cvtColor(_512_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename=_512_image_path, img=_512_image)

    if _512_label.shape != (512, 640) or _512_edge.shape != (512, 640) or _512_image.shape != (512, 640, 3):
        if os.path.exists(_512_label_path):
            os.remove(_512_label_path)
        if os.path.exists(_512_edge_path):
            os.remove(_512_edge_path)
        if os.path.exists(_512_image_path):
            os.remove(_512_image_path)
    else:
        _512_imagenames.append(imagenames[i])

    print("%i.\n "
          "\t Label: Size: %s \t Min: %s \t Max: %s \n"
          "\t Edge: Size: %s \n"
          "\t Image: Size: %s \n"
          % (i, _512_label.shape, np.min(_512_label), np.max(_512_label), _512_edge.shape, _512_image.shape))

    with open(mapillary_names, 'wb') as f:
        pickle.dump(_512_imagenames, f)
