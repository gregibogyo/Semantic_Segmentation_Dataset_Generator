import numpy as np
import cv2
import os
import pickle
from PIL import Image
from shutil import copyfile

mapillary_dict = "D:\\Mapillary"

raw_mapillary_images_dict = os.path.join(mapillary_dict, "raw\\images")
raw_mapillary_labels_dict = os.path.join(mapillary_dict, "\\labels")

_512_mapillary_images_dict = os.path.join(mapillary_dict, "512\\512_images")
if not os.path.exists(_512_mapillary_images_dict):
    os.makedirs(_512_mapillary_images_dict)

_512_mapillary_66labels_dict = os.path.join(mapillary_dict, "512\\labels")
if not os.path.exists(_512_mapillary_66labels_dict):
    os.makedirs(_512_mapillary_66labels_dict)

mapillary_names = os.path.join(mapillary_dict, "trainingimages.dll")
with open(mapillary_names, 'rb') as f:
    imagenames = pickle.load(f)

print(imagenames)

new_imagenames = []
'''
for i in range(len(imagenames)):
    imagename = imagenames[i] + ".jpg"
    mapillary_image_path = os.path.join(mapillary_images_dict, imagename)

    labelname = imagenames[i] + ".png"
    mapillary_label_path = os.path.join(mapillary_labels_dict, labelname)

    base_image = Image.open(mapillary_image_path)
    image = np.array(base_image)


    label_path = os.path.join(mapillary_labels_dict, labelname)
    base_label = Image.open(label_path)
    label = np.array(base_label)
    #label = cv2.resize(label, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)


    labelheight = label.shape[0]
    if labelheight == 768:
        new_imagenames.append(imagenames[i])
        new_label = np.clip(label, a_min=0, a_max=66)
        print(np.max(new_label))
        new_image_label = os.path.join(new_mapillary_66labels_dict, labelname)
        cv2.imwrite(filename=new_image_label, img=new_label)

    imageheight = image.shape[0]
    if imageheight == 768:
        new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        new_image_name = os.path.join(new_mapillary_images_dict, imagename)
        cv2.imwrite(filename=new_image_name, img=new_image)


    with open(new_mapillary_names, 'wb') as f:
        pickle.dump(new_imagenames, f)
'''