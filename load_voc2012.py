import os
from PIL import Image
import numpy as np
import xml.etree.cElementTree as ET


def load_voc2012():
    dataset_path = os.path.join('data', 'voc2012')
    classification_path = os.path.join(dataset_path, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'trainval.txt')
    image_path = os.path.join(dataset_path, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    label_path = os.path.join(dataset_path, 'VOCdevkit', 'VOC2012', 'Annotations')
    image_names = []
    label_names = []
    with open(classification_path, 'r') as f:
        for line in f.readlines():
            image_names.append(line.strip() + '.jpg')
            label_names.append(line.strip() + '.xml')

    images = []
    for image_name in image_names:
        image = Image.open(os.path.join(image_path, image_name))
        image = np.array(image)
        images.append(image)
    images = np.array(images)

    labels = {}

    for label_name in label_names:
        tree = ET.parse(os.path.join(label_path, label_name))
        root = tree.getroot()
        label = []
        for child in root:
            for name in child:
                if name.tag == 'name':
                    label.append(name.text)

        labels[str(label_name).rstrip('.xml')] = list(set(label))

    return images, labels


images, labels = load_voc2012()
print(labels)

print(len(images), len(labels))
