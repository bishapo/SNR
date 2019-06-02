from keras import applications, optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input
from keras.applications.vgg16 import vgg16, preprocess_input
import keras
import numpy as np
#import pickle
import dill
#import ImageGenerator
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *
from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input
from keras.applications.vgg16 import vgg16, preprocess_input


def show_dir_images(breed, n_to_show):
    plt.figure(figsize=(16,16))
    img_dir = "Images/{}/".format(breed)
    images = os.listdir(img_dir)[:n_to_show]
    for i in range(n_to_show):
        img = mpimg.imread(img_dir + images[i])
        plt.subplot(n_to_show/4+1, 4, i+1)
        plt.imshow(img)
        plt.axis('off')
def paths_and_labels():
    paths = list()
    labels = list()
    targets = list()
    for breed in breed_list:
        base_name = "./data/{}/".format(breed)
        for img_name in os.listdir(base_name):
            paths.append(base_name + img_name)
            labels.append(breed)
            targets.append(label_maps[breed])
    return paths, labels, targets
class ImageGenerator(Sequence):
    def __init__(self, paths, targets, batch_size, shape, augment=False):
        self.paths = paths
        self.targets = targets
        self.batch_size = batch_size
        self.shape = shape
        self.augment = augment
        
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = np.zeros((len(batch_paths), self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
        y = np.zeros((self.batch_size, num_classes, 1))
        for i, path in enumerate(batch_paths):
            x[i] = self.__load_image(path)
        y = self.targets[idx * self.batch_size : (idx + 1) * self.batch_size]
        return x, y
    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
    def __load_image(self, path):
        image = imread(path)
        image = preprocess_input(image)
        if self.augment:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.CropAndPad(percent=(-0.25, 0.25)),
                    iaa.Crop(percent=(0, 0.1)),
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])
            ], random_order=True)
            image = seq.augment_image(image)
        return image

breed_list = os.listdir("Images/")
num_classes = len(breed_list)
print("{} breeds".format(num_classes))
n_total_images = 0
for breed in breed_list:
    n_total_images += len(os.listdir("Images/{}".format(breed)))
print("{} images".format(n_total_images))
label_maps = {}
label_maps_rev = {}
for i, v in enumerate(breed_list):
    label_maps.update({v: i})
    label_maps_rev.update({i : v})
show_dir_images(breed_list[0], 16)


#os.mkdir('data')
#for breed in breed_list:
#    os.mkdir('data/' + breed)
#print('Created {} folders to store cropped images of the different breeds.'.format(len(os.listdir('data'))))
#k = 0
#for breed in os.listdir('data'):
#    print((100*k)/(len(os.listdir('data'))))
#    for file in os.listdir('Annotation/{}'.format(breed)):
#        img = Image.open('Images/{}/{}.jpg'.format(breed, file))
#        tree = ET.parse('Annotation/{}/{}'.format(breed, file))
#        xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
#        xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
#        ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
#        ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
#        img = img.crop((xmin, ymin, xmax, ymax))
#        img = img.convert('RGB')
#        img = img.resize((224, 224))
#        img.save('data/' + breed + '/' + file + '.jpg')
#    k = k + 1
    
paths, labels, targets = paths_and_labels()
assert len(paths) == len(labels)
assert len(paths) == len(targets)
targets = np_utils.to_categorical(targets, num_classes=num_classes)
train_paths, val_paths, train_targets, val_targets = train_test_split(paths, 
                                                  targets,
                                                  test_size=0.15, 
                                                  random_state=1029)
train_gen = ImageGenerator(train_paths, train_targets, batch_size=32, shape=(224,224,3), augment=True)
val_gen = ImageGenerator(val_paths, val_targets, batch_size=32, shape=(224,224,3), augment=False)

from keras import backend as K
import tensorflow as tf

top_values, top_indices = K.get_session().run(tf.nn.top_k(_pred_test, k=5))

keras.backend.set_image_dim_ordering('tf')
weights_path = 'vgg16.h5'
img_width, img_height = 224, 224
input_tensor = Input(shape=(224, 224,3))
model = applications.VGG16(weights='imagenet', 
                           include_top=False,
                           input_tensor=input_tensor)
new_model = Sequential()
for l in model.layers:
    new_model.add(l)
for layer in new_model.layers:
    layer.trainable = False
new_model.add(Flatten())
new_model.add(Dense(units=4096, activation='relu'))
new_model.add(Dense(units=4096, activation='relu'))
new_model.add(Dense(units=120, activation='softmax'))
#COMPILE THE MODEL
new_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics= ['accuracy', 'top_k_categorical_accuracy'])
new_model.summary()
epochs = 10
#  FIT THE MODEL
learning = new_model.fit_generator(
    generator=train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=epochs,
    verbose=1,
    )
new_model.save_weights('case1.h5')
