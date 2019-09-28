from load_voc2012 import load_voc2012
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense
from keras.layers import LeakyReLU, Activation
import keras
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split

images, labels = load_voc2012()
seed = 42
np.random.seed(seed)
print(images.shape, labels.shape)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=seed)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
