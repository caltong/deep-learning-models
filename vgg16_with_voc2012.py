from load_voc2012 import load_voc2012
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization
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

model = Sequential()
# block 1
model.add(
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(224, 224, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (112,112,64)
# block 2
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (56,56,128)
# block 3
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (28,28,256)
# block 4
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (14,14,512)
# block 5
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (7,7,512)
# Flatten
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Activation('sigmoid'))

Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam, metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epoch=4, batch_size=8)
model.save('vgg16_with_voc2012.h5')
