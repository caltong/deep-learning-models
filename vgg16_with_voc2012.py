from load_voc2012 import load_voc2012
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.layers import LeakyReLU, Activation, ThresholdedReLU
import keras
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications import VGG16
from keras.models import Model, load_model
from keras.optimizers import SGD

images, labels = load_voc2012()
# seed = 42
# np.random.seed(seed)
print(images.shape, labels.shape)

# x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=seed)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# load pretrained vgg16
base_model = VGG16(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
prediction = Dense(20, activation='sigmoid')(x)
# prediction = ThresholdedReLU(theta=0.8)(x)

for layer in base_model.layers:
    layer.trainable = False
model = Model(inputs=base_model.input, outputs=prediction)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=32)

model.save('pretrained_after_fine_tuning_2_epochs.h5')
# for layer in model.layers:
#     layer.trainable = True
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

# model = load_model('vgg16_original.h5')
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])


# checkpointer = ModelCheckpoint(filepath='vgg16_with_voc2012.h5', verbose=1, save_best_only=False,
#                                save_weights_only=False)
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=128, batch_size=32,
#           callbacks=[checkpointer])
