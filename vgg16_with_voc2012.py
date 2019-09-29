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
prediction  = Dense(20, activation='sigmoid')(x)
# prediction = ThresholdedReLU(theta=0.8)(x)

for layer in base_model.layers:
    layer.trainable = False
model = Model(inputs=base_model.input, outputs=prediction)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=4, batch_size=32)

for layer in model.layers:
    layer.trainable = True
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

# model = load_model('vgg16_original.h5')
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])

tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         batch_size=32,  # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=False,  # 是否可视化梯度直方图
                         write_images=False,  # 是否可视化参数
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None,
                         embeddings_data=None,
                         update_freq='epoch')
checkpointer = ModelCheckpoint(filepath='vgg16_with_voc2012.h5', verbose=1, save_best_only=False,
                               save_weights_only=False)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=128, batch_size=32,
          callbacks=[tbCallBack, checkpointer])
