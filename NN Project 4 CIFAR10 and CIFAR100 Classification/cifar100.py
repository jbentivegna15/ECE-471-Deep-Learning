'''
Project: Classifying CIFAR100
Author: Joseph Bentivegna
Instructor: Christopher Curro

Overview:
This project involved creating a CNN to classify the CIFAR100 dataset that consists of pictures
of 100 objects.  The data was imported from keras and split into three different subsets for training,
verification, and testing. The model consists of three blocks, each with a topology similar to
[CONV BN CONV BN POOL]. The final layer consists of two dense layers with a softmax to compute the final classification.
The model trains in approximately 73s per epoch using Google Colab and achieved a top-5 accuracy of 73% on the test dataset after
10 epochs.

All references are cited inline.
'''

# imports
import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# define globals
batch_size = 32
num_classes = 100
epochs = 10

# input image dimensions
img_rows, img_cols = 32, 32

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train, x_ver, y_train, y_ver = train_test_split(x_train, y_train, test_size=5000)

# properly shape input data
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_ver = x_ver.reshape(x_ver.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

# type edits
x_train = x_train.astype('float32')
x_ver = x_ver.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_ver /=255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_ver.shape[0], 'ver samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_ver = keras.utils.to_categorical(y_ver, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# constants - number achieved from tuning on validation set
l2weight = 0.0001

# full model
'''
Curro I know you're going to write D.R.Y. on this and im sorry but I'm already
handing this in late and I don't want to take more time to functionalize it.

The idea for this topology was inspired from: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
However, many changes were made including:
    -Adding BatchNormaliztion between layers to normalize weights and decrease computation time
    -Increasing dropout to prevent overfitting after many epochs
    -Greatly decreasing final dense layer before softmax
    -Using Adam optimizer to train, taking advantage of varying learning rates
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight), input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# build the graph
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy', 'top_k_categorical_accuracy'])

# fit the data
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_ver, y_ver))

# score the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Top 5 accuracy:', score[2])
