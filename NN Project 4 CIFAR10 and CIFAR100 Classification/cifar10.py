'''
Project: Classifying CIFAR10
Author: Joseph Bentivegna
Instructor: Christopher Curro

Overview:
This project involved creating a CNN to classify the CIFAR10 dataset that consists of pictures
of 10 objects.  The data was imported from keras and split into three different subsets for training,
verification, and testing. The model consists of four blocks, each with a topology similar to
[CONV BN CONV BN POOL]. The final layer is a dense layer with a softmax to compute the final classification.
The model trains in approximately 33s per epoch on a 1080TI and achieved accuracy of 89% on the test dataset
(but 90.5% on the validation) after 200 epochs.

All references are cited inline.
'''

# imports
import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
I had used a batch size of 32 for almost all of my testing until the final
iteration of my code where I tried 64 and it turned out to do much better.

I chose to use 200 epochs just because it seemed like a reaonsable standard
for accuracy.  On the final iteraion it hit 90% accuracy after like 160 epochs
so funning it for 200 was probably unnecessary.

Some of the new pre-processing steps that I chose to implement were data augmentation
and chanel-wise normalization. I found that these both increased the accuracy of the model
by prevent overfitting.
'''

batch_size = 64
num_classes = 10
epochs = 200
imageProcessing = True
normalization = True

# input image dimensions
img_rows, img_cols = 32, 32

# load the data and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_ver, y_train, y_ver = train_test_split(x_train, y_train, test_size=5000)

# when data augmentation is enabled, we initialize the datagen object and fit it to the train set
if imageProcessing:
    datagen = ImageDataGenerator(
              rotation_range=15,
              width_shift_range=0.1,
              height_shift_range=0.1,
              horizontal_flip=True)
    datagen.fit(x_train)

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
x_ver /= 255
x_test /= 255

# when normalization is enabled, we normalize each channel
# values from: https://github.com/facebook/fb.resnet.torch/issues/180
if normalization:

  mus = [0.4914, 0.4822, 0.4465]
  sig = [0.247, 0.243, 0.261]

  for i in range(3):
      x_train[:][:][i] = (x_train[:][:][i] - mus[i]) / sig[i]
      x_ver[:][:][i] = (x_ver[:][:][i] - mus[i]) / sig[i]
      x_test[:][:][i] = (x_test[:][:][i] - mus[i]) / sig[i]

'''
This is another feature that I added at the last minute. I noticed that many cutting
edge approaches to CIFAR10 used variable learning rates so I implemented one of their topologies
and added it to my model.

Implementation from: https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
Also credit to Jacob Maarek for exhaustively testing good learning rate values.
'''
def lr_schedule(epoch):
    lrate=0.001
    if epoch > 150:
        lrate = 0.0000625/4
    elif epoch > 125:
        lrate = 0.0000625/2
    elif epoch > 100:
        lrate = 0.0000625
    elif epoch > 80:
        lrate = 0.000125
    elif epoch > 60:
        lrate = 0.00025
    elif epoch > 40:
        lrate = 0.0005
    elif epoch > 20:
        lrate = 0.00075
    return lrate

print(x_ver.shape[0], 'ver samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_ver = keras.utils.to_categorical(y_ver, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# constants
l2weight = 0.0001

# full model
'''
Curro I know you're going to write D.R.Y. on this and im sorry but I'm already
handing this in late and I don't want to take more time to functionalize it.

The idea for this topology was inspired from: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
However, many changes were made including:
    -Adding 4 block layers of convolutions
    -Using an eLu activation function
    -Adding BatchNormaliztion between layers to normalize weights and decrease computation time
    -Varying pooling sizes that correlate with filter size
    -Eliminating all dense layers before the final softmax layer
'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight), input_shape=input_shape))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(l2weight)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(5, 5), padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

# build the graph
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=lr_schedule(0)), metrics=['accuracy'])

# initialize the learning rate schedule and add learning rate reducer callbacks to help on plateaus
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-8)
callbacks = [lr_reducer, lr_scheduler]

# fit the graph
if imageProcessing:
  model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=int(x_train.shape[0])/batch_size, epochs=epochs, verbose=1, callbacks=callbacks, validation_data=(x_ver, y_ver))
else:
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_ver, y_ver))

# score the model on the test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
