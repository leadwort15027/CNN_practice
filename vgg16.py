from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import cifar100, cifar10
import cv2, numpy as np

classes = 10
batch_size = 50
nb_epoch = 100

(X_train, y_train),(X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_train, classes)


conv1 = Activation("relu")
conv2 = Activation("relu")
conv3 = Activation("relu")
conv4 = Activation("relu")
conv5 = Activation("relu")
conv6 = Activation("relu")
conv7 = Activation("relu")
conv8 = Activation("relu")
conv9 = Activation("relu")
conv10= Activation("relu")
conv11= Activation("relu")
conv12= Activation("relu")
conv13= Activation("relu")


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=X_train.shape[1:]))
model.add(Convolution2D(64, 3, 3))
model.add(conv1)
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3))
model.add(conv2)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3))
model.add(conv3)
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3))
model.add(conv4)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3))
model.add(conv5)
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3))
model.add(conv6)
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3))
model.add(conv7)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3))
model.add(conv8)
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3))
model.add(conv9)
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3))
model.add(conv10)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3))
model.add(conv11)
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3))
model.add(conv12)
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3))
model.add(conv13)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy")

X_train = X_train.astype("float32")
X_test = X_train.astype("float32")
X_train /= 255
X_test /=255

model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_test,Y_test),shuffle=True)


model.save("vgg16.h5")
