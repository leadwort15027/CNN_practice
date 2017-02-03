#coding=utf-8
###########################################
#	python cifar10.py OUTPUT_FILE_NAME.h5 #
###########################################
#from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.models import load_model
import sys

#看幾張圖後調整參數
batch_size = 50
#分類數
nb_classes = 10
#訓練次數
nb_epoch = 15

#輸入照片維度
img_rows, img_cols = 32, 32
#RGB
img_channels = 3

#讀檔
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#將輸入資料轉為分類維度
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#宣告模型型態
model = Sequential()
#增加一個32個3*3的卷積核的卷積層
model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=X_train.shape[1:]))
#設定啟動函式是 ReLu
model.add(Activation('relu'))
#增加一個32個3*3的卷積核的卷積層
model.add(Convolution2D(32, 3, 3))
#設定啟動函式是 ReLu
model.add(Activation('relu'))
#增加一個2*2的池化層
model.add(MaxPooling2D(pool_size=(2, 2)))
#增加一個0.25的dropout層
model.add(Dropout(0.25))

#增加一個64個3*3的卷積核的卷積層
model.add(Convolution2D(64, 3, 3, border_mode='same'))
#設定啟動函式是 ReLu
model.add(Activation('relu'))
#增加一個64個3*3的卷積核的卷積層
model.add(Convolution2D(64, 3, 3))
#設定啟動函式是 ReLu
model.add(Activation('relu'))
#增加一個2*2的池化層
model.add(MaxPooling2D(pool_size=(2, 2)))
#增加一個0.25的dropout層
model.add(Dropout(0.25))


#把2D資料攤平
model.add(Flatten())
#增加一個N-512的full connected層
model.add(Dense(512))
#設定啟動函式是 ReLu
model.add(Activation('relu'))
#增加一個0.5的dropout層
model.add(Dropout(0.5))
#增加一個N-輸出維度大小的full connected層
model.add(Dense(nb_classes))
#設定啟動函式是 softmax
model.add(Activation('softmax'))

#設定目標函式
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#資料預處理
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#開始訓練模型
model.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_test, Y_test),shuffle=True)

#model.save("demo.h5")
model.save(sys.argv[1])
