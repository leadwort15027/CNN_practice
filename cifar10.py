#coding=utf-8
###########################################
#	python cifar10.py OUTPUT_FILE_NAME.h5 #
###########################################
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
import sys

#參數設定
batch_size = 64
nb_classes = 10
nb_epoch = 50
activation_function = 'relu'
drop_out = 0.4

#輸入照片維度
img_rows, img_cols = 32, 32
#RGB
img_channels = 3

#讀檔
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


#資料預處理
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#將輸入資料轉為分類維度
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#宣告模型型態
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation=activation_function, input_shape=X_train.shape[1:]))
model.add(Convolution2D(32, 3, 3, activation=activation_function))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(drop_out))

model.add(Convolution2D(64, 3, 3, activation=activation_function ))
model.add(Convolution2D(64, 3, 3, activation=activation_function))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(drop_out))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(drop_out))
model.add(Dense(nb_classes, activation='softmax'))

model.summary()
#設定目標函式
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#開始訓練模型
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test), shuffle=True)


loss = history.history.get('loss')
acc = history.history.get('acc')
val_loss = history.history.get('val_loss')
val_acc = history.history.get('val_acc')

plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss)), loss, label="Training")
plt.plot(range(len(val_loss)), val_loss, label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc, label='Training')
plt.plot(range(len(val_acc)), val_acc, label='Validation')
plt.title('Accuracy')
plt.savefig('00_firstModel.png', dpi=300, format='png')
plt.close


#儲存模型結構參數
model.save(sys.argv[1])
