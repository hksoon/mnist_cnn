'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K, optimizers
import data_load as d
import numpy as np
import json
from collections import OrderedDict


batch_size = 250
num_classes = 10
epochs = 50

#입력 이미지 차원
img_rows, img_cols = 28, 28

impath = "media/train-images/"
datInf = "media/train-labels.csv"
jsonPath = "media/files/2_models_b.txt"

#데이터 전처리
#데이터, 훈련, 검증, 시험 데이터 분리하기
(x_train, y_train), (x_test, y_test) = d.data_load(impath, datInf)

if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#입력 데이터 확인
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#모델 구성하기

with open(jsonPath, encoding="utf-8") as data_file:
    data=json.load(data_file, object_pairs_hook=OrderedDict)

model = Sequential()
for i in range(len(data)):
    opT = "op_" + str(i)
    op = data[opT]['operator_title']
    if (op == "Convolution2D"):
        actF = data[opT]['activation'].strip()
        outDim = int(data[opT]['number_of_filters'].strip())
        Fcol = int(data[opT]['filter_columns'].strip())
        Frow = int(data[opT]['filter_rows'].strip())
        if (opT == "op_1"):
            model.add(Conv2D(outDim, kernel_size=(Fcol, Frow), activation=actF, input_shape=input_shape))
        else:
            model.add(Conv2D(outDim,kernel_size=(Fcol, Frow), activation=actF))
    elif (op == "Maxpooling2D"):
        FS = int(data[opT]['pool_size'].strip())
        model.add(MaxPooling2D(pool_size=(FS, FS)))
    elif (op == "Dropout"):
        op = float(data[opT]['fraction_to_drop'].strip())
        model.add(Dropout(op))
    elif (op == "Dense"):
        actF = data[opT]['activation'].strip()
        outDim = int( data[opT]['output_dimentions'].strip())
        model.add(Dense(outDim, activation= actF))
    elif (op == "Flatten"):
        model.add(Flatten())
    elif (op == "OutputLayer"):
        actF = data[opT]['activation'].strip()
        model.add(Dense(num_classes, activation=actF))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#모델 학습 설정
model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=sgd,
			  metrics=['accuracy'])

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#모델 학습
model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs=epochs,
          verbose=0,
          callbacks=callbacks_list,
		  validation_data=(x_test, y_test))

#학습 결과 확인
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#모델 저장하기
json_string = model.to_json()
open('cnn_mnist.json', 'w').write(json_string)
model.save('keras_mnist_cnn_model.h5')