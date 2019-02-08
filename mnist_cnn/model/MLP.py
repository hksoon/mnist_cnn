'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils

import data_load as d
import numpy as np
import json
from collections import OrderedDict


batch_size = 250
num_classes = 10
epochs = 12

#입력 이미지 차원
img_rows, img_cols = 28, 28

impath = "media/train-images/"
datInf = "media/train-labels.csv"
jsonPath = "media/files/2_models.txt"

#데이터 전처리
#데이터, 훈련, 검증, 시험 데이터 분리하기
(x_train, y_train), (x_test, y_test) = d.data_load(impath, datInf)

print(len(x_test))
# 데이터셋 전처리
x_train = x_train.reshape(48000, 784).astype('float32') / 255.0
x_test = x_test.reshape(11999, 784).astype('float32') / 255.0

# 원핫인코딩 (one-hot encoding) 처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 훈련셋과 검증셋 분리
x_val = x_train[:42000] # 훈련셋의 30%를 검증셋으로 사용
x_train = x_train[42000:]
y_val = y_train[:42000] # 훈련셋의 30%를 검증셋으로 사용
y_train = y_train[42000:]

model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=12, batch_size=100, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
print('## training loss and acc ##')
print(hist.history['loss'])
print(hist.history['acc'])

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=125)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

# 7. 모델 사용하기
xhat = x_test[0:1]
yhat = model.predict(xhat)
print('## yhat ##')
print(yhat)

# 6. 모델 저장하기
from keras.models import load_model
model.save('mnist_mlp_model.h5')