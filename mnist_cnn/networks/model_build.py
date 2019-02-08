from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from collections import OrderedDict

import json


# 모델 구성하기
def cnn_model(jsonPath, num_classes,  input_shape):
    """

    :param input_shape: 
    :type num_classes: object
    """
    with open(jsonPath, encoding="utf-8") as data_file:
        data = json.load(data_file, object_pairs_hook=OrderedDict)

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
                model.add(Conv2D(outDim, kernel_size=(Fcol, Frow), activation=actF))
        elif (op == "Maxpooling2D"):
            FS = int(data[opT]['pool_size'].strip())
            model.add(MaxPooling2D(pool_size=(FS, FS)))
        elif (op == "Dropout"):
            op = float(data[opT]['fraction_to_drop'].strip())
            model.add(Dropout(op))
        elif (op == "Dense"):
            actF = data[opT]['activation'].strip()
            outDim = int(data[opT]['output_dimentions'].strip())
            model.add(Dense(outDim, activation=actF))
        elif (op == "Flatten"):
            model.add(Flatten())
        elif (op == "OutputLayer"):
            actF = data[opT]['activation'].strip()
            model.add(Dense(num_classes, activation=actF))
    return model
