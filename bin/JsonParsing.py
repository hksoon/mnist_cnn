import json
import sys
from collections import OrderedDict

jsonPath = "media/files/2_models.txt"

def jsonParing(impath):
    original = sys.stdout
    data=[]
    sys.stdout = open('media/rd.txt', 'w')
    with open(impath, encoding="utf-8") as data_file:
        data = json.load(data_file, object_pairs_hook=OrderedDict)
    #print(data)
    print("model = Sequential()")
    for i in range(len(data)):
        opT = "op_"+str(i)
        op = data[opT]['operator_title']
        if(op == "Convolution2D"):
            actF = data[opT]['activation'].strip()
            outDim = data[opT]['number_of_filters'].strip()
            Fcol = data[opT]['filter_columns'].strip()
            Frow = data[opT]['filter_rows'].strip()
            if(opT == "op_1"):
                print("model.add(Conv2D(",outDim, ",", "kernel_size=(", Fcol,",", Frow,"), activation='", actF,"', input_shape=input_shape))")
            else:
                #model.add(Conv2D(32, (3, 3), activation='relu'))
                print("model.add(Conv2D(", outDim, ",", "(",Fcol, ",", Frow, "), activation='", actF,"'))")
        elif (op == "Maxpooling2D"):
            FS = data[opT]['pool_size'].strip()
            print("model.add(MaxPooling2D(pool_size=(",FS,",",FS,")))")
        elif (op == "Dropout"):
            op = data[opT]['fraction_to_drop'].strip()
            print("model.add(Dropout(", op,"))")
        elif (op == "Dense"):
            actF = data[opT]['activation'].strip()
            outDim = data[opT]['output_dimentions'].strip()
            print("model.add(Dense(", outDim, ", activation='", actF, "'))")
        elif (op == "Flatten"):
            print("model.add(Flatten())")
        elif (op == "OutputLayer"):
            actF = data[opT]['activation'].strip()
            print("model.add(Dense(num_classes,"," activation='", actF, "'))")
    sys.stdout = original
    return
#jsonParing(jsonPath)