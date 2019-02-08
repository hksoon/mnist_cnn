#########################################################
# program name : training
# __author__ : 'kyoung soon hwang'
# date : 2019.01.25
# program : Trains a simple convnet on the MNIST dataset.
#########################################################

from __future__ import print_function
import keras
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.callbacks import CSVLogger
import json
import mnist_cnn.data.data as d
import mnist_cnn.networks.model_build as m

def TankConfig():
    batch_size = 128
    num_classes = 10
    epochs = 20
    return batch_size, num_classes, epochs

jsonPath = "datasets/model/2_models_b.txt"


batch_size, num_classes, epochs = TankConfig()


(x_train, y_train), (x_test, y_test), input_shape = d.data_load()


model = m.cnn_model(jsonPath, num_classes, input_shape)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=sgd,
			  metrics=['accuracy'])


csv_logger = CSVLogger('log/perform.csv', append=True, separator=';')
filepath="log/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, csv_logger ]


model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs=epochs,
          verbose=0,
          callbacks=callbacks_list,
		  validation_data=(x_test, y_test)) 


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


json_string = model.to_json()
open('datasets/model/cnn_mnist.json', 'w').write(json_string)
model.save('log/keras_mnist_cnn_model.h5')

def reader(stream):
    for line in stream:
        yield json.loads(line)

with open('datasets/model/cnn_mnist.json') as src, open('datasets/model/out_cnn.json', 'w') as dst:
    for line in reader(src):
        dst.write(json.dumps(line, indent=4))

'''
if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train CNN to Minst.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the Mnist dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TankConfig()    

    # Create model
    if args.command == "train":
        model = networks.model_buld.(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = model.infer(mode="inference", config=config, model_dir=args.logs)
'''