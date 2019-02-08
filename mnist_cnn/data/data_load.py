#####################
__author__ = 'momo'
####################

import numpy as np
import os.path
from pylab import *

from scipy import misc
import glob

from pandas import DataFrame
import pandas as pd


def get_imlist(path):
    """ Returns a list of filenames for
all jpg images in a directory. """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


Dataset ='mnist'
InMM = '1'
dataSplit = '80_20_0'
RecordSize ='1000000'

inputlist = {}
outputPos = 1
im={}
im_rows, imgs_cols = 28, 28
y_pos=0

impath = "media/train-images/"
datInf = "media/train-labels.csv"
   #x_train, y_train, x_valid, y_valid

def data_load(impath, datInf, y_pos=0):
    #################
    # image raw data transform x값 -> input values
    ################
    imlist = get_imlist(impath)
    #im = array(Image.open(imlist[0]).convert())
    imnbr = len(imlist)
    print(imnbr)
    jg = []

    for i in range(imnbr):
        jg.append(imread(imlist[i]))

    x_trainT = np.asarray(jg)
    x_dat = int(imnbr * 0.8)
    print(x_dat)

    x_train = x_trainT[0:x_dat, :]
    x_test = x_trainT[x_dat:-1, :]
    #print(" x_train ", x_train, "\n", "x_test", x_test)
    #print('Importing done...', x_train.shape)

    ##################
    # y값 -> target values
    ####################
    data = pd.read_csv(datInf)
    dlen = len(data)
    df = DataFrame(data)
    y_pos = df.ix[:, -1]
    y_trainT = list(y_pos)
    y_dat = int(imnbr * 0.8)
    y_train = y_trainT[0:x_dat]
    y_test = y_trainT[x_dat:-1]
    #print(len(y_train))
    #print(" y_train ", y_train, "\n", "y_test", y_test)
    return (x_train, y_train), (x_test, y_test)


