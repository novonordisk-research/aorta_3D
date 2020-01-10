#!/usr/bin/env python

"""
3D U-net for aorta anatomy segmentation
"""

from __future__ import print_function
import numpy as np
import os
import argparse
import sys

# set seed:
from numpy.random import seed
#seed(123)
from tensorflow import set_random_seed
#set_random_seed(123)

import random
#random.seed(123)

from PIL import Image
import SimpleITK as sitk
from skimage.transform import resize
from skimage.morphology import binary_erosion
from skimage.measure import label, regionprops

import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics

from keras.utils import to_categorical
from keras.preprocessing import image
from keras import optimizers
from skimage import io

import tensorflow as tf

################################################################################
#   PARSE COMMANDLINE OPTIONS
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-training_data', '--training_data',  help="path to training data file")
parser.add_argument('-out_path', '--out_path',  help="path to training data file")
parser.add_argument('-gpu', '--gpu', action="store_true", help="Use GPU, default = False", default=False )
parser.add_argument('-n_epochs', '--n_epochs', help="Number of epochs, default = 300", default=200 )
parser.add_argument('-learning_rate', '--learning_rate', help="Learning rate, default = 0.0001", default=0.00001 )
parser.add_argument('-seed', '--seed', help="Random number initialization, default = 123", default=123 )
args = parser.parse_args()

# get data path:
if args.training_data != None:
    print("# Path: " + args.training_data )
    training_data = args.training_data
else:
    sys.stderr.write("Please specify path to training data!\n")
    sys.exit(2)

# get output path:
if args.out_path != None:
    print("# Path: " + args.out_path )
    out_path = args.out_path
else:
    sys.stderr.write("Please specify path to output!\n")
    sys.exit(2)

# get number of training epochs:
try:
    print("# Number of epochs: " + str(args.n_epochs ))
    n_epochs = int(args.n_epochs)
except:
    sys.stderr.write("Problem defining number of epochs!\n")
    sys.exit(2)

# get learning rate:
try:
    print("# Learning rate: " + str(args.learning_rate ))
    learning_rate = float(args.learning_rate)
except:
    sys.stderr.write("Problem defining learning rate!\n")
    sys.exit(2)

# set seed:
try:
    print("# Setting seed: " + str(args.seed ))
    seed(int(args.seed))
    set_random_seed(int(args.seed))
    random.seed(int(args.seed))
except:
    sys.stderr.write("Problem setting seed!\n")
    sys.exit(2)

################################################################################
#   SELECT A FREE GPU
################################################################################
if args.gpu==True:
    import GPUtil
    GPU = GPUtil.getAvailable(order = "memory")[0]
    GPU = str(GPU)

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

################################################################################
#   READ DATA
################################################################################

train_im=[]
train_labels=[]


ids=["20"]

for i in ids:
    print(i)
    # read aorta:
    im = sitk.ReadImage(training_data + "ID" + i + "_aorta.nii.gz")
    im = sitk.GetArrayFromImage(im)

    # downsize:
    im=resize(im,(64,64,64),1,preserve_range=True)

    # binarize:
    im[im>0]=1.0

    # save:
    train_im.append(im)

    # rotate and add:
    im=np.rot90(im,2,(0,1))
    train_im.append(im)
    im=np.rot90(im,2,(1,2))
    train_im.append(im)
    im=np.rot90(im,2,(0,2))
    train_im.append(im)

    # read atlas:
    a = sitk.ReadImage(training_data + "ID" + i + "_atlas.nii.gz")
    a = sitk.GetArrayFromImage(a)

    # downsize:
    a=resize(a,(64,64,64),0,preserve_range=True)

    # separate classes:
    l=np.zeros((64,64,64,6))
    #b=im-a

    for idx in range(0,6):
        tmp=np.zeros(a.shape)
        tmp[np.where(a==(idx+1))]=1.0
        l[:,:,:,idx]=tmp

    # save:
    train_labels.append(l)
    # rotate and add:
    l=np.rot90(l,2,(0,1))
    train_labels.append(l)
    l=np.rot90(l,2,(1,2))
    train_labels.append(l)
    l=np.rot90(l,2,(0,2))
    train_labels.append(l)


train_im=np.array(train_im)
train_im=np.reshape(train_im,(train_im.shape[0],64,64,64,1))
train_labels=np.array(train_labels)

################################################################################
#   LOAD NETWORK
################################################################################
print("# Setting up network...")

# modified dice coefficient separating classes:---------------------------------

def dice_coef(y_true, y_pred):

    y_true_1 = K.flatten(y_true[:,:,:,:,0])
    y_pred_1 = K.flatten(y_pred[:,:,:,:,0])
    d_1 = (2 * K.sum(y_true_1 * y_pred_1) +1 )/ (K.sum(y_true_1) + K.sum(y_pred_1) + 1)

    y_true_2 = K.flatten(y_true[:,:,:,:,1])
    y_pred_2 = K.flatten(y_pred[:,:,:,:,1])
    d_2 = (2 * K.sum(y_true_2 * y_pred_2) +1 )/ (K.sum(y_true_2) + K.sum(y_pred_2) + 1)

    y_true_3 = K.flatten(y_true[:,:,:,:,2])
    y_pred_3 = K.flatten(y_pred[:,:,:,:,2])
    d_3 = (2 * K.sum(y_true_3 * y_pred_3) +1 )/ (K.sum(y_true_3) + K.sum(y_pred_3) + 1)

    y_true_4 = K.flatten(y_true[:,:,:,:,3])
    y_pred_4 = K.flatten(y_pred[:,:,:,:,3])
    d_4 = (2 * K.sum(y_true_4 * y_pred_4) +1 )/ (K.sum(y_true_4) + K.sum(y_pred_4) + 1)

    y_true_5 = K.flatten(y_true[:,:,:,:,4])
    y_pred_5 = K.flatten(y_pred[:,:,:,:,4])
    d_5 = (2 * K.sum(y_true_5 * y_pred_5) +1 )/ (K.sum(y_true_5) + K.sum(y_pred_5) + 1)

    y_true_6 = K.flatten(y_true[:,:,:,:,5])
    y_pred_6 = K.flatten(y_pred[:,:,:,:,5])
    d_6 = (2 * K.sum(y_true_6 * y_pred_6) +1 )/ (K.sum(y_true_6) + K.sum(y_pred_6) + 1)

    return (d_1+d_2+d_3+d_4+d_5+d_6)/6.0

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# define network:---------------------------------------------------------------

inputs = Input((None,None,None, 1))
conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)
up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)
up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)
up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)
conv10 = Conv3D(6, (3, 3, 3), activation='sigmoid', padding='same')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])
model.compile(optimizer=Adam(lr = learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

################################################################################
#   TRAINING
################################################################################

checkpointer = ModelCheckpoint(filepath=out_path + '3d_model_weights.hdf5', verbose=1, save_best_only=True)

history=model.fit(train_im, train_labels, epochs=n_epochs, batch_size=1, validation_split=0.1, callbacks=[checkpointer])

v_loss=np.array(history.history['val_loss'])
loss=np.array(history.history['loss'])
d_coef=np.array(history.history['dice_coef'])
v_d_coef=np.array(history.history['val_dice_coef'])

np.savez(out_path + 'model_loss.npz', np.array(v_loss),np.array(loss),np.array(d_coef),np.array(v_d_coef))

print("# Done!")
