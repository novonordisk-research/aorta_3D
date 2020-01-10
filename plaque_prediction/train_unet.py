#!/usr/bin/env python

"""
U-net for 2D plaque segmentation
"""

from __future__ import print_function
import numpy as np
import os
import argparse
import sys

# set seed:
from numpy.random import seed
from tensorflow import set_random_seed
import random

from PIL import Image
import Augmentor

import keras
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

from scipy.misc import imresize
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
parser.add_argument('-batch_size', '--batch_size', help="Mini-batch size, default = 20", default=20 )
parser.add_argument('-n_epochs', '--n_epochs', help="Number of epochs, default = 1000", default=1000 )
parser.add_argument('-learning_rate', '--learning_rate', help="Learning rate, default = 0.0001", default=0.0001 )
parser.add_argument('-augmentor_prob', '--augmentor_prob', help="Probability of data augmentation, default = 0.3", default=0.3 )
parser.add_argument('-model', '--model', help="Model for fine tuning, default = None", default=None )
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

# get batch size:
try:
    print("# Batch size: " + str(args.batch_size) )
    batch_size = int(args.batch_size)
except:
    sys.stderr.write("Problem defining batch size!\n")
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

# get learning rate:
try:
    print("# Augmentor probablility: " + str(args.augmentor_prob ))
    augmentor_prob = float(args.augmentor_prob)
except:
    sys.stderr.write("Problem defining probablility for data augmentor!\n")
    sys.exit(2)

# get learning rate:
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

train_im=np.load(training_data)['arr_0']
train_labels=np.load(training_data)['arr_1']
val_im=np.load(training_data)['arr_2']
val_labels=np.load(training_data)['arr_3']

train_im=np.reshape(train_im,(-1,train_im.shape[1],train_im.shape[2],1))
train_labels=np.reshape(train_labels,(-1,train_im.shape[1],train_im.shape[2],1))
val_im=np.reshape(val_im,(-1,train_im.shape[1],train_im.shape[2],1))
val_labels=np.reshape(val_labels,(-1,train_im.shape[1],train_im.shape[2],1))

train_im = train_im.astype(np.float32)
val_im = val_im.astype(np.float32)
train_labels=train_labels.astype(np.float32)
val_labels=val_labels.astype(np.float32)

################################################################################
#   LOAD NETWORK
################################################################################
print("# Setting up network...")

# define dice coefficient:------------------------------------------------------

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# define network:---------------------------------------------------------------

inputs = Input((None,None, 1))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
conv10 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])
model.compile(optimizer=Adam(lr = learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

# load pre-trained weights:
if args.model != None:
    model.load_weights(args.model)

################################################################################
#   DATA AUGMENTATION
################################################################################

images = list(zip(list(train_im.reshape(train_im.shape[0],train_im.shape[1],train_im.shape[2])),
                    list(train_labels.reshape(train_labels.shape[0],train_labels.shape[1],train_labels.shape[2]))))


# wrap it all up in an infinite loop handling dimensions:
def do_data_gen(g):
    while True:
        a_im=next(g)
        a_im_n=np.array(a_im)
        train_in=a_im_n[:,0,:,:].reshape(a_im_n.shape[0],a_im_n.shape[2],a_im_n.shape[3],1)
        train_out=a_im_n[:,1,:,:].reshape(a_im_n.shape[0],a_im_n.shape[2],a_im_n.shape[3],1)

        # apply random brightness transform (only on original image):
        r=np.random.uniform(0.0, 1.0)
        if r<=augmentor_prob:
            # select scaling factor:
            s=np.random.uniform(0.7, 1.3)
            train_in=s*train_in

        yield train_in,train_out

def make_data_gen_pipeline(images,augmentor_prob):
    # initialize augmentor:
    p = Augmentor.DataPipeline(images)

    # define augmentation:
    p.skew_tilt(probability=augmentor_prob, magnitude=0.2)
    p.random_distortion(probability=augmentor_prob, grid_width=16, grid_height=16, magnitude=8)
    p.rotate(probability=augmentor_prob, max_left_rotation=5, max_right_rotation=5)
    p.flip_top_bottom(probability=augmentor_prob)
    p.flip_left_right(probability=augmentor_prob)
    p.zoom(probability=augmentor_prob,  min_factor=0.9, max_factor=1.1)

    # define generator with the right batch size:
    g = p.generator(batch_size=batch_size)

    data_gen=do_data_gen(g=g)

    return data_gen

################################################################################
#   TRAINING NETWORK
################################################################################
print("# Training network...")

checkpointer = ModelCheckpoint(filepath=out_path + 'model_weights.hdf5', verbose=1, save_best_only=True)

train_steps = train_im.shape[0]//batch_size
val_steps = val_im.shape[0]//batch_size

def train_model(model, images, augmentor_prob, checkpointer, val_im, val_labels, n_epochs, train_steps, val_steps):
    # data augmentation:
    data_gen=make_data_gen_pipeline(images,augmentor_prob)

    # model training:
    history=model.fit_generator(generator = data_gen,
                        epochs = n_epochs,
                        steps_per_epoch = train_steps,
                        validation_data = (val_im,val_labels),
                        validation_steps = val_steps,
                        callbacks=[checkpointer])

    # extract performance:
    return history.history['val_loss'],history.history['loss'],history.history['dice_coef'],history.history['val_dice_coef']

# train:
v_loss,loss,d_coef,v_d_coef=train_model(images=images,
                        model=model,
                        augmentor_prob=augmentor_prob,
                        checkpointer=checkpointer,
                        val_im=val_im,
                        val_labels=val_labels,
                        n_epochs=n_epochs,
                        train_steps=train_steps,
                        val_steps=val_steps)
# save:
v_loss.extend(list(vl))
loss.extend(list(l))
d_coef.extend(list(d))
v_d_coef.extend(list(vd))


# save losses in npz format:
np.savez(out_path + 'model_loss.npz', v_loss,loss,d_coef,v_d_coef)

print("# Done!")
