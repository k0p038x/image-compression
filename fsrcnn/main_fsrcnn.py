# -*- coding: utf-8 -*-


from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import import_ipynb
from reccnn import RecCNN
from comcnn import ComCNN

import matplotlib.image as mpimg
from PIL import Image
import os

def compressThis(x_input):
    # x_input dimension (num, x, y, c)
    # num - number of images
    num = x_input.shape[0]
    ans = []
    for i in range(num):
        x_single = x_input[i]
        # compressing single image
        mpimg.imsave(os.getcwd()+'/org.JPEG', x_single)
        tmp = Image.open(os.getcwd()+'/org.JPEG')
        tmp.save(os.getcwd()+'/com.JPEG',"JPEG", optimize=True, quality=65)
        out_single = mpimg.imread(os.getcwd()+'/com.JPEG')
        ans.append(out_single)
        
    return np.array(ans)


from tqdm import tqdm
import os
import matplotlib.image as mpimg
import cv2
import tensorflow as tf
with tf.device('/gpu:0'):

  loc = os.getcwd()+'/Subset16k/Subset16k'

  count1 = 0
  X = []
  for i in (os.listdir(loc)):
    count1 = count1 + 1
    if count1%100==0:
      print(count1)
    
    if count1==5000:
      break
    
    im_loc = loc + '/'+i
    img = cv2.imread(im_loc)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X.append(img)
      

  X = np.array(X)
  print(X.shape)

  split = int(0.8*X.shape[0])

  x_train = X[:split,:,:,:]
  x_valid = X[split:,:,:,:]
  print("Train shape : " + str(x_train.shape))
  print("Valid shape : " + str(x_valid.shape))

  x_train_lr = []

  for i in range(x_train.shape[0]):
    img = x_train[i]
    img = cv2.resize(img, (64, 64))
    x_train_lr.append(img)

  x_train_lr = np.array(x_train_lr)

  x_valid_lr = []

  for i in range(x_valid.shape[0]):
    img = x_valid[i]
    img = cv2.resize(img, (64, 64))
    x_valid_lr.append(img)

  x_valid_lr = np.array(x_valid_lr)


  x_train = x_train.astype('float32')
  x_valid = x_valid.astype('float32')
  x_train_lr = x_train_lr.astype('float32')
  x_valid_lr = x_valid_lr.astype('float32')

  x_train = x_train / 255
  x_valid = x_valid / 255
  x_train_lr = x_train_lr / 255
  x_valid_lr = x_valid_lr / 255

  print("Train LR shape : " + str(x_train_lr.shape))
  print("Valid LR shape : " + str(x_valid_lr.shape))


  from keras.layers import Input
  from keras.callbacks import ModelCheckpoint
  import tensorflow as tf


  inp2 = Input(shape=(64, 64, 3))
  rec_cnn = RecCNN(3)
  model_reccnn = Model(inp2, rec_cnn.fsrcnn(inp2))
  model_reccnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
  checkpointr = ModelCheckpoint(filepath='reccnn.weights.best.hdf5', save_best_only=True, verbose=1)
  model_reccnn.summary()

  inp1 = Input(shape=(256, 256, 3))
  com_cnn = ComCNN(3)
  model_comcnn = Model(inp1, model_reccnn(com_cnn.compact(inp1)))
  model_comcnn.layers[4].trainable = False
  model_comcnn.compile(optimizer='adam', loss='mean_squared_error')
  checkpointc = ModelCheckpoint(filepath='comcnn.weights.best.hdf5', save_best_only=True, verbose=1)
  model_comcnn.summary()


  # Initial training instead of random weights

  model_reccnn.fit(x=x_train_lr, y=x_train, validation_data=(x_valid_lr, x_valid),epochs=25, shuffle=True, verbose=1, batch_size=8, callbacks=[checkpointr])
  model_comcnn.fit(x=x_train, y=x_train, validation_data=(x_valid, x_valid),epochs=25, shuffle=True, verbose=1, batch_size=8, callbacks=[checkpointc])

  count = 15

  for i in tqdm(range(count)):
    
    
    # calculating xm using comcnn
    upto_comcnn = Model(model_comcnn.input, model_comcnn.layers[3].output)
    xm = upto_comcnn.predict(x_train)
    xm_valid = upto_comcnn.predict(x_valid)
    

    # train RecCNN
    model_reccnn.fit(x=xm, y=x_train, validation_data=(xm_valid, x_valid),epochs=10, shuffle=True, verbose=1, batch_size=8, callbacks=[checkpointr])
    
    # train ComCNN
    model_comcnn.fit(x=x_train, y=x_train, validation_data=(x_valid, x_valid),epochs=10, shuffle=True, verbose=1, batch_size=8, callbacks=[checkpointc])

