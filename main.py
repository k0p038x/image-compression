#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import import_ipynb
from reccnn import RecCNN
from comcnn import ComCNN


# In[2]:


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
        


# In[3]:


import os
import matplotlib.image as mpimg

loc = os.getcwd()+'/DSet-Color'
X = []
for i in os.listdir(loc):
    im_loc = loc + '/'+i
    img = mpimg.imread(im_loc)
    X.append(img)

X = np.array(X)
print(X.shape)


# In[4]:


x_train = X[:190,:,:,:]
x_test = X[190:,:,:,:]
print(x_train.shape)
print(x_test.shape)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train/=255
x_train*=2
x_train-=1
x_test/=255
x_test*=2
x_test-=1


# In[34]:


from keras.layers import Input
from keras.callbacks import ModelCheckpoint
import tensorflow as tf


inp2 = Input(shape=(25,50,3))
rec_cnn = RecCNN(3)
model_reccnn = Model(inp2, rec_cnn.sisr(inp2))
model_reccnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
checkpointr = ModelCheckpoint(filepath='reccnn.weights.best.hdf5', save_best_only=True, verbose=2)
model_reccnn.summary()

inp1 = Input(shape=(375,500,3))
com_cnn = ComCNN(3)
model_comcnn = Model(inp1, model_reccnn(com_cnn.compact(inp1)))
model_comcnn.layers[4].trainable = False
model_comcnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
checkpointc = ModelCheckpoint(filepath='comcnn.weights.best.hdf5', save_best_only=True, verbose=2)
model_comcnn.summary()


# In[12]:


count = 5
x_valid = x_test
while count > 0:
    
    # calculating cm using comcnn
    upto_comcnn = Model(model_comcnn.input, model_comcnn.layers[3].output)
    xm = upto_comcnn.predict(x_train)
    xm_valid = upto_comcnn.predict(x_valid)
    # xm = xm + 1
    # xm = xm / 2
    # xm = xm * 255
    # xm = compressThis(xm)
    # xm = xm / 255
    # xm = xm * 2
    # xm = xm - 1

    # xm_valid = xm_valid + 1
    # xm_valid = xm_valid / 2
    # xm_valid = xm_valid * 255
    # xm_valid = compressThis(xm_valid)
    # xm_valid = xm_valid / 255
    # xm_valid = xm_valid * 2
    # xm_valid = xm_valid - 1

    print("Xm array updated")
#     print(xm.shape)
    
    # train RecCNN
    model_reccnn.fit(x=xm, y=x_train, epochs=10, shuffle=True, verbose=1, batch_size=8, callbacks=[checkpointr], validation_data = (xm_valid, x_valid))
    print("RecCNN model trained")
    # train ComCNN
    model_comcnn.fit(x=x_train, y=x_train, epochs=10, shuffle=True, verbose=1, batch_size=8, callbacks=[checkpointc], validation_data = (x_valid, x_valid))
    print("ComCNN model trained")
    count = count - 1

