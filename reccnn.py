
# coding: utf-8

# In[96]:


import numpy as np
import cv2
from keras.layers import Conv2D, Input
from keras.models import Model


# In[97]:


class RecCNN:
    
    # fx, fy : scaling factors 
    def __init__(self, fx, fy):
        self.fx = fx 
        self.fy = fy
    
    def interpolation(self, input_img): 
        output_img = input_img
        output_img = cv2.resize(src = input_img, dst = output_img, dsize = (0,0), fx = self.fx, fy = self.fy, interpolation = cv2.INTER_CUBIC)
        return output_img
    
    def sisr(self, first):
        
        # Layers
        conv1 = Conv2D(filters=64, kernel_size=(5,5), activation='relu', padding='same')(first)
        conv2 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(conv1)
        last = Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding = 'same')(conv2)
        return last
    
        


# In[98]:


# Testing RecCNN Model

input_img = Input(shape=(28, 28, 1))
rec_cnn = RecCNN(2,2)
model = Model(input_img, rec_cnn.sisr(input_img))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# In[99]:


from keras.datasets import mnist

(x_train,_), (x_test,_) = mnist.load_data()
x_train = x_train[:10,:,:]
x_test = x_test[:1,:,:]
print(x_train.shape)
print(x_test.shape)


# In[100]:


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape)
print(x_test.shape)


# In[101]:


model.fit(x_train, x_train, epochs=5, batch_size=8, shuffle=True, validation_data=(x_test, x_test))


# In[102]:


op_img = model.predict(x_test)
op_img = op_img * 255
x_test = x_test * 255

