#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import import_ipynb
import train
import tensorflow as tf

with tf.device('/gpu:0'):
    train.run()

