#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import import_ipynb
import test
import test2
import tensorflow as tf

with tf.device('/gpu:0'):
    test2.run()

