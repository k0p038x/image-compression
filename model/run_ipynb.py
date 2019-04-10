#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import import_ipynb
import end_to_end_compression
import tensorflow as tf

with tf.device('/gpu:0'):
    end_to_end_compression.run()

