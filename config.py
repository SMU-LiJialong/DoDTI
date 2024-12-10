#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#****************************
# Select which GPU to use
#***************************

import tensorflow as tf

def config_gpu(id):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[id],True)
        tf.config.experimental.set_visible_devices(gpus[id],'GPU')   
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

def config_cpu(num_threads_inter=0,num_threads_intra=0):
    """
    A vlaue of 0 means the system picks an approprite number.
    #### ARGUMENTS
    - num_threads_inter, number of threads used for parallelism between independent operations.
    - num_threads_intra, number of threads used within an individual op for parallelism.
    """
    tf.config.threading.set_inter_op_parallelism_threads(num_threads_inter)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads_intra)
