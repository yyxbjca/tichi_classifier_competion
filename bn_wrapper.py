# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:41:55 2019

@author: Blayneyyx
"""

import tensorflow as tf


def bn_layer(x,is_training,eps=1e-4):
    shape=x.get_shape().as_list()
    #the number of kernel
    out_channle=shape[-1]
    
    gamma=tf.Variable(tf.ones([out_channle]))
    beta=tf.Variable(tf.zeros([out_channle]))
    #global
    pop_mean = tf.Variable(tf.zeros([out_channle]), trainable=False)
    pop_var = tf.Variable(tf.ones([out_channle]), trainable=False)
    
    def batch_normal_training():
        decay=0.999
        batch_mean,batch_var=tf.nn.moments(x,[0,1,2])
        
        train_mean=tf.assign(pop_mean,pop_mean*decay+batch_mean*(1-decay))
        train_var=tf.assign(pop_var,pop_var*decay+batch_var*(1-decay))
        with tf.control_dependencies([train_mean,train_var]):
            return tf.nn.batch_normalization(x,batch_mean,batch_var,beta,gamma,eps)
    def batch_normal_inference():
        return tf.nn.batch_normalization(x,pop_mean,pop_var,beta,gamma,eps)
    
    bn_output=tf.cond(is_training,batch_normal_training,batch_normal_inference)
    return tf.nn.relu(bn_output)
        
        