# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 15:44:36 2019

@author: freeze
"""

import tensorflow as tf
from bn_wrapper import bn_layer



def conv2d(x,ksize,kstride,kfilter,trainable):
    """
    x: tensorflow input tensor
    kfilter: number of output for conv layer
    ksize: kernel size
    kstride: stride
    """
    L=tf.nn.relu(bn_layer(x,trainable))
    c1=tf.nn.relu(tf.layers.conv2d(L,kernel_size=ksize,filters=kfilter,strides=kstride,padding='SAME'))
    return c1



def conv_concate(x,growth_rate,trainable,name):
    with tf.variable_scope(name):
        l=conv2d(x,kfilter=growth_rate,ksize=3,kstride=1,trainable=trainable)
        c=tf.concat([l,x],3)
    return c


def dense_block(x,trainable,layers=12,grow_rate=6):
    for i in range(layers):
        strs=("dense_block_%d"%i)
        x=conv_concate(x,growth_rate=grow_rate,trainable=trainable,name=strs)
    return x
   

     
def transition(x,name,shrink_dim=16):
    with tf.variable_scope(name_or_scope=name):
        c1=tf.nn.relu(tf.layers.conv2d(x,kernel_size=3,filters=shrink_dim,strides=1,padding='SAME'))
        c2=tf.layers.max_pooling2d(c1,pool_size=(2,2),strides=(2,2))
    return c2
 

       
def DenseNet(x,trainable,gr=12):
    L2=tf.contrib.layers.l2_regularizer(5e-4)
    
    with tf.variable_scope('block1'):
        x=dense_block(x,grow_rate=gr,trainable=trainable)
        x=transition(x,'transition1')
    
    with tf.variable_scope('block2'):
        x=dense_block(x,grow_rate=gr,trainable=trainable)
        x=transition(x,'transition2')
        
    with tf.variable_scope('block3'):
        x=dense_block(x,grow_rate=gr,trainable=trainable)

    L=tf.nn.relu(bn_layer(x,trainable))
    flat=tf.contrib.layers.flatten(L)
    fc=tf.layers.dense(flat,units=17,activation=tf.nn.relu,kernel_regularizer=L2)
    return fc
    