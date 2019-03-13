# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 13:56:50 2018

@author: freeze
"""

import tensorflow as tf
from bn_wrapper import bn_layer


def block(x,out_dim,is_training):
    """
    if input tensor dim equals output dim,use identity shortcut
    else use 1*1 conv increse dim
    """
    if(x.shape.as_list()[-1]==out_dim):
        short_cut=tf.identity(x)
        c1=tf.nn.relu(tf.layers.conv2d(x,kernel_size=3,filters=out_dim,strides=1,padding='SAME'))
    else:
        short_cut=tf.layers.conv2d(x,kernel_size=1,filters=out_dim,strides=2,padding='SAME')
        c1=tf.nn.relu(tf.layers.conv2d(x,kernel_size=3,filters=out_dim,strides=2,padding='SAME'))
    c2=tf.layers.conv2d(c1,kernel_size=3,filters=out_dim,strides=1,padding='SAME')
    c3=tf.nn.relu(c2+short_cut)
    return tf.nn.relu(bn_layer(c3,is_training))

def bottleneck(x,mid_filter,is_training):
    """
    bottleneck residual v1 arch
    increase and decrease tensor x dim
    """
    fin_dim=x.shape.as_list()[-1]
    L=bn_layer(x,is_training)
    c1=tf.nn.relu(tf.layers.conv2d(L,kernel_size=1,filters=mid_filter,strides=1,padding='SAME'))
    c2=tf.nn.relu(tf.layers.conv2d(c1,kernel_size=3,filters=mid_filter,strides=1,padding='SAME'))
    c3=tf.layers.conv2d(c2,kernel_size=3,filters=fin_dim,strides=1,padding='SAME')
    return tf.nn.relu(x+c3)
    


def residual_101(x,is_training):
    # x means input tensor with shape[None,32,32,channle]
    L2=tf.contrib.layers.l2_regularizer(5e-4)
    with tf.variable_scope(name_or_scope="res_net_v2_101",reuse=tf.AUTO_REUSE):
        bn1=bottleneck(x,32,is_training)
        bn2=bottleneck(bn1,32,is_training)
        bn2_1=bottleneck(bn2,32,is_training)
        
        block_64=block(bn2_1,64,is_training)
        bn3=bottleneck(block_64,64,is_training)
        bn4=bottleneck(bn3,64,is_training)
        bn4_1=bottleneck(bn4,64,is_training)
        bn4_2=bottleneck(bn4_1,64,is_training)
        
        block_128=block(bn4_2,128,is_training)
        bn5=bottleneck(block_128,128,is_training)
        bn6=bottleneck(bn5,128,is_training)
        bn6_1=bottleneck(bn6,128,is_training)
        bn6_2=bottleneck(bn6_1,128,is_training)
        bn6_3=bottleneck(bn6_2,128,is_training)
        bn6_4=bottleneck(bn6_3,128,is_training)
        bn6_5=bottleneck(bn6_4,128,is_training)
        bn6_6=bottleneck(bn6_5,128,is_training)
        bn6_7=bottleneck(bn6_6,128,is_training)
        bn6_8=bottleneck(bn6_7,128,is_training)
        bn6_9=bottleneck(bn6_8,128,is_training)
        bn6_10=bottleneck(bn6_9,128,is_training)
        bn6_11=bottleneck(bn6_10,128,is_training)
        bn6_12=bottleneck(bn6_11,128,is_training)
        bn6_13=bottleneck(bn6_12,128,is_training)
        bn6_14=bottleneck(bn6_13,128,is_training)
        bn6_15=bottleneck(bn6_14,128,is_training)
        bn6_16=bottleneck(bn6_15,128,is_training)
        bn6_17=bottleneck(bn6_16,128,is_training)
        bn6_18=bottleneck(bn6_17,128,is_training)
        bn6_19=bottleneck(bn6_18,128,is_training)
        bn6_20=bottleneck(bn6_19,128,is_training)
        bn6_21=bottleneck(bn6_20,128,is_training)

        
        block_256=block(bn6_21,256,is_training)
        bn6=bottleneck(block_256,256,is_training)
        bn7=bottleneck(bn6,256,is_training)
        
        #fc3
        flat=tf.contrib.layers.flatten(bn7)
        fc3=tf.layers.dense(flat,units=17,activation=tf.nn.relu,kernel_regularizer=L2)
        return fc3
    
    
    