# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:26:37 2019

@author: freeze
"""


import op
from newDataLoader import indicator
import numpy as np
import h5py
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def repeat_max_frequence(x):
    hash_max=np.max(x)
    L=[]
    for i in range(len(x)):
        if x[i]==hash_max:
            L.append(i)
    return L

def softmax(x):
    shape=x.shape 
    m_row_max=x.max(axis=1).reshape(shape[0],1)
    nx=x-m_row_max
    m_exp=np.exp(nx)
    m_exp_sum=np.sum(m_exp,axis=1)
    

    m_exp_sum=m_exp_sum.reshape(shape[0],1)
    
    y=m_exp/m_exp_sum
    return y


if __name__=='__main__':
    #hypamater  batch_size=200  learning_rate=0.0005
    batch_size=50
    WIDTH=28
    HEIGHT=28
    DEEP=16
    learning_rate=0.00001
    iteration=0
    #define tensorflow graph
    
    sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver=tf.train.import_meta_graph('resnet-101-step2/model.ckpt-800.meta')
    saver.restore(sess,tf.train.latest_checkpoint('resnet-101-step2/'))
    
     #default graph
    graph=tf.get_default_graph()
    
    with tf.device("/gpu:0"):
        x=graph.get_tensor_by_name("Placeholder:0")
        y=graph.get_tensor_by_name("Placeholder_1:0")
        LR=graph.get_tensor_by_name("Placeholder_2:0")
        train_or_not=graph.get_tensor_by_name("Placeholder_3:0")
        
        y_=graph.get_tensor_by_name("res_net_v2_101/dense/Relu:0")
        loss=graph.get_tensor_by_name("conv/AddN:0")
        
        
        pred_number=graph.get_tensor_by_name("ArgMax:0")
        correct_pred=graph.get_tensor_by_name("Equal:0")
        accuracy=graph.get_tensor_by_name("Mean:0")
        
        train=graph.get_operation_by_name("conv/Adam")
      
    path='D:\\dataset\\round2_test_b_20190211.h5'
    h=h5py.File(path,'r')
    s2=h['sen2'].value
    s1=h['sen1'].value
    s3,s4=indicator(s1,s2)
    
    fix=np.concatenate([s3,s2,s4],axis=3)
    
    num=s2.shape[0]
    ans=np.zeros((num,17))
    
    ids=[]
    pairs=[]
    for i in range(num):
        img=fix[i]
        fin_img=op.single_data(img)
        seed=np.random.randint(low=0,high=4)
        k=sess.run(y_,feed_dict={x:fin_img[:,seed:seed+28,seed:seed+28],train_or_not:False})  
        
        yx=softmax(k)
        yy=np.mean(yx,axis=0)
        """
        ml=repeat_max_frequence(yy)
        if(len(ml)==1):
            ans[i][ml[0]]=1
        else:
            print("id %d " % i)
            print (ml)
            ids.append(i)
            pairs.append(ml)
        """
        ans[i]=yy
        print ("create %d "% i)
    
    np.savetxt("101_p1323.csv",ans,delimiter=',')