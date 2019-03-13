# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:00:05 2019

@author: freeze
"""


from newDataLoader import indicator
import tensorflow as tf
import h5py
import numpy as np
import op




sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver=tf.train.import_meta_graph('resnet-101-step1/model.ckpt-215000.meta')
saver.restore(sess,tf.train.latest_checkpoint('resnet-101-step1/'))

#default graph
graph=tf.get_default_graph()

#get placeholder and variable
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



path='D:\\dataset\\round2_test_a_20190121.h5'
h=h5py.File(path,'r')
s2=h['sen2'].value
s1=h['sen1'].value
s3,s4=indicator(s1,s2)
fix=np.concatenate([s3,s2,s4],axis=3)

num=s2.shape[0]
ans=np.zeros((num,17))

for i in range(num):
    
    img=fix[i]
    
    fin_img=op.single_data(img)
    seed=np.random.randint(low=0,high=4)
    
    y_=sess.run(pred_number,feed_dict={x:fin_img[:,seed:seed+28,seed:seed+28],train_or_not:False}) 
    y_=np.argmax(np.bincount(y_))
    
    ans[i][y_]=1
    print ("create %d ans" % i)
    
np.savetxt("2-10.csv",ans,delimiter=',')

