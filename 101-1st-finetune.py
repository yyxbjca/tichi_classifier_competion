# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:10:30 2019

@author: freeze
"""


import op
from newDataLoader import Dataloader
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def up_sampling(path):
    h=h5py.File(path,'r')
    gaint_label=h['label'].value
    #每类的样本数目
    class_num=np.sort(np.sum(gaint_label,axis=0))
    
    down_samp_num=int(class_num[-1])
    
    roman_label=np.argmax(gaint_label,axis=1)
    #init 
    L=[]
    for i in range(17):
        L.append(list())
    #fill in L
    for it in range(len(roman_label)):
        L[roman_label[it]].append(it)
    
    BAL=[]
    for part in L:
        A=np.array(part)
        if(len(part)>down_samp_num):
            #no repeat
            ch=np.random.choice(len(part),down_samp_num,replace=False)
        else:
            #repeat   
            ch=np.random.choice(len(part),down_samp_num,replace=True)
        B=A[ch]
        B.sort()
        B=list(B)
        BAL.append(B)
    return BAL


def valid_acc(sess,valid_x,y):
    div=y.shape[0]
    count=0
    for i in range(div):
        temp_x=valid_x[i]
        #print (temp_x.shape)
        fin_x=op.single_data(temp_x)
      
        #print (fin_x.shape)
        seed=np.random.randint(low=0,high=32-WIDTH)
        y_=sess.run(pred_number,feed_dict={x:fin_x[:,seed:seed+WIDTH,seed:seed+WIDTH],train_or_not:False})
        y_=np.argmax(np.bincount(y_))
        label=np.argmax(y[i])
        if(y_==label):
            count=count+1
    return count/div

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
    saver=tf.train.import_meta_graph('resnet-101-step1/model.ckpt-215000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('resnet-101-step1/'))
    
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
      
    #
    paths='d:\\dataset\\training.h5'
    indces=np.array(up_sampling(paths)) #transform to one dim
    one_indces=indces.reshape(-1)
    one_indces=list(one_indces)
    one_indces.sort()
    
    #data IO
    dl=Dataloader(confidence=list(one_indces),mode='f',epoch=5)
    valid_x,valid_y=dl.get_valid_sample()
    
    train_acc=[]
    test_acc=[]
    loss_set=[]
    
    max_acc=0
    
    saver=tf.train.Saver(max_to_keep=1)
    

  
    while True:
        #batch from dataloader
        bx,by=dl.batch_train_sample(batch_size=batch_size)
        if (bx is None):
            break
        bvx,bvy=dl.batch_valid_sample(batch_size=10)
        tx=np.concatenate([bx,bvx],axis=0)
        ty=np.concatenate([by,bvy],axis=0)
        b_x,b_y=op.batch_data(tx,ty)
        seed=np.random.randint(low=0,high=32-WIDTH)
        

        _,lose,a=sess.run([train,loss,accuracy],feed_dict={x:b_x[:,seed:seed+WIDTH,seed:seed+WIDTH],y:b_y,LR:learning_rate,train_or_not:True})
        print ("epoch %d step %d train batch acc = %f loss= %f" %(dl.epoch,iteration,a,lose))
        if (iteration%100==0):
            b=valid_acc(sess,valid_x,valid_y)
            print ("valid acc %f" % b)
            if(b>max_acc):
                max_acc=b
                saver.save(sess,'resnet-101-step2/model.ckpt',global_step=iteration)
            test_acc.append(b)
        train_acc.append(a)
        loss_set.append(lose)
        iteration=iteration+1
        
        
    
    print ("*****************")
    print ("max test acc is %f" % max_acc)
    print ("*****************")
    
    plt.plot(train_acc,'k-',label='Trainacc')
    plt.plot(test_acc,'r--',label='testacc')
    plt.title('acc per generation')
    plt.xlabel('generation')
    plt.ylabel('acc')
    plt.legend(loc='lower right')
    plt.show()
    
    plt.plot(loss_set,'k-')
    plt.title('loss per generation')
    plt.xlabel('generation')
    plt.ylabel('loss')
    plt.show()