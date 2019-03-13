# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:24:14 2018

@author: freeze
"""
"""
res101
"""

from Resnet_v1 import residual_101
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
    
    down_samp_num=int(class_num[13])
    
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

def down_sampling(path):
    h=h5py.File(path,'r')
    gaint_label=h['label'].value
    #每类的样本数目
    class_num=np.sum(gaint_label,axis=0)
    
    down_samp_num=int(np.min(class_num))
    
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
        if(len(part)>down_samp_num):
            A=np.array(part)
            ch=np.random.choice(len(part),down_samp_num,replace=False)
            B=A[ch]
            B.sort()
            B=list(B)
            BAL.append(B)
        else:
            part.sort()
            BAL.append(part)
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
    learning_rate=0.0005
    iteration=0
    #define tensorflow graph
    
    
    x=tf.placeholder(tf.float32,shape=[None,WIDTH,HEIGHT,DEEP])
    y=tf.placeholder(tf.float32,shape=[None,17])
    LR=tf.placeholder(tf.float32)
    train_or_not=tf.placeholder(tf.bool)
    with tf.device("/gpu:0"):
        y_=residual_101(x,is_training=train_or_not)
        
        with tf.variable_scope(name_or_scope='conv',reuse=tf.AUTO_REUSE):
            entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_))
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss=tf.add_n([entropy_loss]+regularization_losses)
            optimizer=tf.train.AdamOptimizer(learning_rate=LR,beta1=0.5)
            train=optimizer.minimize(loss)
        pred_number=tf.argmax(y_,1)
        correct_pred=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))   
        
    #sess init 
    init=tf.global_variables_initializer()
    
    sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    
    #
    paths='d:\\dataset\\training.h5'
    indces=np.array(up_sampling(paths)) #transform to one dim
    one_indces=indces.reshape(-1)
    one_indces=list(one_indces)
    one_indces.sort()
    
    #data IO
    dl=Dataloader(confidence=list(one_indces),mode='f',epoch=20)
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
        
        #train
        if dl.epoch>10:
            learning_rate=0.0005
        elif dl.epoch>5:
            learning_rate=0.0001
        else:
            learning_rate=0.00005
        _,lose,a=sess.run([train,loss,accuracy],feed_dict={x:b_x[:,seed:seed+WIDTH,seed:seed+WIDTH],y:b_y,LR:learning_rate,train_or_not:True})
        print ("epoch %d step %d train batch acc = %f loss= %f" %(dl.epoch,iteration,a,lose))
        if (iteration%100==0):
            b=valid_acc(sess,valid_x,valid_y)
            print ("valid acc %f" % b)
            if(b>max_acc):
                max_acc=b
                saver.save(sess,'resnet-101-paper/model.ckpt',global_step=iteration)
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