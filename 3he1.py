# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:23:51 2019

@author: Blayneyyx
"""

import numpy as np
import csv
from newDataLoader import indicator
import tensorflow as tf
import h5py
#import op


def read(name):
    cv=csv.reader(open(name))
    L=np.zeros((4835,17))
    c=0
    for row in cv:
        L[c]=row
        c=c+1
    return L



a=read("101_p131.csv")
b=read("101_p132.csv")
c=read("101_p133.csv")
d=read("101_p134.csv")
e=read("101_p135.csv")
f=read("101_p136.csv")
g=read("101_p137.csv")
h=read("101_p138.csv")
i=read("101_p139.csv")
j=read("101_p1310.csv")
k=read("101_p1311.csv")
l=read("101_p1312.csv")
m=read("101_p1313.csv")
n=read("101_p1314.csv")
o=read("101_p1315.csv")
p=read("101_p1316.csv")
q=read("101_p1317.csv")
r=read("101_p1318.csv")
s=read("101_p1319.csv")
t=read("101_p1320.csv")
u=read("101_p1321.csv")
v=read("101_p1322.csv")
w=read("101_p1323.csv")

z=a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w

fz=z/23.0

np.savetxt("res101_pro_23.csv",fz,delimiter=',')  
"""
def repeat_max_frequence(x):
    hash_max=np.max(x)
    L=[]
    for i in range(len(x)):
        if x[i]==hash_max:
            L.append(i)
    return L

sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver=tf.train.import_meta_graph('resnet-101-step2/model.ckpt-800.meta')
saver.restore(sess,tf.train.latest_checkpoint('resnet-101-step2/'))

#default graph
graph=tf.get_default_graph()

#get placeholder and variable
with tf.device("/gpu:0"):
    # #get placeholder and variable
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





a=read('131.csv')
b=read('132.csv')
c=read('133.csv')
d=read('134.csv')
e=read('135.csv')
f=read('136.csv')
g=read('137.csv')
h=read('138.csv')
j=read('139.csv')

aa=read('1310.csv')
ab=read('1311.csv')
ac=read('1312.csv')
ad=read('1313.csv')
ae=read('1314.csv')
af=read('1315.csv')
ag=read('1316.csv')
ah=read('1317.csv')
ai=read('1318.csv')
aj=read('1319.csv')
ak=read('1320.csv')
am=read('1321.csv')
an=read('1322.csv')
az=read('1323.csv')



path='D:\\dataset\\round2_test_b_20190211.h5'
hh=h5py.File(path,'r')
s2=hh['sen2'].value
s1=hh['sen1'].value
s3,s4=indicator(s1,s2)
fix=np.concatenate([s3,s2,s4],axis=3)




ans=np.zeros_like(a)
#z=a+b+c+d+e+f+g+h+j+aa+ab+ac+ad+ae+af+ag+ah+ai+aj+ak+am+an+az
z=a+b+c+d+e+f+g+h+j

count=0  
qus=[]
for i in range(a.shape[0]):
    index=repeat_max_frequence(z[i])
    if(len(index)==1):
        ans[i][index[0]]=1
    else:
        qus.append(i)
        print ("hard to decide!!!!!!!!!!!!!!!!!!!")
        count=count+1
        seed=np.random.randint(low=0,high=4)
        img=fix[i]
        img=np.expand_dims(img,axis=0)
        temp=sess.run(pred_number,feed_dict={x:img[:,seed:seed+28,seed:seed+28],train_or_not:False})
        ans[i][temp[0]]=1
    print ("create %d th " % i)

np.savetxt("bbb.csv",ans,delimiter=',')       
"""
    
    
