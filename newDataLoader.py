# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 11:15:07 2018

1.sen1 4-5独立
2.sen2 10通道
3.4个指标
4.合并通道
5.数据增强
6.使用astype转换成float32形式，引用GPU加速
@author: freeze
"""


import h5py
import numpy as np

def get_index(array,index):
    P=[]
    for i in index:
        P.append(array[i])
    return P

def handle_data(handle,L):
    #handle is h5py object
    #L is a ordered list
    s1=list()
    s2=list()
    lab=list()
    s1.append(handle['sen1'][L[0]])
    s2.append(handle['sen2'][L[0]])
    lab.append(handle['label'][L[0]])
    for i in range(1,len(L)):
        if (L[i]==L[i-1]):
            s1.append(s1[-1])
            s2.append(s2[-1])
            lab.append(lab[-1])
        else:
            s1.append(handle['sen1'][L[i]])
            s2.append(handle['sen2'][L[i]])
            lab.append(handle['label'][L[i]])
    return np.array(s1,dtype='float32'),np.array(s2,dtype='float32'),np.array(lab,dtype='float32')

#归一化差异水指数
def NDWI(sen2):
    ndwi=(sen2[:,:,1]-sen2[:,:,6])/(sen2[:,:,1]+sen2[:,:,6])*0.5+0.5
    return ndwi.reshape(32,32,1)
#归一化植被指数
def NDVI(sen2):
    ndvi=(sen2[:,:,6]-sen2[:,:,2])/(sen2[:,:,6]+sen2[:,:,2])*0.5+0.5
    return ndvi.reshape(32,32,1)
#归一化差异建筑指数
def NDBI(sen2):
    ndbi=(sen2[:,:,8]-sen2[:,:,6])/(sen2[:,:,8]+sen2[:,:,6])*0.5+0.5
    return ndbi.reshape(32,32,1)
#归一化燃烧指数
def NBR(sen2):
    nbr=(sen2[:,:,6]-sen2[:,:,9])/(sen2[:,:,6]+sen2[:,:,9])*0.5+0.5
    return nbr.reshape(32,32,1)


def shrink(data,channle):
    #"data" shape like(weight,height,channle)
    return (1-np.exp(np.negative(data[:,:,channle]))).reshape(32,32,1)


def normalize(data):
    #which has shape like(weight,height)
    ma=np.max(data)
    mi=np.min(data)
    return (data-mi)/(ma-mi)

def nor(tensor):
    #tensor means shape (samples,weight,height,channle)
    s=tensor.shape
    for i in range(s[0]):
        for j in range(s[-1]):
            tensor[i,:,:,j]=normalize(tensor[i,:,:,j])
    return tensor
    


def indicator(sen1,sen2):
    #input shape(batch_size,width,height,channle)
    #return 6 indicator and fix channel*2
    wi=np.array([NDWI(var) for var in sen2])
    vi=np.array([NDVI(var) for var in sen2])
    bi=np.array([NDBI(var) for var in sen2])
    br=np.array([NBR(var) for var in sen2])
    
    #print (isi.shape)
    s4=np.array([shrink(var,4) for var in sen1])
    s5=np.array([shrink(var,5) for var in sen1])
    #print (s3.shape)
    return np.concatenate([s4,s5],axis=3),np.concatenate([wi,vi,bi,br],axis=3)
    
class Dataloader(object):
    def __init__(self,confidence,mode='c',epoch=2):
        self.train_dir='d:\\dataset\\training.h5'
        self.valid_dir='d:\\dataset\\smote.h5'
        self.sample_space=h5py.File(self.train_dir,'r')
        self.valid_sapce=h5py.File(self.valid_dir,'r')
        self.epoch=epoch
        self.confidence_list=confidence
        self.mode=mode
        self.valid_max_space=self.valid_sapce['label'].shape[0]
        self.random_list=list(self.confidence_list)
          
    def batch_train_sample(self,batch_size=100):
        if (self.epoch==0):
            return None,None
        else:
            if (len(self.random_list)<batch_size):
                self.epoch=self.epoch-1
                self.random_list=list(self.confidence_list)
                return self.batch_train_sample(batch_size=batch_size)
            else:
                choose_index=np.random.choice(len(self.random_list),size=batch_size,replace=False)
                choose_index.sort()
                choose_index=list(choose_index)
                choose=get_index(self.random_list,choose_index)
                if (self.mode=='c'):
                    #降采样
                    s1=self.sample_space['sen1'][choose]
                    s2=self.sample_space['sen2'][choose]
                    lab=self.sample_space['label'][choose]
                else:
                    #重采样
                    s1,s2,lab=handle_data(self.sample_space,choose)
                s3,s4=indicator(s1,s2)   
                sam=np.concatenate([s3,s2,s4],axis=3)
                for c in sorted(choose_index,reverse=True):
                    self.random_list.pop(c)

                return sam.astype(np.float32),lab.astype(np.float32)
            
    def batch_valid_sample(self,batch_size=20):
        choose=np.random.choice(self.valid_max_space,batch_size,replace=False)
        choose.sort()
        choose=list(choose)
        
        s1=self.valid_sapce['sen1'][choose]
        s2=self.valid_sapce['sen2'][choose]

        s3,s4=indicator(s1,s2)
        #s2=nor(s2)
        lab=self.valid_sapce['label'][choose]
        sam=np.concatenate([s3,s2,s4],axis=3)
        #x,y=batch_data(sam,lab)
        #return x.astype(np.float32),y.astype(np.float32)
        return sam.astype(np.float32),lab.astype(np.float32)
    
    def get_valid_sample(self):
        h=h5py.File("d:\\dataset\\valid.h5",'r')
        s1=h['sen1'][:]
        s2=h['sen2'][:]

        s3,s4=indicator(s1,s2)
        #s2=nor(s2)
        sample=np.concatenate([s3,s2,s4],axis=3)
        label=h['label'][:]
        return sample.astype(np.float32),label.astype(np.float32)
            
