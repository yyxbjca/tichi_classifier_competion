import cv2
from skimage import transform
import numpy as np
"""
对这个数据集一张图片先进行水平翻转
得到两种表示，再配合0度，90度，180
度，270度的旋转，可以获得一张图的
八种表示
"""
def rotate18(image18,angle=90):
	"""
	image18 shape [weight,height,channle]
	"""
	shape=image18.shape
	image_rotated=np.zeros(shape,dtype='float32')
	for i in range(shape[2]):
		image=image18[:,:,i]
		image_rotated[:,:,i]=transform.rotate(image.astype(np.float64),angle).astype(np.float32)
	return image_rotated

def transpose18(image18,flag=1):
	shape=image18.shape
	image_transposed = np.zeros(shape,dtype='float32')
	for i in range(shape[2]):
		image=image18[:,:,i]
		image_transposed[:,:,i]=cv2.flip(image,flag,dst=None)
	return image_transposed
	
def batch_rotate(image,label):
	"""
	image :[batch_size,weight,height,channle]
	label : one-hot 
	"""
	shape=image.shape
	if(len(shape)!=4):
		return None
	else:
		rotate_90_image=np.array([rotate18(var,90) for var in image])
		rotate_180_image=np.array([rotate18(var,180) for var in image])
		rotate_270_image=np.array([rotate18(var,270) for var in image])
		
		repeat_label=np.array([label]*4)
		repeat_label=repeat_label.reshape(-1,label.shape[1])
		
		fin_image=np.concatenate([image,rotate_90_image,rotate_180_image,rotate_270_image],axis=0)
		
		return fin_image,repeat_label
    
def batch_data(image,label):
    #shape=image.shape
    image_h=np.array([transpose18(var,flag=1) for var in image])
    
    batch_image,batch_label=batch_rotate(image,label)
    batch_image_h,batch_label_h=batch_rotate(image_h,label)
    
    fuck_image=np.concatenate([batch_image,batch_image_h],axis=0)
    fuck_label=np.concatenate([batch_label,batch_label_h],axis=0)
    
    return fuck_image,fuck_label
    

def single_rotate(image):
    rotate_90_image=rotate18(image,90)
    rotate_180_image=rotate18(image,180)
    rotate_270_image=rotate18(image,270)
    
    
    fin_image=np.concatenate([image,rotate_90_image,rotate_180_image,rotate_270_image],axis=0)
    fin_image=fin_image.reshape(-1,image.shape[0],image.shape[1],image.shape[2])
  
    return fin_image

def single_data(image):
    image_h=transpose18(image,flag=1)

    
    batch_image=single_rotate(image)
    batch_image_h=single_rotate(image_h)
 
    fuck_image=np.concatenate([batch_image,batch_image_h],axis=0)
	
    return fuck_image
    
		
		