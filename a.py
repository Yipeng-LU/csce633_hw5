import numpy as np
import cv2
import os
train_dir='Homework5/train'
test_dir='Homework5/test'
dirs=[train_dir,test_dir]
min_width=float('inf')
min_height=float('inf')
for dir in dirs:
    for file in os.listdir(dir):  
        img=cv2.imread(dir+'/'+file)
        height,width,dim=img.shape
        min_width=min(width,min_width)
        min_height=min(height,min_height)
a=min(min_width,min_height)
processed_dir='hw5/processed'
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
for dir in dirs:
    for file in os.listdir(dir):  
        img=cv2.imread(dir+'/'+file)
        height,width,dim=img.shape
        diff=abs(width-height)
        if width>height:
            processed_img=cv2.resize(img[:,diff//2:-diff//2],(a,a))
        elif height>width:
            processed_img=cv2.resize(img[diff//2:-diff//2],(a,a))
        else:
            processed_img=cv2.resize(img,(a,a))  
        np.save(processed_dir+'/'+'{}.npy'.format(file),processed_img)
