import numpy as np
import os
from skimage.filters import gabor
from skimage.filters import prewitt_h,prewitt_v
import cv2
from skimage.feature import hog
features_dir='hw5/features'
processed_dir='hw5/processed'
gabor_feature_dir=features_dir+'/'+'gabor'
if not os.path.exists(gabor_feature_dir):
    os.makedirs(gabor_feature_dir)
for file in os.listdir(processed_dir):
    image=cv2.cvtColor(np.load(processed_dir+'/'+file), cv2.COLOR_BGR2GRAY)
    filt_real, filt_imag = gabor(image, frequency=0.6)
    feature=np.concatenate((filt_real.flatten(),filt_imag.flatten()),axis=0)
    np.save(gabor_feature_dir+'/'+file,feature)
    
prewitt_feature_dir=features_dir+'/'+'prewitt'
if not os.path.exists(prewitt_feature_dir):
    os.makedirs(prewitt_feature_dir)
for file in os.listdir(processed_dir):
    image=cv2.cvtColor(np.load(processed_dir+'/'+file), cv2.COLOR_BGR2GRAY)
    edges_prewitt_horizontal = prewitt_h(image)
    edges_prewitt_vertical = prewitt_v(image)
    feature=np.concatenate((edges_prewitt_horizontal.flatten(),edges_prewitt_vertical.flatten()),axis=0)
    np.save(prewitt_feature_dir+'/'+file,feature)

gray_pexel_feature_dir=features_dir+'/'+'gray'
if not os.path.exists(gray_pexel_feature_dir):
    os.makedirs(gray_pexel_feature_dir)
for file in os.listdir(processed_dir):
    image=cv2.cvtColor(np.load(processed_dir+'/'+file), cv2.COLOR_BGR2GRAY)
    feature=image.flatten()
    np.save(gray_pexel_feature_dir+'/'+file,feature)

hog_feature_dir=features_dir+'/'+'hog'
if not os.path.exists(hog_feature_dir):
    os.makedirs(hog_feature_dir)
for file in os.listdir(processed_dir):
    image=cv2.cvtColor(np.load(processed_dir+'/'+file), cv2.COLOR_BGR2GRAY)
    feature= hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2))
    np.save(hog_feature_dir+'/'+file,feature)

!pip install torchxrayvision
import torchvision
import torch.nn as nn
import csv
model = xrv.models.DenseNet(weights="all")
model = nn.Sequential(
    model,
    nn.Softmax(),
)
dic={}
with open('Homework5/train.csv') as csvfile:
    datareader = csv.reader(csvfile)
    for samp in datareader:
        if samp[0]=='filename':
            continue
        dic[samp[0]]=int(samp[-1])
files=os.listdir(processed_dir)
files.sort()
x=np.zeros((len(dic),1,224,224))
y=np.zeros((len(dic)))
j=0
for i in range(len(files)):
    file=files[i]
    if file[:-4] in dic:
        x[j,0,:,:]=cv2.cvtColor(np.load(dir+'/'+file), cv2.COLOR_BGR2GRAY)
        y[j]=dic[file[:-4]]
        j+=1
x_feature=np.zeros((len(x),18))
for i in range(len(x)):
    x_feature[i]=model(torch.from_numpy(np.array([x[i]])).float())[0].detach().numpy()
np.save('/content/drive/My Drive/xrv_feature.npy',x_feature)

