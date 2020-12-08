import numpy as np
import os
from skimage.filters import gabor
from skimage.filters import prewitt_h,prewitt_v
import cv2
from skimage.feature import hog
import csv
import random
from sklearn.svm import SVC
import copy
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
features_dir='hw5/features'
processed_dir='hw5/processed'
gabor_feature_dir=features_dir+'/'+'gabor' 
prewitt_feature_dir=features_dir+'/'+'prewitt'
gray_pixel_feature_dir=features_dir+'/'+'gray'
hog_feature_dir=features_dir+'/'+'hog'
dic={}
categ_feat={}
location={}
min_age=float('inf')
max_age=float('-inf')
loc_index=0
with open('Homework5/train.csv') as csvfile:
    datareader = csv.reader(csvfile)
    for samp in datareader:
        if samp[0]=='filename':
            continue
        dic[samp[0]]=int(samp[-1])
        if samp[2]:
            age=int(samp[2])
            min_age=min(min_age,age)
            max_age=max(max_age,age)
        else:
            age=''
        loc=samp[3]
        if loc and loc not in location:
            location[loc]=loc_index
            loc_index+=1
        categ_feat[samp[0]]=(samp[1],age,loc)
files=list(dic.keys())
files.sort()

gender_features_x=np.zeros((len(dic),2))
y=np.zeros((len(dic)))
i=0
for file in files:
    if categ_feat[file][0]=='F':      
        gender_features_x[i,0]=1
    else:
        gender_features_x[i,1]=1
    y[i]=dic[file]
    i+=1

age_features_x=np.zeros((len(dic),10))
i=0
for file in files:
    age=categ_feat[file][1]
    if age:
        age_features_x[i,int((age-min_age)/(max_age-min_age+1)*10)]=1
    i+=1

loc_features_x=np.zeros((len(dic),len(location)))
i=0
for file in files:
    loc=categ_feat[file][2]
    if loc:
        loc_features_x[i,location[loc]]=1
    i+=1
    
gabor_features_x=np.zeros((len(dic),48672))

i=0
for file in files:
    gabor_features_x[i]=np.load(gabor_feature_dir+'/'+file+'.npy')
    i+=1

hog_features_x=np.zeros((len(dic),11664))
i=0
for file in files:
    hog_features_x[i]=np.load(hog_feature_dir+'/'+file+'.npy')
    i+=1 

prewitt_features_x=np.zeros((len(dic),48672))
i=0
for file in files:
    prewitt_features_x[i]=np.load(prewitt_feature_dir+'/'+file+'.npy')
    i+=1

gray_pixel_features_x=np.zeros((len(dic),24336))
i=0
for file in files:
    gray_pixel_features_x[i]=np.load(gray_pixel_feature_dir+'/'+file+'.npy')
    i+=1

xrv_features_x=np.load('/content/drive/My Drive/xrv_feature.npy')

feature_ls=[loc_features_x,age_features_x,gray_pexel_features_x,hog_features_x,gender_features_x,prewitt_features_x,gabor_features_x,xrv_feature]

for i in range(1,9):
    if i==1:
        x=copy.deepcopy(feature_ls[0])
    else:
        x=np.concatenate((x,feature_ls[i-1]),axis=1)
start=time.time()
#implement adaboost classifier on all features
result=cross_validate(AdaBoostClassifier(),x,y,cv=5)['test_score']
print(sum(result)/len(result))
print('takes time {} s'.format(time.time()-start))


