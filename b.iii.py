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
from sklearn.model_selection import cross_validate
features_dir='hw5/features'
processed_dir='hw5/processed'
gabor_feature_dir=features_dir+'/'+'gabor' 
prewitt_feature_dir=features_dir+'/'+'prewitt'
gray_pexel_feature_dir=features_dir+'/'+'gray'
hog_feature_dir=features_dir+'/'+'hog'
dic={}
categ_feat={}
location={}
min_age=float('inf')
max_age=float('-inf')
loc_index=0
with open('/content/drive/MyDrive/Homework5/train.csv') as csvfile:
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

xrv_features_x=np.load('/content/drive/My Drive/xrv_feature.npy')

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

num_bins=4
age_features_x=np.zeros((len(dic),num_bins))
i=0
for file in files:
    age=categ_feat[file][1]
    if age:
        age_features_x[i,int((age-min_age)/(max_age-min_age+1)*num_bins)]=1
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

gray_pexel_features_x=np.zeros((len(dic),24336))
i=0
for file in files:
    gray_pexel_features_x[i]=np.load(gray_pexel_feature_dir+'/'+file+'.npy')
    i+=1
    
feature_ls=[age_features_x,loc_features_x,gray_pexel_features_x,hog_features_x,gender_features_x,prewitt_features_x,gabor_features_x,xrv_features_x]

feature_num=range(1,9)
acc=[]
start=time.time()
for i in range(1,9):
    if i==1:
        x=copy.deepcopy(feature_ls[0])
    else:
        x=np.concatenate((x,feature_ls[i-1]),axis=1)
    ac=0
    for c in range(1,7):
        result=cross_validate(SVC(C=c,gamma='scale',random_state=0),x,y,cv=5)['test_score']
        ac=max(ac,sum(result)/len(result))
    acc.append(sum(result)/len(result))
print(acc)
print('takes time {} s'.format(time.time()-start))

plt.plot(feature_num,acc)
plt.xlabel('number of features')
plt.ylabel('accuracy')
plt.show()

accs=[]
feature_selected=[]
start=time.time()
for i in range(1,9):
    best_acc=0
    if i==1:
        for j in range(8):
            acc=0
            for c in range(1,7):
                result=cross_validate(SVC(C=c,gamma='scale',random_state=0),feature_ls[j],y,cv=5)['test_score']
                acc=max(acc,sum(result)/len(result))
            if acc>best_acc:
                best_feature=j
                best_acc=acc
        accs.append(best_acc)
        feature_selected.append(best_feature)
        x=feature_ls[best_feature]
    else:
        for j in range(8):
            if j not in feature_selected:
                acc=0
                for c in range(1,7):
                    result=cross_validate(SVC(C=c,gamma='scale',random_state=0),np.concatenate((x,feature_ls[j]),axis=1),y,cv=5)['test_score']
                    acc=max(acc,sum(result)/len(result))
                if acc>best_acc:
                    best_feature=j
                    best_acc=acc
        accs.append(best_acc)
        feature_selected.append(best_feature)
        x=np.concatenate((x,feature_ls[best_feature]),axis=1)
print(feature_selected)
print(accs)
print('takes time {} s'.format(time.time()-start))

plt.plot(feature_num,accs)
plt.xlabel('number of features')
plt.ylabel('accuracy')
plt.show()
