import numpy as np
import os
from skimage.filters import gabor
from skimage.filters import prewitt_h,prewitt_v
import cv2
from skimage.feature import hog
import csv
features_dir='hw5/features'
processed_dir='hw5/processed'
gabor_feature_dir=features_dir+'/'+'gabor' 
prewitt_feature_dir=features_dir+'/'+'prewitt'
gray_pexel_feature_dir=features_dir+'/'+'gray'
hog_feature_dir=features_dir+'/'+'hog'
def dst(a,b):
    ret=0
    for i in range(len(a)):
        ret+=(a[i]-b[i])**2
    return ret
def fisher(x,y):
    x_mean=np.mean(x,axis=0)
    x1=np.zeros((int(np.sum(y)),x.shape[1]))
    x0=np.zeros((int(x.shape[0]-np.sum(y)),x.shape[1]))
    p0=0
    p1=0
    for i in range(len(x)):
        if y[i]==0:
            x0[p0]=x[i]
            p0+=1
        else:
            x1[p1]=x[i]
            p1+=1
    x0_mean=np.mean(x0,axis=0)
    x1_mean=np.mean(x1,axis=0)
    part1=0
    for i in x0:
        part1+=dst(i,x0_mean)
    part2=0
    for i in x1:
        part2+=dst(i,x1_mean)
    return (part1+part2)/(dst(x0_mean,x_mean)+dst(x1_mean,x_mean))
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
gender_features_x=np.zeros((len(dic),2))
gender_features_y=np.zeros((len(dic)))
i=0
for file in files:
    if categ_feat[file][0]=='F':      
        gender_features_x[i,0]=1
    else:
        gender_features_x[i,1]=1
    gender_features_y[i]=dic[file]
    i+=1
print(fisher(gender_features_x,gender_features_y))

num_bins=4
age_features_x=np.zeros((len(dic),num_bins))
age_features_y=np.zeros((len(dic)))
i=0
for file in files:
    age=categ_feat[file][1]
    if age:
        age_features_x[i,int((age-min_age)/(max_age-min_age+1)*num_bins)]=1
    age_features_y[i]=dic[file]
    i+=1
print(fisher(age_features_x,age_features_y))

loc_features_x=np.zeros((len(dic),len(location)))
loc_features_y=np.zeros((len(dic)))
i=0
for file in files:
    loc=categ_feat[file][2]
    if loc:
        loc_features_x[i,location[loc]]=1
    loc_features_y[i]=dic[file]
    i+=1
print(fisher(loc_features_x,loc_features_y))

'''
gabor_features_x=np.zeros((len(dic),48672))
gabor_features_y=np.zeros((len(dic)))
i=0
for file in files:
    gabor_features_x[i]=np.load(gabor_feature_dir+'/'+file+'.npy')
    gabor_features_y[i]=dic[file]
    i+=1
print(fisher(gabor_features_x,gabor_features_y))

hog_features_x=np.zeros((len(dic),11664))
hog_features_y=np.zeros((len(dic)))
i=0
for file in files:
    hog_features_x[i]=np.load(hog_feature_dir+'/'+file+'.npy')
    hog_features_y[i]=dic[file]
    i+=1
print(fisher(hog_features_x,hog_features_y))  

prewitt_features_x=np.zeros((len(dic),48672))
prewitt_features_y=np.zeros((len(dic)))
i=0
for file in files:
    prewitt_features_x[i]=np.load(prewitt_feature_dir+'/'+file+'.npy')
    prewitt_features_y[i]=dic[file]
    i+=1
print(fisher(prewitt_features_x,prewitt_features_y)) 

gray_pexel_features_x=np.zeros((len(dic),24336))
gray_pexel_features_y=np.zeros((len(dic)))
i=0
for file in files:
    gray_pexel_features_x[i]=np.load(gray_pexel_feature_dir+'/'+file+'.npy')
    gray_pexel_features_y[i]=dic[file]
    i+=1
print(fisher(gray_pexel_features_x,gray_pexel_features_y)) 
'''


