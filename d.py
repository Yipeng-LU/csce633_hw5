from keras.applications import VGG16
from keras import layers
from keras import Input
from keras import models
from keras.models import Sequential, Model
from keras import optimizers
def create_model_cnn():
    vgg= VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
    model= Sequential()
    for layer in vgg.layers: # go through until last layer
        layer.trainable=False
        model.add(layer)
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.00001),
    metrics=['acc'])
    return model
import os
import cv2
import numpy as np
def center_standardization(img):
    img=img.astype('float')
    mean=np.mean(img)
    std=np.std(img)
    for x in range(len(img)):
        for y in range(len(img[0])):
            for z in range(3):
                img[x,y,z]=(img[x,y,z]-mean)/max(std,1/img.size)
    return img
dir='/content/Homework5/test'
files=os.listdir(dir)
x=[]
a=224
for file in files:
    img=cv2.imread(dir+'/'+file)
    height,width,dim=img.shape
    diff=abs(width-height)
    if width>height:
        processed_img=cv2.resize(img[:,diff//2:-diff//2],(a,a))
    elif height>width:
        processed_img=cv2.resize(img[diff//2:-diff//2],(a,a))
    else:
        processed_img=cv2.resize(img,(a,a))  
    x.append(center_standardization(processed_img))
x=np.array(x)
model=create_model_cnn()
model.load_weights('/content/drive/My Drive/hw5_model1.h5')
y_vgg16=model.predict(x)
vgg16_predictions={}
for i in range(len(files)):
    file=files[i]
    prediction=y_vgg16[i]
    if prediction>0.5:
        vgg16_predictions[file]=1
    else:
        vgg16_predictions[file]=0
print(vgg16_predictions)
from sklearn.svm import SVC
import csv
dic={}
categ_feat={}
location={}
loc_index=0
with open('/content/drive/MyDrive/Homework5/train.csv') as csvfile:
    datareader = csv.reader(csvfile)
    for samp in datareader:
        if samp[0]=='filename':
            continue
        dic[samp[0]]=int(samp[-1])
        loc=samp[3]
        if loc and loc not in location:
            location[loc]=loc_index
            loc_index+=1
        categ_feat[samp[0]]=loc
files=list(dic.keys())
loc_features_x=np.zeros((len(dic),len(location)))
y=np.zeros((len(dic)))
i=0
for file in files:
    loc=categ_feat[file]
    if loc:
        loc_features_x[i,location[loc]]=1
    y[i]=dic[file]
    i+=1
clf=SVC(gamma='scale',C=2)
clf.fit(loc_features_x,y)

categ_feat={}
with open('/content/drive/MyDrive/Homework5/test.csv') as csvfile:
    datareader = csv.reader(csvfile)
    for samp in datareader:
        if samp[0]=='filename':
            continue
        loc=samp[3]
        categ_feat[samp[0]]=loc
files=list(categ_feat.keys())
loc_features_x=np.zeros((len(files),len(location)))
i=0
not_exist_loc=set()
for file in files:
    loc=categ_feat[file]
    if loc and loc in location:
        loc_features_x[i,location[loc]]=1
    elif loc and (loc not in location):
        not_exist_loc.add(file)
    i+=1
y_loc=clf.predict(loc_features_x)
loc_predictions={}
for i in range(len(files)):
    file=files[i]
    prediction=y_loc[i]
    if file not in not_exist_loc:
        loc_predictions[file]=prediction
import csv
with open('hw5_predictions.csv', 'w', newline='') as csvfile:
    prediction_writer = csv.writer(csvfile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for file in files:
        pred1=vgg16_predictions[file]
        if file in loc_predictions:
            pred2=int(loc_predictions[file])
        else:
            pred2=pred1
        prediction_writer.writerow([file,pred1,pred2])
