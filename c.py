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
dir='/content/Homework5/train'
files=os.listdir(dir)
x=[]
y=[]
a=224
for file in files:
    y_f=dic[file]
    y.append(y_f)
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
y=np.array(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)
import imutils
import random
def data_augmentation(batch_x):
    batch_size=len(batch_x)
    out=np.zeros((batch_size,224,224,3))
    for i in range(batch_size):
        a=random.randint(-15,15)
        out[i]=imutils.rotate(batch_x[i], angle=a)
    return out
def data_generator(x,y,batch_size,augmentation=True):
    i=0
    while 1:
        if i+batch_size<=len(x):
              batch_x=x[i:i+batch_size]
              batch_y=y[i:i+batch_size]
              i+=batch_size
        else:
              batch_x=np.concatenate((x[i:],x[:i+batch_size-len(x)]),axis=0)
              batch_y=np.concatenate((y[i:],y[:i+batch_size-len(x)]),axis=0)
              i=i+batch_size-len(x)
        if augmentation:
            batch_x=data_augmentation(batch_x)
        yield (batch_x,batch_y)
#create and compile cnn model
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
    optimizer=optimizers.Adam(learning_rate=0.00001),metrics=['acc'])
    return model
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
batch_size=32
numEpoches=200
callbacks_list = [
keras.callbacks.ModelCheckpoint(
filepath='/content/drive/My Drive/hw5_model1.h5',
monitor='val_loss',
save_best_only=True,
)
]
model=create_model_cnn()
train_steps=len(x_train)//batch_size
val_steps=len(x_test)//batch_size
history = model.fit_generator(
data_generator(x_train,y_train,batch_size),
steps_per_epoch=train_steps,
epochs=numEpoches,
validation_data=data_generator(x_test,y_test,batch_size,False),
validation_steps=val_steps,
callbacks=callbacks_list,
class_weight={0:146/250,1:104/250})
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model=None
