import pandas as pd 
import cv2
import numpy as np
df = pd.read_csv('train.csv')
train_y = np.array(df['label'].values,np.int8)
df2 = pd.read_csv('train.csv',index_col=0)
images = np.array(df2.values,np.float32)
images = images.reshape(42000,28,28)
images =np.array(images,np.float32)
train_x = images
cv2.imshow('main',images[0])
cv2.waitKey(0)