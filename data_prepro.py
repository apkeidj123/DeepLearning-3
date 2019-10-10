import numpy as np
import os
import cv2
import random
import time

"""
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
   

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
"""

def center_crop(img):
    short_edge = min(img.shape[:2])
    y_side = int((img.shape[0] - short_edge) / 2)
    x_side = int((img.shape[1] - short_edge) / 2)
    crop_img = img[y_side: y_side + short_edge, x_side: x_side + short_edge]
    return  crop_img

def rotate_image(img):
    rows, cols = img.shape[:2]
    theta = random.randint(1,360)
    M = cv2.getRotationMatrix2D((cols/2,rows/2), theta, 1)  
    rotate_img = cv2.warpAffine(img, M, (200, 200))
    return rotate_img

def pad_image(img): # size will change
    top = random.randint(1,50)
    bottom = random.randint(1,50)
    left = random.randint(1,50)
    right = random.randint(1,50)
    values = random.randint(1,50)
    pad_img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(values, values, values))
    return pad_img

def shift_image(img):
    rows,cols = img.shape[:2]
    horizontal = random.randint(-20,20)
    vertical = random.randint(-20,20)
    M = np.float32([[1,0,horizontal],[0,1,vertical]])
    shift_img = cv2.warpAffine(img,M,(cols,rows))
    return shift_img

def flip_image(img):
    index = random.randint(-1,1)
    flip_img = cv2.flip(img, index)
    return flip_img


#"""
height = 200
width = 200
count = 0
###-------------train data-----------------
t0 = time.time()
path =u"train"
file_list = os.listdir(r'./train')
#print(file_list[:10])
for class_name in file_list:
    name_list = os.listdir(r'./train'+ '/' + class_name)
    #print(name_list)
    for image_name in name_list:
        root, ext = os.path.splitext(image_name)
        #print(root,ext)
        if ext == u'.png' or u'jpeg' or u'jpg':
            count += 1
            abs_name = path + '/' + class_name + '/' + image_name
            image = cv2.imread(abs_name)            
            img = cv2.resize(image,(height,width))  
            ### ---------------Data Augmentation------------------
            """
            if(count % 3 == 0):  ### rotate image
                img = rotate_image(img)
            if(count % 5 == 0):  ### shitf image
                img = shift_image(img)
            if(count % 7 == 0):  ### flip image  
                img = flip_image(img)
            """    
            cv2.imwrite(abs_name,img)                     
t1 = time.time()
print('Handle Image time: ', t1 - t0)
print('Image_num: ', count) 
###--------------test data----------------
t2 = time.time()
count_2 = 0
path_2 =u"test"
file_list_2 = os.listdir(r'./test')
#print(file_list[:10])
for class_name_2 in file_list_2:
    name_list_2 = os.listdir(r'./test'+ '/' + class_name_2)
    #print(name_list_2)
    for image_name_2 in name_list_2:
        root_2, ext_2 = os.path.splitext(image_name_2)
        #print(root,ext)
        if ext_2 == u'.png' or u'jpeg' or u'jpg':
            count_2 += 1
            abs_name_2 = path_2 + '/' + class_name_2 + '/' + image_name_2
            image_2 = cv2.imread(abs_name_2)
            img_2 = cv2.resize(image_2,(height,width))
            cv2.imwrite(abs_name_2,img_2)
t3 = time.time()
print('Handle Image time: ', t3 - t2)
print('Image_num: ', count_2)

#"""


#im = Image.open("./train/accordion/image_0001.jpg")
#print(im.format, im.size, im.mode)
#img = np.array(im)
#img = img.astype("float") / 255.0

#dirfile = "./train/accordion/image_0001.jpg"
#image = cv2.imread(dirfile)
#image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#img = cv2.resize(image,(200,200))
#image = image.astype("float") / 255.0
#B, G, R = cv2.split(image)
#image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
#R_flat = R.flatten()
#G_flat = G.flatten()
#B_flat = B.flatten()

#image_flat = image.flatten()
#x_train = np.zeros((60*60*3))
#image_resize = image_flat.reshape(60,60,3)

#x_train[0:40000]= R_flat
#x_train[40000:80000]= G_flat
#x_train[80000:120000]= B_flat
           
#y_train = np.zeros((7411,101))

#desfile = "./train_prepro/accordion/image_0001.jpg"
#cv2.imwrite(dirfile,img)

#size = img.shape

#cv2.imshow('My Image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
