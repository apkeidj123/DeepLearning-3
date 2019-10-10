###L2 regularization
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from matplotlib.ticker import PercentFormatter
#"""
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
#"""
  
###----------Hyper parameter------------
batch_size = 2
LR = 1e-4              # learning rate
height = 200
width = 200
channel = 3
pixels = height * width * channel
n_classes = 101
training_epochs = 50
calc_iter = 500
alpha = 0.01

#"""
def random_batch(inputs, targets, n_examples, batch_size):          
    indices = np.random.choice(n_examples, n_examples, replace = False)
    
    for batch_i in range(n_examples // batch_size): # 7411 // 2
        start = batch_i * batch_size
        end = start + batch_size       
        batch_xs = inputs[indices[start:end]]
        batch_ys = targets[indices[start:end]]

        yield batch_xs, batch_ys

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    #y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    #result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)  # x = data of image
# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
# filter：convolution kernel is four dimension data：shape：[height,width,in_channels, out_channels]    
    
def max_pool_2x2(x):
    # Must have strides[0] = strides[3] = 1
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# tf.nn.max_pool( value, ksize,strides,padding,data_format=’NHWC’,name=None)     
# default shape :[batch, height, width, channels]
#"""

x_train = np.zeros((7411,pixels))
y_train = np.zeros((7411,n_classes))
x_test = np.zeros((1266,pixels))
y_test = np.zeros((1266,n_classes))

image_index = 0
class_index = 0
image_index_2 = 0
class_index_2 = 0
#""" Read Image
print('Load Image Start')
tStart0 = time.time()
###---------------Load train data-----------------  
path =u"train"
file_list = os.listdir(r'./train')
for class_name in file_list:
    name_list = os.listdir(r'./train'+ '/' + class_name)    
    
    for image_name in name_list:
        root, ext = os.path.splitext(image_name)
        
        y_train[image_index][class_index] = 1
        if ext == u'.png' or u'jpeg' or u'jpg':
            abs_name = path + '/' + class_name + '/' + image_name
            image = cv2.imread(abs_name)
           #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = image.astype("float") / 255.0
            #B, G, R = cv2.split(image)            
            
            x_train[image_index]= image.flatten()
            
            #x_train[image_index][0:height*width]= R.flatten()
            #x_train[image_index][height*width:height*width*2]= G.flatten()
            #x_train[image_index][height*width*2:height*width*3]= B.flatten()
         
        image_index += 1           
    class_index +=1
                    
###---------------Load test data-----------------  
path_2 =u"test"
file_list_2 = os.listdir(r'./test')
#print(file_list[:10])
for class_name_2 in file_list_2:
    name_list_2 = os.listdir(r'./test'+ '/' + class_name_2)
    
    for image_name_2 in name_list_2:
        root_2, ext_2 = os.path.splitext(image_name_2)
        #print(root,ext)
        y_test[image_index_2][class_index_2] = 1
        if ext_2 == u'.png' or u'jpeg' or u'jpg':
            abs_name_2 = path_2 + '/' + class_name_2 + '/' + image_name_2
            image_2 = cv2.imread(abs_name_2)
            #image_2 = cv2.cvtColor(image_2,cv2.COLOR_BGR2RGB)    
            image_2 = image_2.astype("float") / 255.0
              
            x_test[image_index_2]= image_2.flatten()
            
        image_index_2 += 1           
    class_index_2 +=1        
            
print('Load Image Fin')
tEnd0 = time.time()
print("Loading Time = ", tEnd0 - tStart0)
#""" 
   
###-----------Build Model-------------------
  
## with tf.name_scope('inputs'):
#keep_prob = tf.placeholder(tf.float32)  # kepp % not to drop
#xs = tf.placeholder(tf.float32, [None, 200*200*3]) # 200 x 200 x 3
xs = tf.placeholder(tf.float32, [None, pixels]) # 200 x 200 x 3
ys = tf.placeholder(tf.float32, [None, n_classes])
x_image = tf.reshape(xs, [-1, height, width, 3]) # [3]: channel 無顏色為1
#x_image = tf.reshape(xs, [-1, 200, 200, 3]) # [3]: channel 無顏色為1
# print(x_image.shape) # [n_samples, 200, 200 ,3]   

###tf.contrib.layers.l2_regularizer(scale,scope=None)
### conv1 layer ###
W_conv1 = weight_variable([5,5,3,32]) # patch 5x5, in size 1, out size 32
L2_conv1 = tf.contrib.layers.l2_regularizer(alpha)(W_conv1) 
b_conv1 = bias_variable([32])
conv1 = conv2d(x_image, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(conv1) # output size 200x200x32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 100x100x32

### conv2 layer ###
W_conv2 = weight_variable([5,5,32, 32]) # patch 5x5, in size 32, out size 64
L2_conv2 = tf.contrib.layers.l2_regularizer(alpha)(W_conv2) 
b_conv2 = bias_variable([32])
conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(conv2) # output size 100x100x64
h_pool2 = max_pool_2x2(h_conv2)                          # output size 50x50x64

### conv3 layer ###
W_conv3 = weight_variable([5,5,32, 64]) # patch 5x5, in size 32, out size 64
L2_conv3 = tf.contrib.layers.l2_regularizer(alpha)(W_conv3) 
b_conv3 = bias_variable([64])
conv3 = conv2d(h_pool2, W_conv3) + b_conv3
h_conv3 = tf.nn.relu(conv3) # output size 100x100x64
h_pool3 = max_pool_2x2(h_conv3)                          # output size 50x50x64


### func1 layer ###
W_fc1 = weight_variable([25*25*64, 256])
L2_fc1 = tf.contrib.layers.l2_regularizer(alpha)(W_fc1) 
#W_fc1 = weight_variable([50*50*32, 256])
b_fc1 = bias_variable([256])
# [n_samples, 50, 50, 64] ->> [n_samples, 50*50*64]
h_pool3_flat = tf.reshape(h_pool3,[-1,25*25*64]) 
#h_pool2_flat = tf.reshape(h_pool2,[-1,50*50*32]) 
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### func2 layer ###
W_fc2 = weight_variable([256, 512])
L2_fc2 = tf.contrib.layers.l2_regularizer(alpha)(W_fc2)
b_fc2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### Output layer ###
W_fc3 = weight_variable([512, 101])
L2_fc3 = tf.contrib.layers.l2_regularizer(alpha)(W_fc3)
b_fc3 = bias_variable([101])
#prediction = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
y_pred = tf.matmul(h_fc2, W_fc3) + b_fc3
prediction = tf.nn.softmax(y_pred)
#prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

L2_Reg = L2_conv1 + L2_conv2 + L2_conv3 + L2_fc1 + L2_fc2 + L2_fc3

### the error between prediction and real data
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-11,1.0)),reduction_indices=[1])) # loss
#cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels = ys, logits = y_pred)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_pred, labels = ys))
#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_pred, labels = ys)

total_loss = cross_entropy + L2_Reg
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) # loss
train_step = tf.train.AdamOptimizer(LR).minimize(total_loss)

correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###--------------Start Training--------------------
train_loss = np.zeros([training_epochs,1])
train_accuracy = np.zeros([training_epochs,1])
test_loss = np.zeros([training_epochs,1])
test_accuracy = np.zeros([training_epochs,1])



init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('Start Training')
    tStart3 = time.time()
    
    for epoch in range(training_epochs):    
        print('Epoch: ', epoch + 1)
        ###-----------Accuracy / Loss--------------
        accuracy_train = 0.0
        accuracy_test = 0.0
        loss_train = 0.0
        loss_test = 0.0 
        L2_loss1 = 0.0
        L2_loss2 = 0.0
        
        tStart = time.time()
        
        for batch_xs,batch_ys in random_batch(x_train, y_train, y_train.shape[0], batch_size): # 7411
            sess.run(train_step,feed_dict={xs: batch_xs, ys: batch_ys})
            
        #sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})       
        #print('Finish Training')
        tEnd = time.time()
        
        print("Training Time = ", tEnd - tStart)
        
        #print('Start Calculate Accuracy & Loss')
        tStart2 = time.time()
        
        for i in range(calc_iter):
            random_index_train = np.random.choice(y_train.shape[0], 2, replace = False)
            Loss, Acc, l2_loss1, l2_loss2 = sess.run([total_loss, accuracy, cross_entropy, L2_Reg],
                                 feed_dict={xs: x_train[random_index_train],
                                            ys:y_train[random_index_train]})
            L2_loss1 += l2_loss1
            L2_loss2 += l2_loss2
            accuracy_train += Acc
            loss_train += Loss
            
            random_index_test = np.random.choice(y_test.shape[0], 2, replace = False)
            Loss_2, Acc_2 = sess.run([total_loss, accuracy], 
                                     feed_dict={xs: x_test[random_index_test],
                                                ys:y_test[random_index_test]})
            accuracy_test += Acc_2
            loss_test += Loss_2
            
            
        #print('Finish Calculate Accuracy & Loss')
        tEnd2 = time.time()
        print("Calculate Time = ", tEnd2 - tStart2)
        
        print('Train Accuracy: ', accuracy_train/calc_iter)
        print('Train Loss: ', loss_train/calc_iter)
        print('L2_loss1: ', L2_loss1/calc_iter)
        print('L2_loss2: ', L2_loss2/calc_iter)
        print('Test Accuracy: ', accuracy_test/calc_iter)
        print('Test Loss: ', loss_test/calc_iter)
        
        
        train_loss[epoch]= loss_train/calc_iter    
        train_accuracy[epoch] = accuracy_train/calc_iter
        test_loss[epoch] = loss_test/calc_iter
        test_accuracy[epoch] = accuracy_test/calc_iter
     
    tEnd3 = time.time()
    print("Total Training Time = ", tEnd3 - tStart3)
    #print('Loss: ', sess.run(cross_entropy,feed_dict={xs: x_train[:100],ys:y_train[:100]}))  
    #print('Accuracy: ', compute_accuracy(x_train[:100], y_train[:100]))
    
    img = cv2.imread('neko.jpg')
    img = img / 255
    img = np.reshape(img,(1,200*200*3))
    
    Cv1, Cv2 = sess.run([conv1, conv2], feed_dict={xs: img})
    Cv1 = Cv1 * 255
    Cv2 = Cv2 * 255
    #print(np.shape(Cv1))
    for i in range(32):
        cv2.imwrite('output/L2/conv1_' + str(i) + '.jpg', Cv1[0, :, :, i])
    for i in range(32):
        cv2.imwrite('output/L2/conv2_' + str(i) + '.jpg', Cv2[0, :, :, i])
        
    img2 = cv2.imread('tori.jpg')
    img2 = img2 / 255
    img2 = np.reshape(img2,(1,200*200*3))
    
    Cv3, Cv4 = sess.run([conv1, conv2], feed_dict={xs: img2})
    #print(sess.run(prediction, feed_dict={xs: img}))
    
    Cv3 = Cv3 * 255
    Cv4 = Cv4 * 255
    #print(np.shape(Cv1))
    for i in range(32):
        cv2.imwrite('output/L2/conv1v2_' + str(i) + '.jpg', Cv3[0, :, :, i])
    for i in range(32):
        cv2.imwrite('output/L2/conv2v2_' + str(i) + '.jpg', Cv4[0, :, :, i])   
        
    Wt = sess.run(W_conv1)
    W_conv1_flat = Wt.flatten()
    hist1 = W_conv1_flat
     
    Wt = sess.run(W_conv2)
    W_conv2_flat = Wt.flatten()
    hist2 = W_conv2_flat
    
    Wt = sess.run(W_fc1)
    W_fc1_flat = Wt.flatten()
    hist3 = W_fc1_flat
    
    Wt = sess.run(W_fc3)
    W_fc3_flat = Wt.flatten()
    hist4 = W_fc3_flat
    
    #plt.figure()
    plt.hist(hist1, bins = 200, weights = np.ones(len(hist1)) / len(hist1))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Hsitogram of conv1')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.savefig('output/L2/conv1.jpg')
    #plt.show()
    plt.clf()
    
    #plt.figure()
    plt.hist(hist2, bins = 200, weights = np.ones(len(hist2)) / len(hist2))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Hsitogram of conv2')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.savefig('output/L2/conv2.jpg')
    #plt.show()
    plt.clf()
       
    #plt.figure()
    plt.hist(hist3, bins = 200, weights = np.ones(len(hist3)) / len(hist3))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Hsitogram of dense1')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.savefig('output/L2/dense1.jpg')
    #plt.show()
    plt.clf()
    
    #plt.figure()
    plt.hist(hist4, bins = 200, weights = np.ones(len(hist4)) / len(hist4))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Hsitogram of output')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.savefig('output/L2/output.jpg')
    #plt.show()
    plt.clf()
    
    ###----------Save result-------- -
    np.save("result/s15x5l32L2/hist1",hist1)
    np.save("result/s15x5l32L2/hist2",hist2)
    np.save("result/s15x5l32L2/hist3",hist3)
    np.save("result/s15x5l32L2/hist4",hist4)
    
    ###---------Save model---------

    saver = tf.train.Saver()   
    save_path = saver.save(sess, "my_net/s15x5l32L2/Save_Net.ckpt")
    
 
    
#"""
###------3. Plot------
### Loss
plt.figure()
y_range = range(0,training_epochs)       

plt.plot(y_range, train_loss, color='blue', label="training loss")   
plt.plot(y_range, test_loss, color='orange', label="test loss")

plt.xlabel('epoch')
plt.ylabel('Cross entropy')
plt.legend(loc='best')       
plt.show()

### Accuracy
plt.figure()

plt.plot(y_range, train_accuracy, color='blue', label="training acc")   
plt.plot(y_range, test_accuracy, color='orange', label="test acc")

plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')       
plt.show()

#"""

###----------Save result-------- -
np.save("result/s15x5l32L2/Loss_train",train_loss)
np.save("result/s15x5l32L2/Acc_train",train_accuracy)
np.save("result/s15x5l32L2/Loss_test",test_loss)
np.save("result/s15x5l32L2/Acc_test",test_accuracy)





