#------------------------------ Import Module --------------------------------#
import numpy as np
import cv2
import os
import tensorflow as tf
import math
import matplotlib.pyplot as plt 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tf.device('/device:GPU:2')
#--------------------------------- Parameter ---------------------------------#
image_heigh=60            # 統一圖片高度
image_width=60            # 統一圖片寬度
data_number=10000          # 每種類動物要取多少筆data來train
layer1_node=32
layer2_node=64
layer3_node=128
layer4_node=512
parameter_dim=512
batch_size=5             # 多少筆data一起做訓練
epoch_num=20              # 執行多少次epoch
Loss_record=[]

#---------------------------------- Function ---------------------------------#
# 讀取圖片
def read_image(path,data_number):
    imgs = os.listdir(path)      # 獲得該路徑下所有的檔案名稱
    total_image=np.zeros([data_number,image_heigh,image_width,3])   
    # 依序將每張圖片儲存進矩陣total_image當中
    for num_image in range(0,data_number):
        filePath=path+'//'+imgs[num_image]    # 圖片路徑
        cv_img=cv2.imread(filePath)  # 取得圖片
        total_image[num_image,:,:,:] = cv2.resize(cv_img, (image_heigh, image_width), interpolation=cv2.INTER_CUBIC)  # resize並且存入total_image當中      
    return total_image

# 建立ResNet block層
def ResNet(input_node,kernel_height_width,is_activation,name="resnet"):
    # input_node 為輸入的節點
    # kernel_Size為filter的大小, 其中height與width都相同
    filters_number=input_node.shape.as_list()[3]    # 取得feature map數量
    with tf.variable_scope(name):
        # ResNet第一層
        output1=tf.layers.conv2d_transpose(            
                inputs=input_node,
                filters=filters_number,
                kernel_size=[kernel_height_width,kernel_height_width],
                strides=1,
                padding="same"
                )
        if is_activation:
            output1=tf.nn.relu(output1)
        # ResNet第二層
        output2=tf.layers.conv2d_transpose(            
                inputs=input_node,
                filters=filters_number,
                kernel_size=[kernel_height_width,kernel_height_width],
                strides=1,
                padding="same"
                )
        if is_activation:
            output2=tf.nn.relu(output2)
        # 輸出加總
        output=output1+output2
        return output

#-------------------------------- Input Data ---------------------------------#

# 傳入training data
#path=r'C:\Users\USER/Desktop\DL HW3\cartoon'
path=r'/home/alantao/deep learning/DL HW3/cartoon'
#path=r'/home/alantao/deep learning/DL HW3/animation'
training_data=read_image(path,data_number)

# 修改資料型態
training_data=training_data.reshape([-1,image_heigh*image_width*3])  # 把每個顏色的2為圖片(連同RGB)拉長  
training_data=training_data/255    # normalize

# 建立Session
sess=tf.InteractiveSession()    
    
# 輸入點設置 data 與 label
images_placeholder=tf.placeholder(tf.float32,shape=(None,image_heigh*image_width*3))
label_placeholder=tf.placeholder(tf.float32,shape=(None,image_heigh*image_width*3))
x_image=tf.reshape(images_placeholder,[-1,image_heigh,image_width,3])  # 轉回圖片的size
global_step = tf.Variable(0, trainable=False)   # 初始時迭代0次

## Encoder
# 第1層 Convolution
hidden1=tf.layers.conv2d(
    inputs=x_image,
    filters=layer1_node,
    kernel_size=[3,3],
    padding="same",
    activation=tf.nn.relu
)
hidden1 = tf.nn.dropout(hidden1, keep_prob=0.9) 
hidden1=tf.layers.batch_normalization(hidden1)  # batch normalize

# 第2層 Convolution
hidden2=tf.layers.conv2d(
    inputs=hidden1,
    filters=layer2_node,
    kernel_size=[3,3],
    padding="same",
    activation=tf.nn.relu
)
hidden2 = tf.nn.dropout(hidden2, keep_prob=0.9) 
hidden2=tf.layers.batch_normalization(hidden2)  # batch normalize

# 第3層 Convolution
hidden3=tf.layers.conv2d(
    inputs=hidden2,
    filters=layer3_node,
    kernel_size=[3,3],
    padding="same",
    activation=tf.nn.relu
)
hidden3=tf.layers.batch_normalization(hidden3)  # batch normalize

# 通過ResNet
hidden3_1=ResNet(hidden3,3,True,name="resnet_in1")
hidden3_2=ResNet(hidden3_1,3,True,name="resnet_in2")

# 將第3層的輸出拉平(目前有layer3_node張feature map,每張大小為image_heigh*image_width)
hidden_shape=hidden3_2.get_shape().as_list()    # 取得hidden3的矩陣shape(batch_size,image_heigh,image_width,channel=layer3_node)
hidden3_flat=tf.reshape(hidden3,[-1,hidden_shape[1]*hidden_shape[2]*hidden_shape[3]]) # 拉平
#hidden3_flat=tf.reshape(hidden3,[-1,int(image_heigh*image_width*layer3_node)]) # 拉平

# 第4層 Fully connected
hidden4=tf.layers.dense(
    hidden3_flat,
    units=layer4_node,
    #activation=tf.nn.relu
)
hidden4=tf.layers.batch_normalization(hidden4)  # batch normalize
hidden4 = tf.nn.dropout(hidden4, keep_prob=0.9) 

# 第5層 Fully connected
Gaussian=tf.layers.dense(
    hidden4,
    units=parameter_dim*2,
    #activation=tf.nn.relu
)
Gaussian=tf.layers.batch_normalization(Gaussian)  # batch normalize

# mean parameter 
mean=tf.layers.dense(
    Gaussian,
    units=parameter_dim
)

# std parameter
std=tf.layers.dense(
    Gaussian,
    units=parameter_dim
)
std = 1e-6 + tf.nn.softplus(std) 
# softplus是光滑版的RELU函数，函数是log(exp(x)+1)

# sampling by re-parameterization 
z = mean + std * tf.random_normal([tf.shape(images_placeholder)[0],parameter_dim], 0, 1, dtype=tf.float32)
# tf.shape(images_placeholder)[0]即為batch size

## Decoder
# 第1層 Fully connected
output1=tf.layers.dense(
    z,
    units=layer4_node
)
output1=tf.layers.batch_normalization(output1)  # batch normalize
output1 = tf.nn.dropout(output1, keep_prob=0.9)  # 有0.1的機率會丟掉輸入

# 第2層 Fully connected
output2=tf.layers.dense(
    output1,
    units=hidden_shape[1]*hidden_shape[2]*hidden_shape[3],
    #activation=tf.nn.relu
)
output2=tf.layers.batch_normalization(output2)  # batch normalize
output2 = tf.nn.dropout(output2, keep_prob=0.9)  # 有0.1的機率會丟掉輸入

# 還原圖片形狀
output2=tf.reshape(output2,[-1,hidden_shape[1],hidden_shape[2],hidden_shape[3]]) # 轉成圖片形狀

# 通過ResNet
output2_1=ResNet(output2,3,True,name="resnet_out1")
output2_2=ResNet(output2_1,3,True,name="resnet_out2")

# 第3層 Transpose Convolution
output3=tf.layers.conv2d_transpose(
    inputs=output2_2,
    filters=layer3_node,
    kernel_size=[3,3],
    strides=1,
    padding="same",
    #activation=tf.nn.relu
)
#output3 = tf.nn.tanh(output3)
#output3 = tf.nn.dropout(output3, rate=0.1)  # 有0.1的機率會丟掉輸入
output3=tf.layers.batch_normalization(output3)  # batch normalize

# 第4層 Transpose Convolution
output4=tf.layers.conv2d_transpose(
    inputs=output3,
    filters=layer2_node,
    kernel_size=[3,3],
    strides=1,
    padding="same",
    #activation=tf.nn.relu
)
#output4=conv_trans(output3,W_out4,output4_shape)+b_out4  
output4=tf.layers.batch_normalization(output4)  # batch normalize

# 第5層 Transpose Convolution
output5=tf.layers.conv2d_transpose(
    inputs=output4,
    filters=layer1_node,
    kernel_size=[3,3],
    strides=1,
    padding="same",
    #activation=tf.nn.relu
)
output5=tf.layers.batch_normalization(output5)  # batch normalize

# 第6層 Transpose Convolution
output6=tf.layers.conv2d_transpose(
    inputs=output5,
    filters=6,
    kernel_size=[3,3],
    strides=1,
    padding="same",
    #activation=tf.nn.relu
)
output6=tf.layers.batch_normalization(output6)  # batch normalize

# 第7層 Convolution
output7=tf.layers.conv2d(
    inputs=output6,
    filters=3,
    kernel_size=[3,3],
    padding="same",
)
output = tf.sigmoid(output7)


# 輸出層
output = tf.clip_by_value(output, 1e-8, 1 - 1e-8)

## loss
#marginal_likelihood = tf.reduce_sum(label_placeholder * tf.log(output) + (1 - label_placeholder) * tf.log(1 - output), 1)
marginal_likelihood = tf.reduce_sum(x_image * tf.log(output+1e-20) + (1 - x_image) * tf.log(1 - output+1e-20), [1,2,3])
marginal_likelihood = tf.reduce_mean(marginal_likelihood)

KL_divergence = 0.5 * tf.reduce_sum(tf.square(mean) + tf.square(std) - tf.log(1e-8 + tf.square(std)) - 1, 1)
KL_divergence = tf.reduce_mean(KL_divergence)

ELBO = marginal_likelihood - KL_divergence
loss = -ELBO


# 評估模型
# learning rate decay
# [decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)]
initial_learning_rate = 1e-4                    # 初始的learning rate


#global_step = tf.Variable(0, trainable=False)   # 初始時迭代0次
learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step=global_step,decay_steps=5,decay_rate=0.5) # 每500epoch，學習綠會衰減0.9


training_method=tf.train.AdamOptimizer(learning_rate).minimize(loss)
#training_method=tf.train.AdamOptimizer(1e-8).minimize(loss) # 用Adam做參數修正,學習率10^-5,最小化Loss function=loss

# 開始訓練
sess.run(tf.global_variables_initializer())      # 激活所有變數
for epoch_times in range(0,epoch_num):       # 要執行多次epoch
    print('epoch times=',epoch_times)
    for batch_times in range(0,int(data_number/batch_size)):  # 全部的資料可以分成多少個batch
        get_x=training_data[batch_times*batch_size:(batch_times+1)*batch_size,:]
        # 修正learning rate
        sess.run(learning_rate,feed_dict={global_step: epoch_times})  # 修正learning rate
        # 做training  
        training_method.run(feed_dict={images_placeholder:get_x,label_placeholder:get_x})
        # 觀察Loss(每個batch)
        Loss=loss.eval(feed_dict={images_placeholder:get_x,label_placeholder:get_x})
        Loss=Loss/batch_size
        Loss_record.append(Loss)
        #print('Loss=',Loss)
        
    # 觀察Loss(每個epoch)
    #get_x=training_data
    Loss=loss.eval(feed_dict={images_placeholder:get_x,label_placeholder:get_x})
    Loss=Loss/batch_size
    #Loss_record.append(Loss)
    print('Loss=',Loss)
    
    print(learning_rate.eval(feed_dict={global_step: epoch_times}))
    
    # Shuffle
    np.random.shuffle(training_data)     # shuffle

# 在plot之前要加上這行    
%matplotlib inline       
    
# 印出結果 (Learning curve)
plt.figure(1)
plt.plot(Loss_record)
plt.xlabel('Number of batch')
plt.ylabel('loss')
plt.show()       
    
print('///////////////////////////////////////////')
print('Image test')
# 比較經過VAE之前與之後圖片的差異
for image_num in range(0,15):  # 取30個圖片做比較
    # 原圖片
    temp=training_data[image_num,:]
    temp=temp.reshape([1,image_heigh*image_width*3]) 
    test1=temp.reshape([image_heigh,image_width,3])
    b,g,r = cv2.split(test1)
    test1 = cv2.merge([r,g,b])     # 轉為RGB的順序,因為cv2.imread()导入图片时是BGR通道顺序,而Matplotlib則是RGB順序
    plt.imshow(test1) 
    plt.show()
    # 經VAE後的圖片
    test2=output.eval(feed_dict={images_placeholder:temp,label_placeholder:temp})
    test2= test2.reshape([image_heigh,image_width,3])
    b,g,r = cv2.split(test2)
    test2 = cv2.merge([r,g,b])
    plt.imshow(test2) 
    plt.show()
    
print('///////////////////////////////////////////')
print('Image interpolate')
# 取得第1張圖的embedding vector
image1=training_data[1,:]
image1=image1.reshape([1,image_heigh*image_width*3])
embedding1=z.eval(feed_dict={images_placeholder:image1,label_placeholder:image1})
embedding1=embedding1.reshape([1,parameter_dim])
print(embedding1)
# 取得第2張圖的embedding vector
image2=training_data[2,:]
image2=image2.reshape([1,image_heigh*image_width*3])
embedding2=z.eval(feed_dict={images_placeholder:image2,label_placeholder:image2})
embedding2=embedding2.reshape([1,parameter_dim])
# 使用2個embedding vector的平均來通過decoder觀察output
embedding_test1=(embedding1+embedding2)/2
test3=output.eval(feed_dict={z:embedding_test1}) 
test3= test3.reshape([image_heigh,image_width,3])
b,g,r = cv2.split(test3)
test3 = cv2.merge([r,g,b])
plt.imshow(test3) 
plt.show()
# 使用偏向image1
print('close to image1 ')
embedding_test2=embedding1*0.75+embedding2*0.25
test4=output.eval(feed_dict={z:embedding_test2}) 
test4= test4.reshape([image_heigh,image_width,3])
b,g,r = cv2.split(test4)
test4 = cv2.merge([r,g,b])
plt.imshow(test4) 
plt.show()
# 使用偏向image2
print('close to image2 ')
embedding_test3=embedding1*0.25+embedding2*0.75
test5=output.eval(feed_dict={z:embedding_test3}) 
test5= test5.reshape([image_heigh,image_width,3])
b,g,r = cv2.split(test5)
test5 = cv2.merge([r,g,b])
plt.imshow(test5) 
plt.show()

print('///////////////////////////////////////////')
print('Random generate image')
for random_image_num in range(0,30):  # 產生15張隨機圖片
    randon_z=np.random.randn(1,parameter_dim)*0.8
    test6=output.eval(feed_dict={z:randon_z}) 
    test6= test6.reshape([image_heigh,image_width,3])
    b,g,r = cv2.split(test6)
    test6 = cv2.merge([r,g,b])
    plt.imshow(test6) 
    plt.show()
print(randon_z)
    

sess.close()