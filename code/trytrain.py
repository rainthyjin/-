from __future__ import absolute_import, division, print_function, unicode_literals 

# 导入TensorFlow和tf.keras
import tensorflow as tf               
from tensorflow import keras

# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt

#导入自定义的载入数据模块（官网给出的直接下载数据集操作不可用，所以手动下载数据集并读取数据）
import Load_Data

#print(tf.__version__)  
#显示tensorflow版本信息

#载入训练数据及其标签，测试数据及其标签
#具体载入方法见load_data函数内部
(train_images, train_labels), (test_images, test_labels) = Load_Data.load_data()

#由于标签中的类别名称是0-9，这里将类别名写出来
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#查看数据信息
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
'''
#检查训练图像
#创建一个图形，显示第一个训练图像
plt.figure()  
plt.imshow(train_images[0])
#在图像旁边添加颜色条，不生成网格，显示图片
plt.colorbar()
plt.grid(False)
plt.show()
'''
#将图像的像素值缩放到0-1之间
train_images = train_images / 255.0
test_images = test_images / 255.0

'''
#创建10*10英尺画布，分成25个axes区域并逐个操作
plt.figure(figsize=(10,10))
#X，Y轴不设置刻度以及文组
#不显示网格
#显示第i张图，cm.binary代表用二值颜色图显示图片，即黑白图
#X轴标签为类别名字
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

#三层网络
#第一层将输入的28*28尺寸的图片展开成一维
#第二三层为全连接层（稠密层）
#第一个参数为神经元数量，第二个参数为激活函数类型

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation=tf.nn.softplus),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


#编译模型并在该过程优化模型（优化器，损失函数，评价方式）  
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])   
#训练模型（训练数据，训练数据标签，epochs即全部数据被训练多少次）
model.fit(train_images, train_labels, epochs=5)    

#返回测试集的损失以及精确度
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
#使用模型预测测试图片，返回值为每张图片的所有预测值列表
print('Test loss:', test_loss)
predictions = model.predict(test_images)
#查看对第一张图属于哪类的预测
#print(predictions[0])
#返回最大的的索引值
#print(np.argmax(predictions[0]))
#返回第一张图的标签
#print(test_labels[0])

#自定义函数，输入图片序号，需要预测的图片列表，真实标签列表，图片列表
#输出为在画布上显示这些变量
def plot_image(i, predictions_array, true_label, img):
#
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#不显示网格，XY轴没有刻度和文字
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
#显示图片二值图
  plt.imshow(img, cmap=plt.cm.binary)
#取预测的标签值
  predicted_label = np.argmax(predictions_array)
#预测正确为蓝色，错误为红色
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
#X轴标签信息（类别名称，最大概率%，真实类别），颜色根据color确定  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

#显示预测值序列的函数
#输入（图片序号，预测值列表，真实标签列表）
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  #根据十个分类的预测值，绘制条形图（条形图x坐标，高度，颜色color）
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  #Y轴范围0~1
  plt.ylim([0, 1])
  #取预测标签值
  predicted_label = np.argmax(predictions_array)
  
  #对应预测值和真实值的条形图分别设置成红色和蓝色
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#显示预测的图片和预测值（可视化 ）
i = 20
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

i = 30
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# 绘制前X个测试图像，预测标签和真实标签
# 以蓝色显示正确的预测，红色显示不正确的预测
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# 从测试数据集中获取图像
img = test_images[0]

#print(img.shape)

# 将图像添加到批次中，即使它是唯一的成员。
img = (np.expand_dims(img,0))

#print(img.shape)

predictions_single = model.predict(img)

#print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

prediction_result = np.argmax(predictions_single[0])
#print(prediction_result)
