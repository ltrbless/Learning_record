[TOC]
### `reduce_sum() `   

$$
x = [ [1, 1, 1] \\ \ \ \ \ \ \ \ \ \ \ [1,1,1] ] \\
tf.reduce.sum(x) = 6\\
tf.reduce.sum(x, 0) = [2, 2, 2]\\
tf.reduce.sum(x, 1) = [3, 3]\\
tf.reduce.sum(x, [0, 1]) = 6\\
$$

​	前缀reduce就是“对矩阵降维”的含义，下划线后面的部分就是降维的方式，在reduce_sum()中就是按照求和的方式对矩阵降维。那么其他reduce前缀的函数也举一反三了，比如reduce_mean()就是按照某个维度求平均值，等等。

### `tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)`

一文搞懂交叉熵在机器学习中的使用，透彻理解交叉熵背后的直觉 : https://blog.csdn.net/tsyccnh/article/details/79163834

![picture_01](E:\ltr_do\deep_learn\picture_01.png)

​	第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes
	第二个参数labels：实际的标签，大小同上

设 pre_y 为预测值，tag_y 为真值 :

>1. 把 pre_y 通过 `softmax` 映射为 [0, 1] 之间的概率值。
>2. 把使用 `softmax` 处理过的 pre_y 和标签 tag_y 做交叉熵 `cross_entropy`

~~~python
import tensorflow as tf
#our NN's output
logits=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])

#step1:do softmax
y=tf.nn.softmax(logits)

#true label
y_=tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])

#step2:do cross_entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#do cross_entropy just one step
cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, y_))#dont forget tf.reduce_sum()!!

with tf.Session() as sess
    softmax=sess.run(y)
    c_e = sess.run(cross_entropy)
    c_e2 = sess.run(cross_entropy2)
    print("step1:softmax result=")
    print(softmax)
    print("step2:cross_entropy result=")
    print(c_e)
    print("Function(softmax_cross_entropy_with_logits) result="
    print(c_e2)

          
# output:

step1:softmax result=
[[ 0.09003057  0.24472848  0.66524094]
 [ 0.09003057  0.24472848  0.66524094]
 [ 0.09003057  0.24472848  0.66524094]]
step2:cross_entropy result=1.22282
Function(softmax_cross_entropy_with_logits) result=1.2228
  注：        
e^1/(e^1+e^2+e^3) == 0.09003057
~~~

### `placeholder`

​	placeholder是TensorFlow的占位符节点，由placeholder方法创建，其也是一种常量，但是由用户在调用run方法是传递的，也可以将placeholder理解为一种形参。即其不像constant那样直接可以使用，需要用户传递常数值。

创建方式：

>X = tf.placeholder(dtype=tf.float32, shape=[144, 10], name='X')

参数说明:
>dtype：数据类型，必填，默认为value的数据类型，传入参数为tensorflow下的枚举值（float32，float64.......）
>shape：数据形状，选填，不填则随传入数据的形状自行变动，可以在多次调用中传入不同形状的数据
>name：常量名，选填，默认值不重复，根据创建顺序为（Placeholder，Placeholder_1，Placeholder_2.......

示例代码:

~~~python
import tensorflow as tf
import numpy.random as random
#占位符shape不设时会按传入参数自行匹配
node1 = tf.placeholder(tf.float32) # , shape=[4, 5])
node2 = tf.placeholder(tf.float32) # , shape=[4, 5])
op = tf.multiply(node1, node2)
session = tf.Session()
onst1 = tf.constant(random.rand(4, 5))
const2 = tf.constant(random.rand(4, 5))
#可以传入初始化后的常量
print(session.rn(op, {node1: session.run(const1), node2: session.run(const2)}))
#也可以直接传入张量，其实同初始化后的常量一致
print(session.run(op, {node1: random.rand(2, 3), node2: random.rand(2, 3)}))
~~~

### `argmax`

~~~python
def argmax(input,
           axis=None,
           name=None,
           dimension=None,
           output_type=dtypes.int64)
numpy.argmax(a, axis=None, out=None) 
返回沿轴axis最大值的索引。

Parameters: 
input: array_like，数组
axis : int, 可选，默认情况下，索引的是平铺的数组，否则沿指定的轴。 
out : array, 可选 如果提供，结果以合适的形状和类型被插入到此数组中。 
Returns: 
index_array : ndarray of ints 
索引数组。它具有与a.shape相同的形状，其中axis被移除。     
tf.argmax() 与 numpy.argmax() 方法的用法是一致的
axis = 0 的时候返回每一列最大值的位置索引
axis = 1 的时候返回每一行最大值的位置索引
axis = 2、3、4 ...，即为多维张量时，同理推断


array([[0, 1, 2],
       [3, 4, 5]])
np.argmax(a)
5
np.argmax(a, axis=0)  #0代表列
array([1, 1, 1])
np.argmax(a, axis=1)  #1代表行
array([2, 2])
~~~

### 	`cast`

​	改变数据的类型：

~~~python
x = tf.constant([1.8, 2.2], dtype=tf.float32) 
y = tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32 
~~~

### `tf.nn.conv2d`

方法定义

* tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

参数：

* input : 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
* filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
* strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
* padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
* use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true

### `tf.nn.l2_loss()`

* tf.nn.l2_loss(t, name="")   --> $ \frac{1}{2}\sum t^2 $

~~~python
import tensorflow as tf
x = tf.constant([[1, 2], [3, 4]],dtype=tf.float32)
with tf.Session() as sess:
    print(sess.run(tf.nn.l2_loss(x))) # (1 + 4 + 9 + 16) / 2 = 30 / 2  = 15
~~~

正则化（Regularization）
机器学习中几乎都可以看到损失函数后面会添加一个额外项，常用的额外项一般有两种，一般英文称作 ℓ1-norm 和ℓ2-norm，中文称作 L1正则化 和 L2正则化，或者 L1范数 和 L2范数。
L1正则化和L2正则化可以看做是损失函数的惩罚项。所谓『惩罚』是指对损失函数中的某些参数做一些限制。

L1正则化和L2正则化的说明如下：

* L1正则化是指权值向量$w$中各个元素的绝对值之和，通常表示为$||w||_1$
* L2正则化是指权值向量$w$中各个元素的平方和然后再求平方根（可以看到Ridge回归的L2正则化项有平方符号），通常表示为$||w||_2$

一般都会在正则化项之前添加一个系数，Python中用 $\alpha$ 表示，一些文章也用 $\lambda$ 表示。这个系数需要用户指定。
下面是L1正则化和L2正则化的作用 :

* L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择
* L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合

