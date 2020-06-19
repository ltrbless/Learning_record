[TOC]



### softmax

​	$ 对于数组 [a_1, a_2, ... , a_n] ， S_i = \frac{e^{a_i}}{\sum_{j=1}^{n}e^{a_j}}$
	他把一些输入映射为0-1之间的实数，并且归一化保证和为1，因此多分类的概率之和也刚好为1。 

​	大小相当于概率了

### 梯度下降法
​	既然在变量空间的某一点处，函数沿梯度方向具有最大的变化率，那么在优化目标函数的时候，自然是沿着负梯度方向去减小函数值，以此达到我们的优化目标。 	同时梯度和偏导数都是向量，那么参考向量运算法则，我们在每个变量轴上减小对应变量值即可，梯度下降法可以描述如下：
								$$ gradf(x_0, x_1, ... , x_n) = (\frac{\partial f}{\partial x_0}, ... , \frac{\partial f}{\partial x_n}) $$
	同时梯度和偏导数都是向量，那么参考向量运算法则，我们在每个变量轴上减小对应变量值即可，梯度下降法可以描述如下：
									$ x_0 := x_0 - \alpha\frac{\partial f}{\partial x_0} \\ ... \ ... \\ x_n := x_n - \alpha\frac{\partial f}{\partial x_n} $

​	该点方向导数最大值和该点梯度向量的模相等。
通俗理解：
	把这一点带入到梯度函数中,结果为正,那我们就把这一点的值变小一些,同时就是让梯度变小些;当这一点带入梯度函数中的结果为负的时候,就给这一点的值增大一些. 整体来说就是让梯度趋近于0，从而逼近极值点。

### 反向传播

知乎一遍很好的文章：	https://zhuanlan.zhihu.com/p/21407711

### 激活函数

​	作用：引入非线性因素，解决线性不可分之类的问题，使得具有更强大的拟合能力

> * Sigmoid
>   > * 公式：$ y = \frac{1}{1 + e^{-x}} $    
>   > * 导数: $y^{'} = (1 - y) \times y$
>   > * 缺陷：
>   > > 1.   **sigmoid 极容易导致梯度消失问题。** 这一问题在[RNN 的梯度消失问题](https://zhuanlan.zhihu.com/p/44163528) 已经做了详细的讲解， 值得一提的是， sigmoid 神经元在值为 0 或 1 的时候接近饱和，这些区域，梯度几乎为 0。 
>   > > 2.   **计算费时。** 在神经网络训练中，常常要计算sigmid的值， 幂计算会导致耗时增加。
>   > > 3.   **sigmoid 函数不是关于原点中心对称的（zero-centered)。**
>
> * Relu
>   >- 公式：$ y = max(0, x) $    
>   >- 导数: $y^{'} =  \begin{cases} 0& x<0\\ 1& x \ge 0  \end{cases}$
>   >- 优点：
>   >
>   >> 1. Relu一定程度上缓解了梯度问题（正区间）
>   >> 2. 计算速度非常快
>   >
>   >* 缺陷：
>   >
>   >>1. Relu的输出不是zero-centered
>   >>2. Dead ReLU Problem。这表示某些神经元可能永远不会被激活， 导致其相应的参数永远不能被更新。其本质是由于Relu在的小于0时其梯度为0所导致的。
>
> * Tanh函数 -- 双曲正切函数

### tensorflow安装

* 遇到np问题，说明numpy版本不对，改成1.16.0
* 如果import tensorflow 没反应，在pycharm中tensorflow载入报错Process finished with exit code -1073741819 (0xC0000005)  ，那么更新下 h5py  这个包就好了，如果没用可以参考<https://blog.csdn.net/qiao1025566574/article/details/81037908> 
* 如果找不到dll，可能是环境问题，或者没安装 VS2015 ，如果没有这个模块常常 版本不对

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

### 正则化（Regularization）

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

