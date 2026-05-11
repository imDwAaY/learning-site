---
tags:
  - CS188
  - "#Optimization"
  - "#Perceptron"
  - "#Multi-layer_Perceptron"
  - "#Hidden_Layer"
  - "#Neural_Networks"
  - "#Backpropagation"
source: "9.6( Optimization ) - 9.9( Neural Networks: Motivation )"
---
# Non-linear Separators
对于我们上一个Note提及到的`binary classification`，`Perceptron`可以学习一个线性分类边界。但是如果数据本身不是线性可分的，如下图所示，那么单纯的线性模型无论怎么调整权重，都无法找到一个正确的分界线。也即是说线性感知机表达能力不够。
![](notes/CS188/static/截屏2026-05-11%2016.52.21.png)
我们就想到了一种办法，人为添加新特征。比如把原来的一纬空间变成二维空间
![](notes/CS188/static/截屏2026-05-11%2016.54.18.png)
这样原本不可分的数据就可以被一条直线分开了。
- - -
# Multi-layer Perceptron
我们在上一讲中提及到的[普通`perceptron`](notes/CS188/Note/Note20#Binary%20Perceptron)的结构大概是从`input`到`weighted sum`到`activation`再到`output`，这是单层感知机。
![](notes/CS188/static/截屏2026-05-11%2016.58.36.png)
多层感知机的重要思想就是它让一个感知机的输出作为另一个感知机的输入，不再只做一次线性组合，而是做很多层，中间的层叫做`hidden layer`
![](notes/CS188/static/截屏2026-05-11%2017.00.37.png)
## Hidden Layer的作用
隐藏层的本质作用是**Learn intermediate representations**，就是学习中间特征。例如在图像分类中，网络可能会逐层学到：
```text
像素 → 边缘 → 局部纹理 → 局部形状 → 物体部件 → 整体类别
```
这就是`deep learning`的核心思想，越往深走越是抽象的特征。
同时我们要记住一个特点，多层感知机是`universal function approximator`，理论上可以逼近任何连续函数。注意这里说的是表达能力不是训练难度。也就是下面这段话 :
```text
Universal Function Approximation Theorem:
A neural network with enough hidden units can approximate any continuous function arbitrarily well.
```
- - -
# Measuring Accuracy
这里原Note的内容和上一讲的内容很多都是重复的，只挑重点来说了。
- 对于`binary perceptron`，我们想要计算它的accuracy，我们的公式是
$$
\large
\begin{align*}
l_{\mathrm{acc}}(w) &= \frac{1}{n}\sum_{i=1}^{n}\big(\operatorname{sgn}(w\cdot f(x_i))==y_i\big)
\end{align*}
$$
其中符号函数:
$$
\large
\begin{align*}
\operatorname{sgn}(x) &= \begin{cases}
1, & x \ge 0\\[4pt]
-1, & x < 0
\end{cases}
\end{align*}
$$
所以表达式:
$$
\large
\begin{align*}
\operatorname{sgn}\big(w \cdot f(x_i)\big)
\end{align*}
$$
就是模型对于第$i$个样本的预测类别。
下面的表达式:
$$
\large
\begin{align*}
(\operatorname{sgn}(w \cdot f(x_i)) == y_i)
\end{align*}
$$
实际上就是预测对了就是1，预测不对就是0。总共进行求和除于总样本数n就是准确率
- 如果我们想要对N个类别进行判断确信程度，我们用的就是上一讲的[Softmax](notes/CS188/Note/Note20#Multi-Class%20Logistic%20Regression)
- 如果我们用[Likelihood](notes/CS188/Note/Note20#Likelihood)来表示在当前权重下模型把所有训练样本都预测成真实标签的概率有多大。
- - -
# Multi-layer Feedforward Neural Networks
首先明白什么叫做`Feedforward`，它指的是信息只从输入层向输出层，中间没有循环结构，也就是只向前传递。
## Activation Function
我们先辨别一下，`Activation Function`和我们上一讲提到的`activation`不是一回事。激活函数接受激活前的输入值之后输出一个激活值。
我们之前用到的激活函数有`step function`，在最原始的`perceptron`中使用的:
![](notes/CS188/static/截屏2026-05-11%2018.03.29.png)
$$
\large
\begin{align*}
f(x) &= \begin{cases}
1, & x \ge 0,\\[4pt]
-1, & \text{otherwise.}
\end{cases}
\end{align*}
$$
这是一个激活函数，但是问题很大，因为它不连续而且几乎处处导数为0。
- - -
然后我们又用到了[`sigmoid Function`](notes/CS188/Note/Note20#Logistic%20Function%20/%20Sigmoid%20Function)
$$
\large
\begin{align*}
g(z) &= \frac{1}{1+e^{-z}}
\end{align*}
$$
即下图

![](notes/CS188/static/截屏2026-05-08%2018.35.03.png)
问题也很明显，当x很大或者很小的时候梯度会非常小
- - -
然后我们又提到了`ReLU`函数，全称为`Rectified Linear Unit`，定义为:
$$
\large
\begin{align*}
\operatorname{ReLU}(x) &= \begin{cases}
0, & x < 0\\[6pt]
x, & x \ge 0
\end{cases}
\end{align*}
$$
![](notes/CS188/static/截屏2026-05-11%2018.02.37.png)
缺点是负数区域梯度为0
可以在[Tensorflow网站](https://playground.tensorflow.org/)查看不同的`Activation Function`对于训练的影响。
- - -
# Backpropagation
首先我们明确一个目标，神经网络训练的核心就是找一个权重让损失最小，从概率角度来说就是找到一个权重让`log-likelihood`最小。这一段和`Backpropagation`没关系
神经网络中的参数非常多，如果直接手动对着每个参数求导会非常麻烦也不现实。`Backpropagation`就是一种能让你高效求梯度的算法。计算图把复杂函数拆成很多简单操作，这里和高数内容基本一致，不再过多赘述，只给几个图片和简单介绍
`Backpropagation`包括前向传播和反向传播两个过程：

- 前向传播就是图片中的绿色部分，这一块就是正常进行算数运算。
- 反向传播就是从最终输出开始，反向计算每个节点对最终输出的影响。

![](notes/CS188/static/截屏2026-05-11%2019.46.20.png)
说明一下图2中梯度为什么要想加，其实就是求偏导，x通过两条路径影响最终输出f:
```text
x → h → g → f
x → i → g → f
```

$$
\large
\begin{align*}
\frac{\partial f}{\partial x} &= \frac{\partial f}{\partial h}\frac{\partial h}{\partial x} + \frac{\partial f}{\partial i}\frac{\partial i}{\partial x} = 4 + 12 = 16
\end{align*}
$$

![](notes/CS188/static/截屏2026-05-11%2019.46.04.png)