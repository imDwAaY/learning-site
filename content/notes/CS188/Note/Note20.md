---
tags:
  - CS188
  - "#Perceptron"
  - "#Binary_Perceptron"
  - "#Linear_Regression"
  - "#Logistic_Regression"
  - "#Multi-Class_Logistic_Regression"
  - "#Softmax"
source: 9.3( Perceptron ) - 9.5( Linear Regression ) & 9.7( Logistic Regressioin ) - 9.8( Multi-Class Logistic Regression )
---
# Perceptron
## Linear Classifiers
我们在上次的Note19中提及到了`Naive Bayes`中的[提取feature](notes/CS188/Note/Note19#Naive%20Bayes)的思想,我们在这里尝试把一个数据点的所有特征提取出来，提取成为一个向量的形式
```text
f(x) = [f1(x), f2(x), ..., fn(x)]
```
与之对应的，每个feature还有一个权重
```text
w = [w1, w2, ..., wn]
```
线性分类器的基本思想是利用特征的线性组合来进行分类，我们把这个值叫做激活值即**activation**。具体公式如下
$$
\large
\begin{align*}
\text{activation}_w(x) &= h_w(x) = \sum_i w_i f_i(x) = \mathbf{w}^\top \mathbf{f}(x) = \mathbf{w}\cdot\mathbf{f}(x)
\end{align*}
$$
我们来着重看一下$h_w(x)$这个值：
如果我们只有两个`lable`，可以回忆一下之前提及到的[垃圾邮件](notes/CS188/Note/Note19#Naive%20Bayes)的例子，这就是只有两个标签--只有ham和spam。

- 这时候$h_w(x)$如果为正，我们就把数据点标记为正类。
- 如果$h_w(x)$值为负，我们就把数据点标记为负类

### Decision Boundary
我们用数学的角度来看一下$h_w(x)$的值：
$$
\large
\begin{align*}
h_{\mathbf{w}}(\mathbf{x}) &= \mathbf{w}\cdot\mathbf{f}(\mathbf{x}) = \|\mathbf{w}\|\;\|\mathbf{f}(\mathbf{x})\|\cos(\theta)
\end{align*}
$$
看最后一串，决定$h_w(x)$值正负的是cosθ，因为两个向量的模是正的。也就是说
$$
\large
\begin{align*}
\text{classify}(\mathbf{x}) &= 
\begin{cases}
+ & \text{if }\ \theta < \dfrac{\pi}{2}\\[6pt]
- & \text{if }\ \theta > \dfrac{\pi}{2}
\end{cases}
\end{align*}
$$
我们已知了向量w，那我们是不是可以画一条与向量w垂直的虚线，任何位于这条线上的特征向量其$h_w(x)$的值都为0，即满足式子$\mathbf{w}^T\mathbf{f}(\mathbf{x}) = 0$,我们把这条线叫做决策边界即`Decision Boundary`
![](notes/CS188/static/截屏2026-05-08%2015.26.15.png)
我们可以根据决策边界来判断$h_w(x)$的值
![](notes/CS188/static/截屏2026-05-08%2015.27.47.png)
- - -
# Binary Perceptron
二分类感知机是一个简单的线性分类器，它的目的是为了找到一个权重向量`w`让训练集中的样本都可以正确分类。
## Perceptron Algorithm
```text
1. Initialize weights: w = 0
2. For each training example (x, y*):
   a. Compute prediction:
      y = classify(x)
   b. If y == y*, do nothing
   c. If y != y*, update weights:
      w ← w + y* f(x)
3. Repeat until all samples are classified correctly in one pass
```
其中：
- `y*`是真实的`lable`
- `y`是模型预测的`lable`
- `f(x)`是样本特征向量
### 算法正确性验证
核心的更新规则就是$w ← w + y^* f(x)$ 
1.我们假设$y^* = 1$，$y = -1$。即原本为正类的数据点被分错分到负类里去了
2.我们可以推断的是：当前的$h_w(x)$是偏小的，我们期望是让$h_w(x)$变大
3.更新后的权重为
$$
\large
\begin{align*}
w' &= w + f(x)
\end{align*}
$$
4.更新后的激活值为
$$
\large
\begin{align*}
h_{w'}(x) &= (w + f(x))^{T} f(x) = w^{T} f(x) + f(x)^{T} f(x) = h_w(x) + f(x)^Tf(x)
\end{align*}
$$
5.因为
$$
\large
\begin{align*}
f(x)^Tf(x) \ge 0
\end{align*}
$$
即激活值会变大，这也就表明了这种更新法则是符合我们的预期--让$h_w(x)$变大
![](notes/CS188/static/截屏2026-05-08%2015.43.56.png)
## Bias
如果我们的决策边界模型只有$w^Tf(x)$,那么我们的决策边界就必须经过原点，这非常限制模型的能力，因为很多不同`lable`的数据点虽然能被一条直线分开，但那条直线不一定经过原点。我们就参考着一次函数的样式加入了`bias term`,让它变成
$$
\large
\begin{align*}
w^{T} f(x) + b &= 0
\end{align*}
$$
实现方法通常是给每个特征向量额外加一个恒等于1的`feature`，然后通过控制权重`w`来控制大小，这样模型仍然可以写成点积的形式:
$$
\large
\begin{align*}
h_w(x) &= w^{T} f(x)
\end{align*}
$$
![](notes/CS188/static/截屏2026-05-08%2015.50.28.png)
- - -
# Multiclass Perceptron
多个类别的和binary非常类似，如果有`K`个`lable`，那么就有`K`个权重。与二分类感知机对应的是，二分类感知机只有一个权重，因为可以用正负来区别两个`lable`.
对于输入样本，计算它的每个`lable`的score:
$$
\large
\begin{align*}
\text{score}_k &= w_k^{\!T}\, f(x)
\end{align*}
$$
选择分数最高的`lable`:
$$
\large
\begin{align*}
\hat{y} &= \arg\max_{k} w_k^{T} f(x)
\end{align*}
$$
## 多分类感知机更新规则
同样的：

- `y*`是真实的并且正确的`lable`
- `y`是被错误预测的`lable`
- `f(x)`是样本特征向量

那么就可以得到：
$$
\large
\begin{align*}
w_{y^*} &\leftarrow w_{y^*} + f(x)\\
w_{y} &\leftarrow w_{y} - f(x)
\end{align*}
$$
给正确类别的权重加上这个样本，给错误类别减去这个样本
- - -
# Linear Regression
和前面不同的是:
- `Regression`预测的是连续的数值，比如房价温度销量等等
- `Classification`预测的是类别
和前面相同的是：
- 模型相同，即权重和特征向量的格式相同
特征向量是
$$
\large
\begin{align*}
x &= [1, x_1, x_2, \dots, x_n]
\end{align*}
$$
对应的权重也是和之前的格式，注意特征向量的第一项`1`是`bias term`，那么我们可以得到$h_w(x)$为：
$$
\large
\begin{align*}
h_w(x) &= w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n = \mathbf{w}^T\mathbf{x}
\end{align*}
$$

## L2 Loss
训练线性回归时，我们希望预测值接近真实值
对于第 `j` 个样本：
$$
\large
\begin{align*}
\text{error}_j &= y_j - h_w(x_j)
\end{align*}
$$
`L2 Loss`是误差平方:
$$
\large
\begin{align*}
(y_j - h_w(x_j))^2
\end{align*}
$$
整个训练集上的loss：
$$
\large
\begin{align*}
Loss(h_w)\; &=\; \frac{1}{2}\sum_{j=1}^{N}\big(y_j - h_w(x_j)\big)^2
\end{align*}
$$
前面加上`1/2`是为了求导时抵消平方项前面的2，让整体公式更加整洁
## Matrix Form矩阵形式
将所有的训练样本堆起来:
$$
\large
\begin{align*}
\mathbf{y} &= \begin{bmatrix}
y_1\\[4pt]
y_2\\[4pt]
\vdots\\[4pt]
y_N
\end{bmatrix}
\end{align*}
$$
设计矩阵：
$$
\large
\begin{align*}
\mathbf{X} &= \begin{bmatrix}
1 & x_{1}^{1} & \cdots & x_{n}^{1} \\
1 & x_{1}^{2} & \cdots & x_{n}^{2} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{1}^{N} & \cdots & x_{n}^{N}
\end{bmatrix}
\end{align*}
$$
权重为:
$$
\large
\begin{align*}
\mathbf{w} &= \begin{bmatrix}
w_0\\[4pt]
w_1\\[4pt]
\vdots\\[4pt]
w_n
\end{bmatrix}
\end{align*}
$$
那么`loss`可以写成：
$$
\large
\begin{align*}
\operatorname{Loss}(h_w) &= \tfrac{1}{2}\left\lVert \mathbf{y} - \mathbf{X}\mathbf{w} \right\rVert_2^2
\end{align*}
$$
线性回归最重要的一个特点是它有闭式解( closed-form solution )
我们对`loss`求梯度:
$$
\large
\begin{align*}
\nabla_w \frac{1}{2}\|y - Xw\|_2^2 &= -X^{T}y + X^{T}Xw
\end{align*}
$$
令梯度为0:
$$
\large
\begin{align*}
X^{T} X w &= X^{T} y
\end{align*}
$$
如果$X^TX$可逆的话，那么就可以得到:
$$
\large
\begin{align*}
\hat{\mathbf{w}} &= (X^\top X)^{-1} X^\top \mathbf{y}
\end{align*}
$$

- - -
# Logistic Regression
Logistic Regression用`logistic function`把线性模型输出转成概率，需要注意的是

> `Logistic Regression` 名字里有 `regression`，但它主要用于 `classification`

## Logistic Function / Sigmoid Function
Logistic Function:
$$
\large
\begin{align*}
g(z) &= \frac{1}{1+e^{-z}}
\end{align*}
$$

其中
$$
\large
\begin{align*}
z &= \mathbf{w}^{T} \mathbf{x}
\end{align*}
$$
$$
\large
\begin{align*}
h_{\mathbf{w}}(\mathbf{x}) &= \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}.
\end{align*}
$$
![](notes/CS188/static/截屏2026-05-08%2018.35.03.png)
它的输出一定在0到1之间，因此可以解释为:
当$h_w(x) > 0.5$时预测为正类，下面的式子为样本属于正类的概率
$$
\large
\begin{align*}
P\bigl(y=+1\mid \mathbf{f}(x);\mathbf{w}\bigr) &= \frac{1}{1+e^{-\mathbf{w}^\top \mathbf{f}(x)}}
\end{align*}
$$
和当$h_w(x) < 0.5$时预测为负类，下面的式子属于样本属于负类的概率
$$
\large
\begin{align*}
P\!\left(y=-1\mid f(\mathbf{x});\mathbf{w}\right) &= 1 - \frac{1}{1 + e^{-\mathbf{w}^\top f(\mathbf{x})}}\,
\end{align*}
$$
## Logistic Regression的损失函数和梯度
首先有一个数学小性质
$$
\large
\begin{align*}
g'(z) &= g(z)\bigl(1 - g(z)\bigr)
\end{align*}
$$
然后我们看`L2 Loss`的函数：
$$
\large
\begin{align*}
Loss(w) &= \tfrac{1}{2}\bigl(y - h_w(x)\bigr)^2
\end{align*}
$$
然后对第`i`个权重求偏导
$$
\begin{align*}
\frac{\partial}{\partial w_i}\frac{1}{2}\big(y-h_w(x)\big)^2 &= \big(y-h_w(x)\big)\frac{\partial}{\partial w_i}\big(y-h_w(x)\big)
= -\big(y-h_w(x)\big)h_w(x)\big(1-h_w(x)\big)x_i
\end{align*}
$$
因为`logistic regression`没有简单的`closed-form solution`, 所以通常用`gradient descent`来估计权重
- - -
# Multi-Class Logistic Regression
和之前Perceptron的思路一样，都是从`binary`变成多类别的，对于多分类逻辑回归我们希望模型输出一个概率分布:
```text
P(y=1|x), P(y=2|x), ..., P(y=K|x)
其中需要满足:
每个概率都 ≥ 0
所有概率加起来 = 1
```
我们用的模型是`Softmax Function`，`Softmax`是`logistic function`的多分类拓展:
对于类别`i`我们有:
$$
\large
\begin{align*}
P\bigl(y=i\mid \mathbf{f(x)};\mathbf{w}\bigr) &= \frac{e^{\mathbf{w}_i^{\top} \mathbf{f(x)}}}{\sum_{k=1}^K e^{\mathbf{w}_k^{\top} \mathbf{f(x)}}}.
\end{align*}
$$
其中:
- 每个类别都有自己的权重向量 `w_i`；
- 每个类别都会得到一个 score；
- 对 score 做指数变换；
- 再除以所有类别指数分数之和；
- 得到每个类别的概率
## Likelihood
我们在这里用`Likelihood`方法来表示参数`w`以使观测到的数据有最大的可能性,我们的训练目标就是最大化这个`likelihood`：
$$
\large
\begin{align*}
\ell(\mathbf{w}_1,\dots,\mathbf{w}_K) &= \prod_{i=1}^n P\big(y_i\mid f(x_i);\mathbf{w}\big)
\end{align*}
$$
注意一下区分:

>`Softmax` 负责算每个类别的概率；  
 `Likelihood` 负责把每个样本“真实类别的概率”拿出来乘在一起

然后我们为了写出多分类的`likelihood`，定义下面:
$$
\large
\begin{align*}
t_{i,k} &= \begin{cases}
1, & y_i = k\\[4pt]
0, & y_i \neq k
\end{cases}
\end{align*}
$$
即:
- 如果第 `i` 个样本真实类别是 `k`，那么 `t_{i,k}=1`；
- 否则 `t_{i,k}=0`
这里举个例子更容易理解，对于某个样本`x_i`，Softmax会输出：
```text
P(猫 | x_i)
P(狗 | x_i)
P(鸟 | x_i)
```
但是真实标签只有一个，比如真实标签是狗，我们只想保留:
```text
P(狗 | x_i)
```
我们就用$t_{i, k}$来表示：
```text
如果第 i 个样本真实类别是 k，那么 t_{i,k} = 1
否则 t_{i,k} = 0
```
所以就可以得到如果真实类别是狗，也就是第二类那么:
```text
t_i = [0, 1, 0]
```
所以说：
$$
\large
\begin{align*}
P(\mathrm{猫}\mid x_i)^{0}\times P(\mathrm{狗}\mid x_i)^{1}\times P(\mathrm{鸟}\mid x_i)^{0} = P(\mathrm{狗}\mid x_i)
\end{align*}
$$
然后我们的`likelihood`公式就可以写成:
$$
\large
\begin{align*}
\ell(\mathbf{w}_1,\dots,\mathbf{w}_K) &= \prod_{i=1}^n \prod_{k=1}^K \left(\frac{e^{\mathbf{w}_k^{\!T} \mathbf{f}(\mathbf{x}_i)}}{\sum_{\ell=1}^K e^{\mathbf{w}_\ell^{\!T} \mathbf{f}(\mathbf{x}_i)}}\right)^{t_{i,k}}
\end{align*}
$$
对应的log- likelihood就是:
$$
\large
\begin{align*}
log\ell(\mathbf{w}_1,\dots,\mathbf{w}_K) &= \sum_{i=1}^n \sum_{k=1}^K t_{i, k}\log\left(\frac{e^{\mathbf{w}_k^{\!T} \mathbf{f}(\mathbf{x}_i)}}{\sum_{\ell=1}^K e^{\mathbf{w}_\ell^{\!T} \mathbf{f}(\mathbf{x}_i)}}\right)
\end{align*}
$$
## Softmax的梯度
$$
\begin{align*}
\nabla_{\mathbf{w}_j}\log \ell(\mathbf w) &= \sum_{i=1}^n \nabla_{\mathbf{w}_j}\sum_{k=1}^K t_{i,k}\log\!\left(\frac{e^{\mathbf{w}_k^{\top}\mathbf{f}(\mathbf{x}_i)}}{\sum_{\ell=1}^K e^{\mathbf{w}_\ell^{\top}\mathbf{f}(\mathbf{x}_i)}}\right)
= \sum_{i=1}^n\left(t_{i,j}-\frac{e^{\mathbf{w}_j^{\top}\mathbf{f}(\mathbf{x}_i)}}{\sum_{\ell=1}^K e^{\mathbf{w}_\ell^{\top}\mathbf{f}(\mathbf{x}_i)}}\right)\mathbf{f}(\mathbf{x}_i)
\end{align*}
$$
