# Lecture_2_Loss_Function_and_Optimization

## 0. 英文词汇

## 1. Loss Function
### 1.1. Loss Function
```math
{(x_i, y_i)}_{i=1}^N,\\
其中x_i是image,y_i是label(在这门课的例子中是相似度评分)\\
L = \frac{1}{N}\sum_i L_i(f(x_i,W), y_i),\\
L_i是单个样本的损失函数,L是所有样本损失函数的平均值
```

### 1.2. Hinge Loss

<img src="https://s2.loli.net/2022/08/24/T6KFvfHzDkSpsM7.png" width="30%">

这里，`$S_j$`是其他分类，`$S_{y_i}$`是正确的分类的分数。这个Loss Function表达的是，如果最后的分数中，正确分类和不正确分类区分度很大的话，就认为Loss Function为0，否则Loss Funtion线性变化
- Example

<img src="https://s2.loli.net/2022/08/24/fZvVj1QThtHMlbW.png" width="100%">

#### 1.2.1. 思考
1. W是唯一的吗？
不是的。毕竟W可以缩放的。这样看来，我们选取的`$S_{y_i} - S_j + 1$`实际上，这个1是和W的缩放是等价的。毕竟如果我将W翻倍，那么两个相似度值的差也会翻倍，这样的话，相当于是1只起到了0.5的效果。

### 1.3. 正则化项
#### 1.3.1. 过拟合
machine learning的时候，我们并不关心classidier在数据集中拟合的如何，我们关心的是，classifier在测试集中拟合的如何。就好像，我们完全可以使用一条极其复杂的曲线，将1000个点串联起来，但是这样做在预测数据的时候是完全错误的，我们要做的反而是用一条直线将这些点拟合。

#### 1.3.2. ragularization正则化项
为了鼓励classifier用更简单的W拟合数据点，我们为Loss Functino 加上正则化项
<img src="https://s2.loli.net/2022/08/27/gqS75KTXcGUmjxN.png" width="50%">
```math
L(W) = \frac{1}{N}\sum_{i=1}^N L_i(f(x_i, W), y_i)+\lambda R(W)
```
R为ragularization penalty(正则惩罚项), `$\lambda$`是超参数，用于平衡损失项和正则项
<img src="https://s2.loli.net/2022/08/27/yHl18q79FnvhmZc.png" width="100%">

## 2. Softmax classifier
### 2.1. Multinomial logistic loss funtion
scores = unnormalized log probabilities of the classes：
(用指数让概率为正，并进行归一化，而且指数会让较大的数变得更大，从而使得概率接近1)
```math
P(Y=k|X=x_i) = \frac{e^{S_k}}{\sum_j e^{S_j}}\\
where \,S = f(x_i;W)
```
让true class的概率尽可能高且接近于1,我们在概率前加上log，保证较大的概率接近1，且log函数单调。
```math
L_i = -\log P(Y=y_i|X=x_i)\\
L_i = -\log(\frac{e^{S_{y_i}}}{\sum_j e^{S_j}})
```

## 3. Optimization
### 3.1. 梯度下降
#### 3.1.1. 有限差分法求解gradient(infinite definition)

<img src="https://s2.loli.net/2022/08/27/ED1BMnRemQ7urjy.png" width="100%">

这种方法计算gradient非常慢，计算量很大。通常在CNN中向量的维度会很高。不过这个方法可以用来debug，来验证自己求得的解析梯度是否和数值梯度一样。
#### 3.1.2. 伪代码
```
# 首先，用随机数(seed)初始化W矩阵
while True:
    # 求解gradient
    weight_grad = evaluate_gradient(loss_fun, data, weights)
    # 沿着gradient相反的方向更新W
    weight += - step_size * weight_grad
```
这里，步长step size是一个超参数
#### 3.1.3. 随机梯度下降
```math
L(W) = \frac{1}{N}\sum_{i=1}^N L_i(f(x_i, W), y_i)+\lambda R(W)
```
在一些很大的数据集中，比如ImageNet，N常常会很大，这样的话，如果所有的data都计算一遍梯度的话，梯度的更新将会很慢。所以我们采用随机梯度下降的方法进行计算。每次取一些小规模样本计算梯度(minibatch), 用ninibatch来估算梯度

#### 3.2. Image feature
对于一些多模态的问题（比如说上一讲中，马头朝向问题），很多时候我们不能直接将原属像素当作数据进行训练。传统上我们需要人为提取特征向量来进行训练。

这本质上，是因为原始像素数据是线性不可分的。我们需要通过空间变换将其转化为线性可分的问题。比如在下图中，用将笛卡尔坐标变换为极坐标之后，就转化为来一个线性可分问题。

<img src="https://s2.loli.net/2022/08/27/cXPvY52LUd3JfSx.png" width="100%">

example: Histogram of Oriented Gradient边缘梯度