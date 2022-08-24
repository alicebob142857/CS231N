# Lecture 1 -- Image Classification Pipeline 图像分类

## 0. 英文词汇
holy grail 圣杯

## 1. 数据驱动方法
### 1.1. problem : Semantic gap我们为图像提供的标签和图像本身像素的性质有很大的不同。对图像的微调就会对像素网格造成巨大的变化。我们成对这种变化的抵抗为鲁棒性(robust)
1. 拍摄角度，光线等差异
2. 同一类物体不同个体的差异
3. 遮挡环境

### 1.2. method
数据驱动方法：
训练函数 + 预测函数

#### 1.2.1. Distance Metric to compare images最近邻分类器(nearest neighbor)
L1 distance:
```math
d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|
```
example:
<img src="https://s2.loli.net/2022/08/09/gLBKRyhCXS9Fvjd.png" width="100%">

#### 1.2.2. 缺点
1. 我们希望一个模型在训练时可以比较满，但是在预测时需要很快。然而，最近邻分类器几乎没有训练（只是加载了图像集），但是他在预测的时候工作量很大，需要逐一对比所有的数据。
2. 给个一维的例子，120cm一下的一般是孩子(8岁以下)，150cm以上成人(18以上)，如果是nn算法，给一个身高110的数据，一般是分成孩子类，但是数据集就有个噪音，有个残疾人双腿截肢了一截，身高恰好110，然后110cm想要分类的数据就会被到截肢的这个人一个类。这样会导致最近邻算法对噪音的抗性很差。（就像图片中绿色区域中间有一个黄色的点，但是这个区域应该是绿色区域）

<img src="https://s2.loli.net/2022/08/09/tLjVDcEvwOoen8C.png" width="100%">

## 2. k-nearest neighbor k近邻分类

### 2.1. k近邻
k近邻：对周围最近的k个点进行投票，防止过拟合的发生
<img src="https://s2.loli.net/2022/08/09/QfWRTBV7tpvxU6i.png" width="100%">
noticeable:
绿色中间的黄色区域，以及红色和蓝色犬牙交错的区域都没有了。中间的白色区域表示没有获得任何一个颜色的投票，白色区域没有最近邻

### 2.2. L2距离（欧氏距离）
<img src="https://s2.loli.net/2022/08/09/jYofnqQNsyXSUtP.png" width="100%">
noticeable:欧氏距离经过坐标系的旋转变换是不变的。可以根据这一点选择是使用L1还是L2距离。

### 2.3. Hyperparameters超参数
模型不能学习的参数。你需要提前为模型预设这些参数（比如k近邻算法里面的k, 以及是使用L1距离还是L2距离

#### 2.3.1. 如何设置超参数
idea #1: 选择在数据集上运行准确率最高的超参数
<font color="red"> **BAD!!!**</font>我们训练数据不是为了让模型最好地拟合数据，而是需要更好的预测未出现的数据。也许k=1的最近邻效果最好，但是我们需要使用更大的k，这样模型才能最robust

Idea #2: 将数据集分成两份，一份用于训练，一份用于测试，选择在测试数据上运行最好的超参数
<font color="red">**BAD!!!**</font>同样，机器学习的目的是为了让我们了解我们的算法表现如何。这种idea也会导致在新数据集上过拟合。

idea #3: 将数据集分为三份，一份为train,一份为validation,一份为test。先在train上训练模型，再在validation上调整参数，找出最好的超参数，最后在test上进行验证，看这组超参数到底如何。这个数据将是评价一组超参数是否合理的最终数据(是要写在论文上的)，检验是否出现过拟合。
<font color="green">RIGHT!!!</font>关键就是隔离valifdation和test，就像考试的时候不能看答案一样。

#### 2.3.2. Cross Validation(适用于小数据)

#### 2.3.3. k近邻的缺点
1. 对图像进行平移(shift)，染色(tint)，遮挡(box)的时候，可以刻意地让这几张图片的L2距离相同，但是显然这几张图片有明显的不同。向量化的表达方式很不适合表达图像之间的视觉感知差异。
2. curse of dimensionality 维度灾难
我们看到k近邻上一张图。如果我们使用k近邻算法的话，我们需要很稠密的数据。因为数据过于稀疏，那么稀疏部分和样本的相似度就不高，会被划分为白色区域。
一个图片会包含很多的特征。一个特征占据一个维度。每个维度都需要密集地数据去占据这个空间。这样会导致数据量非常巨大。

## 3. Linear Classification
### 3.1. Parameter Approach
#### 3.1.1. Introduction
<img src="https://s2.loli.net/2022/08/24/Zc6htSMfqY2W3sF.png" width="100%">
f最简单的设计方法就是将x和W相乘

```math
f(x, W) = Wx
```
其中，x表示数据集中的数据，为`$32\times 32\times 3 = 3072$`维响亮，f为10维向量(相似度的评分)，所以W为`$10\times 3072$`维矩阵
有时候我们会给上面的式子加上bias
```math
f(x, W) = Wx + b
```
b是一个10维向量。不与数据集交互，只会给我们一些人为的偏好值(例如，如果数据集中猫比狗要多，那么应该将bias里面狗的偏好值调高一点以平衡这个数据集)

值得注意的是，矩阵的行向量和x是相同维度的。这些行向量是可以可视化的。他们相当于是一个模版，相似度越高(模一样的情况下)的向量点乘的值越大

#### 3.1.2 Linear Classifier的困境

<img src="https://s2.loli.net/2022/08/24/AzgBTnxXMfhFZ17.png" width="100%">

我们看到，classifier认为，车是红色的，有一块深色阴影(挡风玻璃)的物体。这样又一个缺陷就是，一个对象只有一个模版。当一个对象出现一些变体的时候，classifier会求解所有变体并求平均值，并且只使用单一的模版识别所有的类别。会导致classifier判断出错。
例如，我们看到，马的模版中，由于马一般在草地上，所以图像下方是绿色的(这个其实是无关的特征)，而且，这个马有两个头。

<img src="https://s2.loli.net/2022/08/24/RUASevWhnKa1xHb.png" width="100%">

Linear Classifier就是在高维空间用“直线”将不同的类别区分开，而有些情况是线性不可分的

<img src="https://s2.loli.net/2022/08/24/ESA2LCkhTBteZDG.png" width="100%">

