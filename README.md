# [数据分析中的降维方法初探](https://www.cnblogs.com/LittleHann/p/6558575.html)

**阅读目录(Content)**

- [1. 引言](https://www.cnblogs.com/littlehann/p/6558575.html#_label0)

- - [0x1：从维灾难说起](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_0_0)
  - [0x2：降维定义](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_0_1)
  - [0x3：为什么要降维](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_0_2)

- [2. PCA主成分分析(Principal components analysis)](https://www.cnblogs.com/littlehann/p/6558575.html#_label1)

- - [0x1：PCA算法模型](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_1_0)

  - [0x2: 在讨论PCA约简前先要讨论向量的表示及基变换 - PCA低损压缩的理论基础](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_1_1)

  - - [1. 内积与投影](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_1_0)
    - [2. 基](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_1_1)
    - [3. 基变换的矩阵表示 ](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_1_2)

  - [0x3: 协方差矩阵及优化目标 - 如何找到损失最低的变换基](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_1_2)

  - - [1. 投影后的新坐标点的方差 - 一种表征信息丢失程度的度量](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_2_0)
    - [2. 协方差](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_2_1)
    - [3. 协方差矩阵 - 字段内方差及字段间协方差的统一数学表示](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_2_2)
    - [4. 协方差矩阵对角化](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_2_3)

  - [0x4：PCA优化问题的解 - 和协方差矩阵对角化的关系](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_1_3)

  - [0x5: PCA算法过程](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_1_4)

  - - [1. PCA算法过程公式化描述](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_4_0)
    - [2. 一个例子](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_4_1)

  - [0x6：PCA的限制](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_1_5)

  - - [1. 它可以很好的解除线性相关，但是对于高阶相关性就没有办法了](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_5_0)
    - [2. PCA假设数据各主特征是分布在正交方向上，如果在非正交方向上存在几个方差较大的方向，PCA的效果就大打折扣了](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_5_1)
    - [3. PCA是一种无参数技术，无法实现个性优化](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_5_2)

  - [0x7: 基于原生python+numpy实现PCA算法](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_1_6)

  - [0x8: 对图像数据应用PCA算法](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_1_7)

  - - [1. 利用PCA进行人脸识别](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_1_7_0)

  - [0x9: 选择主成分个数](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_1_8)

- [3. Random Projection（随机投影）](https://www.cnblogs.com/littlehann/p/6558575.html#_label2)

- - [0x1：为什么需要随机投影](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_2_0)

  - [0x2: Johnson–Lindenstrauss lemma，随机投影有效性的理论依据](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_2_1)

  - - [1. 问题定义](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_2_1_0)
    - [2. 问题证明](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_2_1_1)

  - [0x3: Random projection算法](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_2_2)

  - - [1. Gaussian random projection - 高斯随机投影](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_2_2_0)
    - [2. a uniform random k-dimensional subspace.](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_2_2_1)
    - [3. Sparse random projection](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_2_2_2)
    - [4. More computationally efficient random projections](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_2_2_3)

  - [0x4: Random projection在手写图像识别里的具体应用](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_2_3)

  - - [1. 计算100%完全分类的baseline基线值](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_2_3_0)
    - [2. random projection的components个数是否和欧几里德分类准确度是正相关的](https://www.cnblogs.com/littlehann/p/6558575.html#_label3_2_3_1)

- [4. 压缩感知](https://www.cnblogs.com/littlehann/p/6558575.html#_label3)

- [5. 主成分回归（principle component regression，PCR）](https://www.cnblogs.com/littlehann/p/6558575.html#_label4)

- - [0x1：主成份的组成](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_4_0)

- [5. 偏最小二乘回归（partial least-squares regression）](https://www.cnblogs.com/littlehann/p/6558575.html#_label5)

- - [0x1：偏最小二乘计算方法](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_5_0)

- [7. 降维在其他领域的应用](https://www.cnblogs.com/littlehann/p/6558575.html#_label6)

- - [0x1: PHP SSDEEP模糊化HASH](https://www.cnblogs.com/littlehann/p/6558575.html#_lab2_6_0)

[回到顶部(go to top)](https://www.cnblogs.com/littlehann/p/6558575.html#_labelTop)

# 1. 引言



## 0x1：从维灾难说起

在多项式曲线拟合的例子中，我们只有一个输入变量x。但是对于模式识别的实际应用来说，我们不得不处理由许多输入变量组成的高维空间，这个问题是个很大的挑战，也是影响模式识别技术设计的重要因素。

为了说明这个问题，我们考虑一个人工合成的数据集。这个数据集中的数据表示一个管道中 石油、水、天然气各自所占的比例。这三种物质在管道中的几何形状有三种不同的配置，被称为：

- “同质状”
- “环状”
- “薄片状”

三种物质各自的比例也会变化。这个测量技术的原则的思想是，如果一窄束伽马射线穿过管道，射线强度的衰减提供了管道中材料密度的信息。例如，射线通过石油之后的衰减会强于通过天然气之后的衰减。

![img](https://img2020.cnblogs.com/blog/532548/202007/532548-20200705220200806-1382837125.png)

石油、水、天然气的三种几何配置，用来生成石油流数据集。对于每种配置，三种成分的比例可以改变

![img](https://img2020.cnblogs.com/blog/532548/202007/532548-20200705220227570-305884719.png)

管道的横切面，表示六个射线束的配置，每个射线对应着一个双能量伽马射线密度计。注意，垂直射线束关于中心轴(虚线表示)不是对称的。 

每个数据点由一个12维的输入向量组成。输入向量是伽马射线密度计的读数，度量了一窄束伽马射线穿过管道后强度的衰减。

下图图给出了数据集里的100个点，每个点只画出了两个分量x6和x7(为了说明的方便，剩余的10个分量被忽略)。

![img](https://img2020.cnblogs.com/blog/532548/202007/532548-20200705220355154-665041171.png)

石油流数据的输入变量x6和x7的散点图，其中红色表示“同质状”类别，绿色表示“环状”类别，蓝 色表示“薄片状”类别。我们的目标是分类新的数据点，记作“×”。 

每个数据点根据它属于的三种几何类别之一被标记。我们的目标是使用这个数据作为训练集，训练一个模型，能够对于一个新的(x6, x7)的观测(图中标记为“叉”的点)进行分类。

我们观察到，标记为“叉”的点周围由许多红色的点，因此我们可以猜想它属于红色类别。然而，它附近也有很多绿色的点，因此我们也可以猜想它属于绿色类别。似乎它不太可能属于蓝色类别。

直观看来，标记为“叉”的点的类别应该与训练集中它附近的点强烈相关，与距离比较远的点的相关性比较弱。事实上，这种直观的想法是合理的。我们如何把这种直观想法转化为学习算法呢? 

一种简单的方式是把输入空间划分成小的单元格，如下图所示。当给出测试点，我们要预测类别的时候，我们首先判断它属于哪个单元格，然后我们寻找训练集中落在同一个单元格中的训练数据点。测试点的类别就是测试点所在的单元格中数量最多的训练数据点的类别，这其实就是KNN的原理。

![img](https://img2020.cnblogs.com/blog/532548/202007/532548-20200705220955062-163942081.png)

这种朴素的观点有很多问题。当需要处理的问题有很多输入数据，并且对应于高维的输入空间时，有一个问题就变得尤为突出。

问题的来源如下图所示。

![img](https://img2020.cnblogs.com/blog/532548/202007/532548-20200705221043354-1084213882.png)

如果我们把空间的区域分割成一个个的单元格，那么这些单元格的数量会随着空间的维数以指数的形式增大。当单元格的数量指数增大时，为了保证单元格不为空，我们就不得不需要指数量级的训练数据。 

让我们回到多项式拟合的问题，考虑一 下我们如何把上面的方法推广到输入空间有多个变量的情形。如果我们有D个输入变量，那么一个三阶多项式就可以写成如下的形式 

![img](https://img2020.cnblogs.com/blog/532548/202007/532548-20200705221128135-593313672.png)

随着D的增加，独立的系数的数量（并非所有的系数都独立，因为变量x之间的互换对称性）的增长速度正比于D3。

在实际应用中，为了描述数据中复杂的依存关系，我们可能需要使用高阶多项式。对于一个M阶多项式，系数数量的增长速度类似于DM 。虽然增长速度是一个幂函数，而不是指数函数，但是这仍然说明了，这种方法会迅速变得很笨重，因此在实际应用中很受限。 

我们在三维空间中建立的几何直觉会在考虑高维空间时不起作用。例如，考虑D维空间的一 个半径r = 1的球体，请问，位于半径r = 1 − ε和半径r = 1之间的部分占球的总体积的百分比是多少？

我们注意到，D维空间的半径为r的球体的体积一定是rD 的倍数，因此我们有：

![img](https://img2020.cnblogs.com/blog/532548/202007/532548-20200705221442934-652870895.png)

其中常数KD 值依赖于D。因此我们要求解的体积比就是：

![img](https://img2020.cnblogs.com/blog/532548/202007/532548-20200705221511117-1220375059.png)

![img](https://img2020.cnblogs.com/blog/532548/202007/532548-20200705221522430-1770977329.png)

上图给出了不同D值下，上式与ε的关系。我们看到，对于较大的D，这个体积比趋近于1，即使对于小的ε也是这样。

因此，**在高维空间中，一个球体的大部分体积都聚集在表面附近的薄球壳上**! 

考虑高维空间的高斯分布的行为。如果我们从笛卡 尔坐标系变换到极坐标系，然后把方向变量积分出来，我们就得到了一个概率密度的表达式p(r)，这个表达式是关于距离原点的半径r的函数。

因此 p(r)δr 就是位于半径 r 处厚度为 δr 的薄 球壳内部的概率质量。对于不同的D值，这个概率分布的图像如下图所示。我们看到，对于大 的D值，高斯分布的概率质量集中在薄球壳处。

![img](https://img2020.cnblogs.com/blog/532548/202007/532548-20200705221650151-633665868.png)

不同的维度D中的高斯分布的概率密度关于半径r的关系。在高维空间中，高斯分布的大部分概 率质量位于某个半径上的一个薄球壳上。 

高维空间产生的这种困难有时被称为**维度灾难（curse of dimensionality）**。

虽然维度灾难在模式识别应用中是一个重要的问题，但是它并不能阻止我们寻找应用于高维空间的有效技术。原因有两方面。

- 第一，真实的数据经常被限制在有着较低的有效维度的空间区域中，特别地，在目标值会发生重要变化的方向上也会有这种限制。
- 第二，真实数据通常比较光滑(至少局部上比较光滑)，因此大多数情况下，对于输入变量的微小改变，目标值的改变也很小，因此对于新的输入变量，我们可以通过局部的类似于插值的技术来进行预测。

成功的模式识别技术利用上述的两个性质中的一个，或者都用。  



## 0x2：降维定义

降维是将高维数据映射到低维空间的过程，该过程与信息论中有损压缩概念密切相关。同时要明白的，不存在完全无损的降维。

有很多种算法可以完成对原始数据的降维，在这些方法中，降维是通过对原始数据的线性变换实现的。即，如果原始数据是 d 维的，我们想将其约简到 n 维（n < d），则需要找到一个矩阵![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722093644215-37847583.png)使得映射![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722093657172-1384158722.png)。选择 W 的一个最自然的选择的是在降维的同时那能够复原原始的数据 x，但通常这是不可能，区别只是损失多少的问题。



## 0x3：为什么要降维

降维的原因通常有以下几个：

```
1. 首先，高维数据增加了运算的难度
2. 其次，高维使得学习算法的泛化能力变弱（例如，在最近邻分类器中，样本复杂度随着维度成指数增长），维度越高，算法的搜索难度和成本就越大。
3. 最后，降维能够增加数据的可读性，利于发掘数据的有意义的结构
```

以一个具体的业务场景来说：

malware detection这种non-linear分类问题中，我们提取的feature往往是sparce high-dimension vector(稀疏高维向量)，典型地例如对malware binary的code .text section提取byte n-gram，这个时候，x轴(代码段的byte向量)高达45w，再乘上y轴(最少也是256)，直接就遇到了维数灾难问题，导致神经网络求解速度极慢，甚至内存MMO问题。

这个时候就需要维度约简技术，值得注意的是，深度神经网络CNN本身就包含“冗余信息剔除”机制，在完成了对训练样本的拟合之后，网络之后的权重调整会朝着剔除训练样本中的信息冗余目标前进，即我们所谓的信息瓶颈。

**Relevant Link:**

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
http://www.tomshardware.com/news/deep-instinct-deep-learning-malware-detection,31079.html
https://www.computerpoweruser.com/article/18961/israeli-company-aims-to-be-first-to-apply-deep-learning-to-cybersecurity
https://www.technologyreview.com/s/542971/antivirus-that-mimics-the-brain-could-catch-more-malware/
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MalwareRandomProjections.pdf
http://e-nns.org/
https://arxiv.org/pdf/1703.02244.pdf
http://www.dartmouth.edu/~gvc/
http://www.cs.toronto.edu/~gdahl/ 
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

 

[回到顶部(go to top)](https://www.cnblogs.com/littlehann/p/6558575.html#_labelTop)

# 2. PCA主成分分析(Principal components analysis)



## 0x1：PCA算法模型

令 x1，....，xm 为 m 个 d 维向量，我们想利用线性变换对这些向量进行降维。给定矩阵![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722094506040-757879939.png)，则存在映射![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722093657172-1384158722.png)，其中![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722094544055-188950211.png)是 x 的低维表示。

另外，矩阵![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722094609284-972235222.png)能够将压缩后的信息（近似）复原为原始的信号。即，对于压缩向量![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722094644126-2026599233.png)，其中 y 在低维空间![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722094659396-1765595769.png)中，我们能够构建![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722094714110-1020749824.png)，使得![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722094747607-1880622874.png)是 x 的复原版本，处于原始的高维空间![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722094820973-1466636501.png)中。

在PCA中，我们要找的压缩矩阵 W 和复原矩阵 U 使得原始信号和复原信号在平方距离上最小，即，我们需要求解如下问题：

![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722095425922-1046756971.png)，即尽量无损压缩。

**令（U，W）是上式的一个解，则 U 的列是单位正交的（即![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722101553492-753549215.png)是![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722094659396-1765595769.png)上的单位矩阵）以及![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722101625135-350673595.png)**

PCA(Principal Component Analysis)是一种常用的数据分析方法。PCA通过线性变换将原始数据变换为一组各维度线性无关（单位正交）的表示，可用于提取数据的主要特征分量，常用于高维数据的降维。

**其实“信息瓶颈理论”的核心观点也是认为：所有的信息都是存在冗余的，其需要抽取其中最核心关键的部分就可以大致代表该原始信息。**

降维当然意味着信息的丢失，不过鉴于实际数据本身常常存在的相关性，我们可以想办法在降维的同时将信息的损失尽量降低



## 0x2: 在讨论PCA约简前先要讨论向量的表示及基变换 - PCA低损压缩的理论基础

既然我们面对的数据被抽象为一组向量，那么下面有必要研究一些向量的数学性质。而这些数学性质将成为后续导出PCA的理论基础



### 1. 内积与投影

向量运算内积。两个维数相同的向量的内积被定义为，即向量对应的各维度元素两两相乘累加和。

```
(a1,a2,⋯,an)T⋅(b1,b2,⋯,bn)T=a1b1+a2b2+⋯+anbn
```

内积运算将两个向量映射为一个实数。其计算方式非常容易理解，但是其意义并不明显。

下面我们分析内积的几何意义。假设A和B是两个n维向量，我们知道n维向量可以等价表示为n维空间中的一条从原点发射的有向线段，为了简单起见我们假设A和B均为二维向量，则A=(x1,y1)，B=(x2,y2)。则在二维平面上A和B可以用两条发自原点的有向线段表示。现在我们从A点向B所在直线引一条垂线。我们知道垂线与B的交点叫做A在B上的投影，再设A与B的夹角是a，如下图所示：

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316125807823-306818668.png)

则投影的矢量长度为|A|cos(a)，其中![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316130121760-1332681301.png)是向量A的模，也就是A线段的标量长度。注意这里我们专门区分了矢量长度和标量长度，标量长度总是大于等于0，值就是线段的长度；而矢量长度可能为负，其绝对值是线段长度，而符号取决于其方向与标准方向相同或相反。

接着我们将内积表示为另一种我们熟悉的形式：![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316130226276-842001153.png)。A与B的内积等于：**A到B的投影长度乘以B的模**。再进一步，如果我们假设B的模为1，即让|B|=1，那么就变成了![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316130306166-65626745.png)，可以看到：**设向量B的模为1，则A与B的内积值等于A向B所在直线投影的矢量长度**

这就是内积的一种几何解释！！



### **2. 基**

上文说过，一个二维向量可以对应二维笛卡尔直角坐标系中从原点出发的一个有向线段

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316131309135-1141095304.png)

在代数表示方面，我们经常用线段终点的点坐标表示向量，例如上面的向量可以表示为(3,2)，不过我们常常忽略，只有一个(3,2)本身是不能够精确表示一个向量的。我们仔细看一下，这里的3实际表示的是向量在x轴上的投影值是3，在y轴上的投影值是2。

也就是说我们其实隐式引入了一个定义：**以x轴和y轴上正方向长度为1的向量为标准**。那么一个向量(3,2)实际是说在x轴投影为3而y轴的投影为2。注意投影是一个矢量，所以可以为负。
更正式的说，向量(x,y)实际上表示线性组合 

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316131325573-7733229.png)

**所有二维向量都可以表示为一定数量的基的线性组合**。此处(1,0)和(0,1)叫做二维空间中的一组基

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316131617010-1936178248.png)

所以，要准确描述向量，首先要确定一组基，然后给出在基所在的各个直线上的投影值，就可以了。只不过我们经常省略第一步，而默认以(1,0)和(0,1)为基

我们之所以默认选择(1,0)和(0,1)为基，当然是比较方便，因为它们分别是x和y轴正方向上的单位向量，因此就使得二维平面上点坐标和向量一一对应，非常方便。

但实际上**任何两个线性无关的二维向量都可以成为一组基（基不一定要正交，正交是一个更强的条件）**，所谓线性无关在二维平面内可以直观认为是两个不在一条直线上的向量(这个概念非常重要，因为PCA分析中用于降维投影的基常常就不是x/y轴单位向量)

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316132119151-1405701427.png)

这里就引出了一个概念，坐标是一个相对的概念，只是我们平时见标准的0-90.的坐标轴看多了，其实所有的向量/坐标都是一个相对于坐标轴的概念值而已。

另外这里要注意的是，我们列举的例子中基是正交的（即内积为0，或直观说相互垂直），**但可以成为一组基的唯一要求就是线性无关，非正交的基也是可以的**。不过因为正交基有较好的性质，所以一般使用的基都是正交的。

我们来继续看上图，（1,1）和（-1,1）也可以成为一组基。一般来说，我们希望基的模是1，因为从内积的意义可以看到，如果基的模是1，那么就可以方便的用向量点乘基而直接获得其在新基上的坐标了！

实际上，对应任何一个向量我们总可以找到其同方向上模为1的向量，只要让两个分量分别除以模就好了。例如，上面的基可以变为![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316133506807-660732440.png)，各个基的模为1



### 3. 基变换的矩阵表示  

在上一小节我们讨论了一个非常重要的概念，即任意的一组线性无关的向量都可以表示为基，而不仅限于90°的 x-y 坐标轴。

同时我们现在熟悉的（x，y）坐标其实本质是在一组特定基上的表示方法，一旦我们的基发生概念，坐标值也会发生改变，这个改变的过程就叫基变换。

我们换一种更简便的方式来表示基变换，继续以上图的坐标系为例：

**将(3,2)变换为新基上的坐标，就是用(3,2)与第新基的各分量分别做内积运算(将一个坐标系上的点"转换"到另一个坐标系本质就是在投影)，得到的结果作为第新的坐标**。实际上，我们可以用矩阵相乘的形式简洁的表示这个变换，这里![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316133506807-660732440.png)是新基的向量，（3，2）是原基的向量。

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316133953260-867278434.png)

其中矩阵的两行分别为两个基，乘以原向量，其结果刚好为新基的坐标。

可以稍微推广一下，如果我们有m个二维向量，只要将二维向量**按列**排成一个两行m列矩阵，然后用“基矩阵”乘以这个矩阵，就得到了所有这些向量在新基下的值。例如(1,1)，(2,2)，(3,3)，想变换到刚才那组基上，则可以这样表示

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316134230104-1701777513.png)

于是一组向量的基变换被干净的表示为矩阵的相乘

一般的，如果我们有M个N维向量，想将其变换为由R个N维向量表示的新空间中，那么首先将**R个基按行组成矩阵A**，然后将**原始向量按列组成矩阵B**，那么两矩阵的乘积AB就是变换结果，其中AB的第m列为A中第m列变换后的结果。

数学表示为

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316135107682-365315199.png)

（R x N）. （N x M）=（R x M）

其中pi是一个行向量，表示第i个基；aj是一个列向量，表示第j个原始数据记录

特别要注意的是，**这里R可以小于N，而R决定了变换后数据的维数。也就是说，我们可以将 N维数据变换到更低维度的空间中去，变换后的维度取决于基的数量。因此这种矩阵相乘的表示也可以表示"降维变换"**

最后，上述分析同时给矩阵相乘找到了一种物理解释：**两个矩阵相乘的意义是将右边矩阵中的每一列列向量变换到左边矩阵中每一行行向量为基所表示的空间中去**。更抽象的说，**一个矩阵可以表示一种线性变换。**



## 0x3: 协方差矩阵及优化目标 - 如何找到损失最低的变换基

上面我们讨论了选择不同的基可以对同样一组数据给出不同的表示，而且如果基的数量少于向量本身的维数，则可以达到降维的效果。

但是我们还没有回答一个最最关键的问题：**如何选择基才是最优的**。或者说，如果我们有一组N维向量，现在要将其降到K维（K小于N），那么我们应该如何选择K个基才能最大程度保留原有的信息
假设我们的数据由五条记录组成，将它们表示成矩阵形式：

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316135901729-276506540.png)

其中每一列为一条数据记录(列向量)，而一行为一个字段。为了后续处理方便，我们首先将每个字段内所有值都减去字段均值，其结果是将每个字段都变为均值为0。我们看上面的数据，第一个字段均值为2，第二个字段均值为3，所以变换后：

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316140134666-1821041506.png)

我们可以看下五条数据在平面直角坐标系内的样子

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316140206901-226577857.png)

现在问题来了：如果我们必须使用一组新的基来表示这些数据，又希望尽量保留原始的信息（保留原始数据的概率分布），我们应该如何选择？

通过上一节对基变换的讨论我们知道，这个问题实际上是要在二维平面中选择一个方向，将所有数据都投影到这个方向所在直线上，用投影值表示原始记录。这是一个实际的二维降到一维的问题。
那么如何选择这个方向(或者说基)才能尽量保留最多的原始信息呢？**一种直观的看法是：希望投影后的投影值尽可能分散。数据越分散，可分性就越强，可分性越强，概率分布保存的就越完整**。
以上图为例：

可以看出如果向x轴投影，那么最左边的两个点会重叠在一起，中间的两个点也会重叠在一起，于是本身四个各不相同的二维点投影后只剩下两个不同的值了，这是一种严重的信息丢失。

同理，如果向y轴投影最上面的两个点和分布在x轴上的两个点也会重叠。

所以看来x和y轴都不是最好的投影选择。我们直观目测，如果向通过第一象限和第三象限的斜线投影，则五个点在投影后还是可以区分的。
下面，我们用数学方法表述和讨论这个问题



### 1. 投影后的新坐标点的方差 - 一种表征信息丢失程度的度量

上文说到，我们希望投影后投影值尽可能分散，而这种分散程度，可以用数学上的方差来表述。此处，一个字段的方差可以看做是每个元素与字段均值的差的平方和的均值，即

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316142323541-1010284860.png)

**在使用应用中，在运行PCA之前需要对样本进行“中心化”。即，我们首先计算![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722155603843-1872330243.png)，然后再进行PCA过程。**

由于上面我们已经将每个字段的均值都化为0了，因此方差可以直接用每个元素的平方和除以元素个数表示

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316142354573-1709792255.png)

于是上面的问题被形式化表述为：**寻找一个一维基，使得所有数据变换为这个基上的坐标表示后，方差值最大**



### 2. 协方差

对于二维降成一维的问题来说，找到那个使得方差最大的方向就可以了。不过对于更高维，还有一个问题需要解决。

考虑三维降到二维问题。与之前相同，首先我们希望找到一个方向使得投影后方差最大，这样就完成了第一个方向的选择，继而我们选择第二个投影方向。如果我们还是单纯只选择方差最大的方向，很明显，这个方向与第一个方向应该是“几乎重合在一起”，显然这样的维度是没有用的，因此，应该有其他约束条件。

**从直观上说，让两个字段尽可能表示更多的原始信息，我们是不希望它们之间存在（线性）相关性的，因为相关性意味着两个字段不是完全独立，必然存在重复表示的信息**。
数学上可以用两个字段的协方差表示其相关性，由于已经让每个字段均值为0，则

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316142916260-426093925.png)

可以看到，在字段均值为0的情况下，两个字段的协方差简洁的表示为其内积除以元素数m
当协方差为0时，表示两个字段完全独立。**为了让协方差为0，我们选择第二个基时只能在与第一个基正交的方向上选择。因此最终选择的两个方向一定是正交的**。至此，我们得到了降维问题的优化目标

```
将一组N维向量降为K维(K大于0，小于N)，其目标是选择K个单位(模为1)正交基，使得原始数据变换到这组基上后，各字段两两间协方差为0(各自独立)；而字段的方差则尽可能大(投影后的点尽可能离散)。在正交的约束下，取最大的K个方差
```



### 3. 协方差矩阵 - 字段内方差及字段间协方差的统一数学表示

我们看到，最终要达到的目的与字段内方差及字段间协方差有密切关系。因此我们希望能将两者统一表示，仔细观察发现，两者均可以表示为内积的形式，而内积又与矩阵相乘密切相关
假设我们只有a和b两个字段，那么我们将它们按行组成矩阵X

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316145314120-2047820321.png)

然后我们用X乘以X的转置，并乘上系数1/m

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316145401276-1638186705.png)

这个矩阵对角线上的两个元素分别是两个字段的方差，而其它元素是a和b的协方差。两者被统一到了一个矩阵的，根据矩阵相乘的运算法则，这个结论很容易被推广到一般情况

**设我们有m个n维数据记录，将其按列排成n乘m的矩阵X，设**![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316145552791-346353351.png)**，则C是一个对称矩阵，其对角线分别是各个字段的方差，而第i行j列和j行i列元素相同，表示i和j两个字段的协方差**。



### 4. 协方差矩阵对角化

根据上述推导，我们发现要达到优化目前，等价于将协方差矩阵对角化：即除对角线（方差要尽可能大）外的其它元素化为0（协方差为0），并且在对角线上将元素按大小从上到下排列，这样我们就达到了优化目的

设原始数据矩阵X对应的协方差矩阵为C，而P是一组基按行组成的矩阵，设Y=PX，则Y为X对P做基变换后的数据。设Y的协方差矩阵为D，我们推导一下D与C的关系：

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316150644932-2110707000.png)

现在事情很明白了！我们要找的P不是别的，而是能让原始协方差矩阵对角化的P。换句话说：

**优化目标变成了寻找一个矩阵P，满足![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722152219277-1044647742.png)是一个对角矩阵，并且对角元素按从大到小依次排列，那么P的前K行就是要寻找的基（因为要取尽可能大的方差）**，用P的前K行组成的矩阵乘以X就使得X从N维降到了K维并满足上述优化条件

由上文知道，协方差矩阵C是一个是对称矩阵，在线性代数上，实对称矩阵有一系列非常好的性质

```
1）实对称矩阵不同特征值对应的特征向量必然正交。
2）设特征向量λλ重数为r，则必然存在r个线性无关的特征向量对应于λλ，因此可以将这r个特征向量单位正交化。
```

由上面两条可知，一个n行n列的实对称矩阵一定可以找到n个单位正交特征向量，设这n个特征向量为e1,e2,⋯,en，我们将其按列组成矩阵![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316154642807-1887235172.png)

则对协方差矩阵C有如下结论

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316154708370-758565764.png)

其中ΛΛ为对角矩阵，**其对角元素为各特征向量对应的特征值**。到这里，我们发现我们已经找到了需要的矩阵P：![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170316154826557-377819284.png)

P是协方差矩阵的特征向量单位化后按行排列出的矩阵，其中每一行都是C的一个特征向量。如果设P按照ΛΛ中特征值的从大到小，将特征向量从上到下排列，则用P的前K行组成的矩阵乘以原始数据矩阵X，就得到了我们需要的降维后的数据矩阵Y



## 0x4：PCA优化问题的解 - 和协方差矩阵对角化的关系

我们上一小节讨论了协方差矩阵的最小化问题，其实PCA算法模型的优化策略和基变换降维的核心思想是一样的。都是希望在尽可能不丢失信息的前提下，让维度尽可能地约简。

在PCA中，我们要找的压缩矩阵 W 和复原矩阵 U 使得原始信号和复原信号在平方距离上最小，即，我们需要求解如下问题：

![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722095425922-1046756971.png)，即尽量无损压缩。

令 x1，....，xm是![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722160202240-463517875.png)中的任意向量，![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722160230722-89773577.png)，以及 u1，....，un是 A 中最大的 n 个特征值对应的特征向量。那么，上式PCA优化问题的解为：令 U 的列等于 u1，....，un，以及 ![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180722160441488-2125590477.png)

```
1. 降维后的信息损失尽可能小，尽可能保留原始样本的概率分布
2. 降维后的基之间是完全正交的
```



## 0x5: PCA算法过程



### 1. PCA算法过程公式化描述

总结一下PCA的算法步骤

\1. 设有m条n维数据

\2. 将原始数据按列组成n行m列矩阵X

\3. 将X的每一行(代表一个属性字段，即一个维度)进行零均值化，即减去这一行的均值

\4. 求出协方差矩阵![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317200644666-551936719.png)

\5. 求出协方差矩阵的特征值（矩阵特征值）及对应的特征向量（矩阵特征向量）

\6. 将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P

\7. ![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317200749666-2065182325.png)即为降维到k维后的数据。

**总的来说，PCA降维的目的是让降维后的向量方差最大(最离散)，协方差最小(目标维的各个基之间的相关性最小)**



### **2. 一个例子**

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317201043948-743825884.png)

我们用PCA方法将这组二维数据其降到一维。
因为这个矩阵的每行已经是零均值，这里我们直接求协方差矩阵

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317201157385-153234170.png)

然后求其特征值和特征向量。求解后特征值为![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317201413088-529921131.png)

其对应的特征向量分别是![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317201436073-1072655368.png)

其中对应的特征向量分别是一个通解，c1和c2可取任意实数。那么标准化后的特征向量为![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317201508401-197373466.png)

因此我们的矩阵P是![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317201520838-1454210677.png)

可以验证协方差矩阵C的对角化![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317201551338-1008900679.png)

最后我们用P的第一行乘以数据矩阵，就得到了降维后的表示![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317201654604-1796490767.png)

降维投影结果如下图

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170317201706526-500412799.png)



## 0x6：PCA的限制

PCA也存在一些限制



### 1. 它可以很好的解除线性相关，但是对于高阶相关性就没有办法了

对于存在高阶相关性的数据，可以考虑Kernel PCA，通过Kernel函数将非线性相关转为线性相关



### 2. PCA假设数据各主特征是分布在正交方向上，如果在非正交方向上存在几个方差较大的方向，PCA的效果就大打折扣了



### 3. PCA是一种无参数技术，无法实现个性优化

也就是说面对同样的数据，如果不考虑清洗，谁来做结果都一样，没有主观参数的介入，所以PCA便于通用实现，但是本身无法个性化的优化



## 0x7: 基于原生python+numpy实现PCA算法

先对原始数据零均值化(在图像里表现为白化处理，忽略各个图像不同的亮度)，然后求协方差矩阵，接着对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间

**1. 零均值化**

假如原始数据集为矩阵dataMat，dataMat中每一行代表一个样本，每一列代表同一个特征。零均值化就是求每一列的平均值，然后该列上的所有数都减去这个均值。也就是说，这里零均值化是对每一个特征而言的

```
def zeroMean(dataMat):        
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值  
    newData=dataMat-meanVal  
    return newData,meanVal  
```

用numpy中的mean方法来求均值，axis=0表示按列求均值

**2. 求协方差矩阵**

```
newData,meanVal=zeroMean(dataMat)  
covMat=np.cov(newData,rowvar=0)  
```

numpy中的cov函数用于求协方差矩阵，参数rowvar很重要！若rowvar=0，说明传入的数据一行代表一个样本，若非0，说明传入的数据一列代表一个样本。因为newData每一行代表一个样本，所以将rowvar设置为0

**3. 求特征值、特征矩阵**

调用numpy中的线性代数模块linalg中的eig函数，可以直接由协方差矩阵求得特征值和特征向量

```
eigVals,eigVects=np.linalg.eig(np.mat(covMat))  
```

eigVals存放特征值，行向量。
eigVects存放特征向量，每一列带别一个特征向量。
特征值和特征向量是一一对应的

**4. 保留主要的成分[即保留值比较大的前n个特征]**

第三步得到了特征值向量eigVals，假设里面有m个特征值，我们可以对其排序，排在前面的n个特征值所对应的特征向量就是我们要保留的，它们组成了新的特征空间的一组基n_eigVect。将零均值化后的数据乘以n_eigVect就可以得到降维后的数据

```
eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序  
n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标  
n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量  
lowDDataMat=newData*n_eigVect               #低维特征空间的数据  
reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据  
return lowDDataMat,reconMat 
```

**5. 完整code**

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
# 零均值化
def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataMat - meanVal
    return newData, meanVal


def pca(dataMat, n):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
    return lowDDataMat, reconMat
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

**Relevant Link:**

```
http://www.cnblogs.com/jerrylead/archive/2011/04/18/2020209.html
http://blog.codinglabs.org/articles/pca-tutorial.html
```



## 0x8: 对图像数据应用PCA算法

为使PCA算法能有效工作，通常我们希望所有的特征![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170318225112026-2110079335.png)都有相似的取值范围(并且均值接近于0)。我们有必要单独对每个特征做预处理，即通过估算每个特征的均值和方差，而后将其取值范围规整化为零均值和单位方差。

但是，对于大部分图像类型，我们却不需要进行这样的预处理。在实践中我们发现，大多数特征学习算法对训练图片的确切类型并不敏感，所以大多数用普通照相机拍摄的图片，只要不是特别的模糊或带有非常奇怪的人工痕迹，都可以使用。在自然图像上进行训练时，对每一个像素单独估计均值和方差意义不大，因为(理论上)图像任一部分的统计性质都应该和其它部分相同，图像的这种特性被称作**平稳性(stationarity)。**

具体而言，为使PCA算法正常工作，我们通常需要满足以下要求

```
1. 特征的均值大致为0
2. 不同特征的方差值彼此相似
```

对于自然图片，即使不进行方差归一化操作，条件(2)也自然满足，故而我们不再进行任何方差归一化操作(对音频数据,如声谱,或文本数据,如词袋向量，我们通常也不进行方差归一化)
实际上，PCA算法对输入数据具有缩放不变性，无论输入数据的值被如何放大(或缩小)，返回的特征向量都不改变。更正式的说：如果将每个特征向量x 都乘以某个正数(即所有特征量被放大或缩小相同的倍数)，PCA的输出特征向量都将不会发生变化

既然我们不做方差归一化，唯一还需进行的规整化操作就是均值规整化，其目的是保证所有特征的均值都在0附近。根据应用场景，在大多数情况下，我们并不关注所输入图像的整体明亮程度。比如在对象识别任务中，图像的整体明亮程度并不会影响图像中存在的是什么物体。

更为正式地说，我们对图像块的平均亮度值不感兴趣，所以可以减去这个值来进行均值规整化。

需要注意的是，如果你处理的图像并非自然图像（比如，手写文字，或者白背景正中摆放单独物体），其他规整化操作就值得考虑了，而哪种做法最合适也取决于具体应用场合。但对自然图像而言，对每幅图像进行上述的零均值规整化，是默认而合理的处理



### 1. 利用PCA进行人脸识别

接下来我们尝试对一个图像进行PCA处理，这里我们对一张图像进行PCA降维处理，进而基于降维后的低维度像素图进行人脸相似度检测。

大致思路是，收集一个基准样本集(标准人像)，然后通过PCA降维提高运算效率，之后的测试过程就是拿待测试样本图像和基准样本集中的所有图片依次计算"欧式距离"，最后的判定结果以离基准样本集欧式距离最近的那张图像为"人脸"

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170320131741533-1191644791.png)

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linA # 为了激活线性代数库mkl
from PIL import Image
from resizeimage import resizeimage
import os,glob

imageWidth = 230
imageHigth = 300
imageSize = imageWidth * imageHigth

def sim_distance(train,test):
    '''
    计算欧氏距离相似度
    :param train: 二维训练集
    :param test: 一维测试集
    :return: 该测试集到每一个训练集的欧氏距离
    '''
    return [np.linalg.norm(i - test) for i in train]


def resizeImage(filepath):
    img = Image.open(filepath)
    img = img.resize((imageWidth, imageHigth), Image.BILINEAR)
    img.save(filepath)


def resizeImages():
    picture_path = os.getcwd() + '/images/'
    for name in glob.glob(picture_path + '*.jpeg'):
        print name
        resizeImage(name)


def calcVector(arr1, arr2):
    distance1, distance2 = 0, 0
    for i in arr1:
        distance1 += i * i
    distance1 = distance1 / len(arr1)
    for i in arr2:
        distance2 += i * i
    distance2 = distance2 / len(arr2)

    return distance1 < distance2



def main():
    picture_path = os.getcwd() + '/images/'
    print "picture_path: ", picture_path
    array_list = []
    for name in glob.glob(picture_path + '*.jpeg'):
        print name
        # 读取每张图片并生成灰度（0-255）的一维序列 1*120000
        img = Image.open(name)
        # img_binary = img.convert('1') 二值化
        img_grey = img.convert('L')  # 灰度化
        array_list.append(np.array(img_grey).reshape((1, imageSize)))  # 拉长为1维

    mat = np.vstack((array_list))  # 将上述多个一维序列(每个序列代表一张图片)合并成矩阵 3*69000
    P = np.dot(mat, mat.transpose())  # 计算P
    v, d = np.linalg.eig(P)  # 计算P的特征值和特征向量
    print 'P Eigenvalues'
    print v
    print "Feature vector"
    print d

    d = np.dot(mat.transpose(), d)  # 计算Sigma的特征向量 69000 * 3
    train = np.dot(d.transpose(), mat.transpose())  # 计算训练集的主成分值 3*3
    print '训练集pca降维后的向量数组'
    print train

    # 开始测试
    # 用于测试的图片也需要resize为和训练基准样本集相同的size
    resizeImage('images/test_1.jpg')
    test_pic = np.array(Image.open('images/test_1.jpg').convert('L')).reshape((1, imageSize))
    # 计算测试集到每一个训练集的欧氏距离
    result1 = sim_distance(train.transpose(), np.dot(test_pic, d))
    print 'test_1.jpg 降维后的向量'
    print result1

    resizeImage('images/test_2.jpg')
    test_pic = np.array(Image.open('images/test_2.jpg').convert('L')).reshape((1, imageSize))
    result2 = sim_distance(train.transpose(), np.dot(test_pic, d))
    print 'test_2.jpg 降维后的向量'
    print result2

    # 欧式距离最小的即为最接近训练样本集的测试样本
    if calcVector(result1, result2):
        print 'test_1.jpg is a human'
    else:
        print 'test_2.jpg is a human'


if __name__ == '__main__':
    resizeImages()
    main()
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

训练集的计算结果为

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python /Users/zhenghan/PycharmProjects/littlehann/just4fun.py
/Users/zhenghan/PycharmProjects/littlehann/images/train_2.jpeg
/Users/zhenghan/PycharmProjects/littlehann/images/train_3.jpeg
/Users/zhenghan/PycharmProjects/littlehann/images/train_1.jpeg
picture_path:  /Users/zhenghan/PycharmProjects/littlehann/images/
/Users/zhenghan/PycharmProjects/littlehann/images/train_2.jpeg
/Users/zhenghan/PycharmProjects/littlehann/images/train_3.jpeg
/Users/zhenghan/PycharmProjects/littlehann/images/train_1.jpeg
P Eigenvalues
[ 444.76007266 -199.2827456    -8.47732705]
Feature vector
[[-0.557454   -0.7252759   0.40400484]
 [-0.69022539  0.1344664  -0.71099065]
 [-0.46133931  0.67519898  0.57556266]]
pca
[[ -2.94130809e+09  -2.81400683e+09  -2.27967171e+09]
 [ -4.53920521e+08   2.41231868e+07   4.49796574e+07]
 [  5.06334430e+08   1.43429000e+08   2.56660545e+08]]
test_1.jpg
[859150941.34167683, 507130780.35877681, 98296821.771007225]
test_2.jpg
[921097812.32432926, 784122768.95719075, 323861431.46721846]
test_1.jpg is a human

Process finished with exit code 0
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

利用如下图片进行识别测试，首先右乘得到各自在三个主轴上的值(对测试样本也同样进行PCA化)，然后计算出该图片到训练样本中的三张图片的欧式距离

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170320131804736-361100242.png)

```
test_1.jpg
[859150941.34167683, 507130780.35877681, 98296821.771007225]
test_2.jpg
[921097812.32432926, 784122768.95719075, 323861431.46721846]
test_1.jpg is a human
```

再用别的测试集类来测试

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170320133814471-1824437127.png)

上述的代码中我们自己实现了PCA的代码，实际上这个逻辑可以用sklearn来完成

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)  

1. n_components: PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
    1) 缺省时默认为None，所有成分被保留 
    2) 赋值为int，比如n_components=1，将把原始数据降到一个维度 
    3) 赋值为string，比如n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。
2. copy: 表示是否在运行算法时，将原始训练数据复制一份。若为True，则运行PCA算法后，原始训练数据的值不会有任何改变，因为是在原始数据的副本上进行运算；若为False，则运行PCA算法后，原始训练数据的值会改，因为是在原始数据上进行降维计算
    1) 缺省时默认为True 
3. whiten: 白化，使得每个特征具有相同的方差(即去均值化)
    1) 缺省时默认为False 
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170320141810252-1051901952.png)

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA

data = [[ 1.  ,  1.  ],
       [ 0.9 ,  0.95],
       [ 1.01,  1.03],
       [ 2.  ,  2.  ],
       [ 2.03,  2.06],
       [ 1.98,  1.89],
       [ 3.  ,  3.  ],
       [ 3.03,  3.05],
       [ 2.89,  3.1 ],
       [ 4.  ,  4.  ],
       [ 4.06,  4.02],
       [ 3.97,  4.01]]

if __name__ == '__main__':
    pca = PCA(n_components=1)
    newData = pca.fit_transform(data)
    print newData
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)



## 0x9: 选择主成分个数

使用PCA降维技术进行了一个简单的小实验之后，我们来继续思考一个更深入的问题，应用PCA的时候，对于一个1000维的数据，我们怎么知道要降到几维的数据才是合理的？即n要取多少，才能保留最多信息同时去除最多的噪声？一般，我们是通过方差百分比来确定n的

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170320134708799-526127391.png)

保留的方差越大，对应于特征向量的离散程度就越大，就越容易被分类器进行有效分类

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
def percentage2n(eigVals,percentage):  
    sortArray=np.sort(eigVals)   #升序  
    sortArray=sortArray[-1::-1]  #逆转，即降序  
    arraySum=sum(sortArray)  
    tmpSum=0  
    num=0  
    for i in sortArray:  
        tmpSum+=i  
        num+=1  
        if tmpSum>=arraySum*percentage:  
            return num  

def pca(dataMat,percentage=0.99):  
    newData,meanVal=zeroMean(dataMat)  
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本  
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量  
    n=percentage2n(eigVals,percentage)                 #要达到percent的方差百分比，需要前n个特征向量  
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序  
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标  
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量  
    lowDDataMat=newData*n_eigVect               #低维特征空间的数据  
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据  
    return lowDDataMat,reconMat  
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

**Relevant Link:**

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
http://ufldl.stanford.edu/wiki/index.php/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90
http://blog.csdn.net/watkinsong/article/details/823476
http://www.cnblogs.com/theskulls/p/4925147.html
http://ufldl.stanford.edu/wiki/index.php/%E5%AE%9E%E7%8E%B0%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90%E5%92%8C%E7%99%BD%E5%8C%96
http://book.2cto.com/201406/43853.html
http://pythoncentral.io/resize-image-python-batch/
https://opensource.com/life/15/2/resize-images-python
https://pypi.python.org/pypi/python-resize-image
https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.ndarray.resize.html
http://www.ctolib.com/topics-58310.html
http://www.cnblogs.com/chenbjin/p/4200790.html
http://blog.csdn.net/u012162613/article/details/42177327
http://blog.csdn.net/u012162613/article/details/42192293
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

 

[回到顶部(go to top)](https://www.cnblogs.com/littlehann/p/6558575.html#_labelTop)

# **3. Random Projection（随机投影）**



## **0x1：为什么需要随机投影**

主成分分析将数据线性转换到低维空间，但代价昂贵。为了找出这个转换，需要计算协方差矩阵，花费的时间将是数据维数的立方。这对于属性数目庞大的数据集是不可行的。

一个更为简便的替代方法是将数据随机投影到一个维数预先设定好的子空间，也即找到一个所谓的随机投影矩阵。

那么问题来了，找到随机投影矩阵是很容易，但效果是否好呢？



## 0x2: Johnson–Lindenstrauss lemma，随机投影有效性的理论依据

随机投影的理论依据是J-L Lemma，公式的核心思想总结一句话就是：

**在高维欧氏空间里的点集映射到低维空间里相对距离，可以在一定的误差范围内，得到保持**

至于为什么要保持，主要是很多机器学习算法都是在以利用点与点之间的距离信息（欧氏距仅是明氏距的特例），及相对位序展开计算分析的。

也就是说，很多的机器学习算法都作了一个假设：**点集之间的距离，包含了数据集蕴含的概率分布**。



### 1. 问题定义 

首先, JL要解决的问题非常简单(只是陈述比较简单而已), 在一个高维的欧式空间(距离用欧式距离表示) ![\mathbf{R}^d](http://tcs.nju.edu.cn/wiki/images/math/d/e/5/de53a132a02a8c3cf22cc838d973aac6.png). 我们想要把这些点转换到一个低维的空间![\mathbf{R}^k](http://tcs.nju.edu.cn/wiki/images/math/c/a/0/ca0dc8a07b8cc7f0739b253b4eab83ee.png), 当时要保证空间转换后,没两两个点之间的距离几乎不变.

正规点说就是, 找到一个映射关系:![f:\mathbf{R}^d\rightarrow\mathbf{R}^k](http://tcs.nju.edu.cn/wiki/images/math/9/4/8/9487e98b84877dfad00e6c93fe756879.png),里面任意两个点u,v,使得![\|f(u)-f(v)\|](http://tcs.nju.edu.cn/wiki/images/math/2/0/2/202be4300898d7a1b769fa797c9813a6.png)和![\|u-v\|](http://tcs.nju.edu.cn/wiki/images/math/c/4/2/c421e1e2cf8cd7d0fc24b3cc01f381f3.png)只有一点点的不同,其中![\|u-v\|=\sqrt{(u_1-v_1)^2+(u_2-v_2)^2+\ldots+(u_d-v_d)^2}](http://tcs.nju.edu.cn/wiki/images/math/c/4/8/c48e7b68cad91b41465ba9cbc596c9fb.png) ,![\|u-v\|](http://tcs.nju.edu.cn/wiki/images/math/c/4/2/c421e1e2cf8cd7d0fc24b3cc01f381f3.png)是两点的欧式距离.

令 x1，x2 为![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180723203240078-436873726.png)上的两个向量，如果![img](https://images2018.cnblogs.com/blog/532548/201807/532548-20180723203313978-251441339.png)接近1，则矩阵 W 没有扭曲 x1，x2 之间的距离太多，或具有保距特性。

**随机投影不会扭曲欧式距离太多**。



### 2. 问题证明

JL理论证明了解决这个问题的可能性，即这种映射理论上是存在的。

对于任意一个样本大小为m的集合，如果我们通过随机投影将其维度降到一个合适的范围内，那么我们将以较高的概率保证投影后的数据点之间的距离信息变化不大。

这样我们在做K-mean之类的算法时，就可以先将高维度的数据利用随机投影进行降维处理，然后在执行算法，且不会影响算法的最终聚类效果太多。



## 0x3: Random projection算法

随机投影技术是一个理论框架，类似马尔科夫性质理论一样，具体实现随机投影的算法有很多



### 1. Gaussian random projection - 高斯随机投影 

要讨论高斯随机投影为什么有效，我们需要先来讨论下一个核心问题，高斯投影是否可以从理论上保证投影降维前后，数据点的空间分布距离基本保持不变呢？这个问题其实可以等价于证明另一个问题，即高斯投影能否做到将高维空间的点均匀的投影到低维空间中，如果能做到这一点，那么我们也可以证明其具备“投影降维前后，数据点的空间分布距离基本保持不变”的能力。

#### 1）考虑在二维情况下如何获得均匀采样？

首先，考虑二维的情况，即如何在球形的周长上采样。我们考虑如下方法

```
1. 先在一个包含该圆形的外接正方形内均匀的采样
2. 第二，将采样到的点投影到圆形上
# 具体地说就是
1. 先独立均匀的从区间[−1,1](我们假设圆形跟正方形的中心点都在原点)内产生两个值组成一个二维的点(x1,x2)
2. 将该二维点投影到圆形上
```

例如，如下图所示：

如果我们产生点是图中的A,B两点，那么投影到圆形上就是C点；

如果产生的是点D，那么投影到圆形上就是E点。

但是，用这样的方法得到点在圆形上并不是均匀分布的，比如产生C点的概率将大于产生E点概率，因为可以投影到C点对应的那条直线比E点对应的那条直线要长。

解决的办法是去掉圆形外面的点，也就是如果我们首先产生的点在圆形外的话(比如点B)，那么我们就丢弃该点，重新在产生，这样的话产生的点在圆形上是均匀分布的，从这里可以看出，**降维的一个核心概念都是投影，投影是一种数据降维的通用方法。**

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170321125327815-1454018188.png)

那么，我们能否将此方法扩展到高维的情况下呢？答案是不行的。

因为在高维的情况下球与正方体的体积比将非常非常小，几乎接近于零。也就是我们在正方体内产生的点几乎不可能落到球体内部，那么也就无法产生有效的投射点。那么，在高维的球体上，我们应该怎样才能产生一个均匀分布与球体表面的点呢？答案是利用高斯分布

#### 2）利用高斯分布**在高维球体表面产生均匀分布点的方法**

即将上述第一步改成：以均值为零方差为1的高斯分布独立地产生d个值，形成一个d维的点![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170321145019611-927076109.png)；然后第二步：将点x归一化为![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170321145101721-1131463498.png)用这种方法产生点必定均匀分布在高维球体表面。原因如下：

d个独立的高斯分布的密度函数为![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170321145147002-926561760.png)那么![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170321145216893-798646096.png)，为常数，**说明高斯分布产生的每个投射点的概率都一样，即均匀分布**。

#### 3）**高维空间下的高斯分布性质**

高斯分布是概率统计里面最常见也是最有用的分布之一，即使在高维情况下的也有一些非常有用的特：

**对于低维高斯分布来说，其概率质量主要集中在均值附近。在高维情况下，这样的结论也同样成立**

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import random_projection 

if __name__ == '__main__':
    X = np.random.rand(100, 10000)
    transformer = random_projection.GaussianRandomProjection()
    X_new = transformer.fit_transform(X)
    print X_new.shape

output:
(100, 3947)
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

**Relevant Link:**

```
http://blog.csdn.net/ljj583905183/article/details/47980169
http://cjbme.csbme.org/CN/abstract/abstract500.shtm
```



### 2. a uniform random k-dimensional subspace.

映射关系 ![f:\mathbf{R}^d\rightarrow\mathbf{R}^k](http://tcs.nju.edu.cn/wiki/images/math/9/4/8/9487e98b84877dfad00e6c93fe756879.png)是可以随机构造的, 以下这种是JL在论文中用到的一种:



 构造后的点![\sqrt{\frac{d}{k}}Av](http://tcs.nju.edu.cn/wiki/images/math/b/6/9/b6991601a185a5f65893e22c711bd602.png) 是 ![\mathbf{R}^k](http://tcs.nju.edu.cn/wiki/images/math/c/a/0/ca0dc8a07b8cc7f0739b253b4eab83ee.png)的其中一个向量.

参数 ![\sqrt{\frac{d}{k}}](http://tcs.nju.edu.cn/wiki/images/math/f/d/8/fd83b639e550b42bb4a5621f646fc7cf.png) 是为了保证![\mathbf{E}\left[\left\|\sqrt{\frac{d}{k}}Av\right\|^2\right]=\|v\|^2](http://tcs.nju.edu.cn/wiki/images/math/3/f/8/3f82f4ea4e2c854b3d7c73f90b5a1191.png).



### 3. **Sparse random projection**

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import random_projection

if __name__ == '__main__':
    X = np.random.rand(100, 10000)
    transformer = random_projection.SparseRandomProjection()
    X_new = transformer.fit_transform(X)
    print X_new.shape

output:
(100, 3947)
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)



### 4. More computationally efficient random projections

the Gaussian distribution can be replaced by a much simpler distribution such as

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170321101244518-196792203.png)



## 0x4: Random projection在手写图像识别里的具体应用 



### 1. 计算100%完全分类的baseline基线值

我们使用scikit-learn的手写数字图像数据集，其中包含了1,797个图像样本（0-9数字），每个图像是一个 8*8（总共64维）的像素图，我们将其展开为64bit的行向量。

我们使用random projection来训练这批样本，并且计算出一个基线值，即需要多少dimensions维度组成部分才能100%完全地对这批样本进行分类。

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
# -*- coding: utf-8 -*-

from sklearn.random_projection import johnson_lindenstrauss_min_dim

if __name__ == '__main__':
    print johnson_lindenstrauss_min_dim(1797,eps=0.1)

Out[2]:
6423
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

实验运行的结果是：对于1797个64维的样本集来说，在0.1欧几里德空间距离的离散容忍度（即至少样本间距离要大于0.1欧几里得距离）的前提下，需要6423个rp维度组成部分来进行有效分类。

同时，考察另一个参数，即离散容忍度：

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
>>> from sklearn.random_projection import johnson_lindenstrauss_min_dim
>>> johnson_lindenstrauss_min_dim(n_samples=1e6, eps=0.5)
663
>>> johnson_lindenstrauss_min_dim(n_samples=1e6, eps=[0.5, 0.1, 0.01])
array([    663,   11841, 1112658])
>>> johnson_lindenstrauss_min_dim(n_samples=[1e4, 1e5, 1e6], eps=0.1)
array([ 7894,  9868, 11841])
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170321111809518-1832530372.png)

可以看到，对样本集的离散容忍度越小（可分性要求越高），需要的维度components就指数增大，显然这个理论值在工程化中不可接受的。在实际的应用场景中，往往需要我们去寻找一个折中值。



### **2. random projection的components个数是否和欧几里德分类准确度是正相关的**

scikit_learn给出的baseline是6423，这显然太大了，我们的目的是降维，是为了提高运算效率的，为此，我们选取的components一定是小于64(原始8*8的图像维度)的。

但是，那么如何判断降维的程度呢，即至少需要多少components才能基本不影响原始样本集的分类准确度呢，同时又尽可能地实现降维目录呢？我们遍历以下的components，逐个看它们的分类准确度表现

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
# -*- coding: utf-8 -*-

from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    #  initializes our lists of accuracies. We’ll be evaluating 20 different component sizes, equally spaced from 2 to 64.
    accuracies = []
    components = np.int32(np.linspace(2, 64, 20))
    print "components"
    print components
    '''
    components
    [ 2  5  8 11 15 18 21 24 28 31 34 37 41 44 47 50 54 57 60 64]
    '''

    # load the digits data set, perform a training and testing split, train a Linear SVM, and obtain a baseline accuracy:
    digits = datasets.load_digits()
    split = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)
    (trainData, testData, trainTarget, testTarget) = split

    model = LinearSVC()
    model.fit(trainData, trainTarget)
    baseline = metrics.accuracy_score(model.predict(testData), testTarget)

    # loop over the projection sizes
    '''
    1. We start looping over our number of components, equally spaced, in the range 2 to 64.
    2. Then we instantiate our SparseRandomProjection using the current number of components, fit our random projection to the data (which essentially means generating a sparse matrix of values), and then transforming our original training data by projecting the data.
    3. Now that we have obtained a sparse representation of the data, let’s train a Linear SVM on it.
    4. Our Linear SVM is now trained; it’s time to see how it performs on the testing data and update our list of accuracies.
    '''
    for comp in components:
        # create the random projection
        sp = SparseRandomProjection(n_components=comp)
        X = sp.fit_transform(trainData)

        # train a classifier on the sparse random projection
        model = LinearSVC()
        model.fit(X, trainTarget)

        # evaluate the model and update the list of accuracies
        test = sp.transform(testData)
        accuracies.append(metrics.accuracy_score(model.predict(test), testTarget))

    '''
    At this point all the hard work is done.
    We’ve evaluated a series of sparse random projections for varying numbers of components.
    Let’s plot our results and see what we have:
    '''
    # create the figure
    plt.figure()
    plt.suptitle("Accuracy of Sparse Projection on Digits")
    plt.xlabel("# of Components")
    plt.ylabel("Accuracy")
    plt.xlim([2, 64])
    plt.ylim([0, 1.0])

    # plot the baseline and random projection accuracies
    plt.plot(components, [baseline] * len(accuracies), color="r")
    plt.plot(components, accuracies)

    plt.show()
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

![img](https://images2015.cnblogs.com/blog/532548/201703/532548-20170321110301080-1661171591.png)
从结果上看，30维的稀疏降维就可以基本达到较高的分类准确度，同时又尽可能地实现了降维的目的。

**Relevant Link:**

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
http://forum.ai100.com.cn/blog/thread/py-2015-02-19-3811867870631018/
https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
https://en.wikipedia.org/wiki/Random_projection
http://blog.yhat.com/posts/sparse-random-projections.html
http://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html 
http://scikit-learn.org/stable/modules/random_projection.htmlhttps://blog.csdn.net/luoyun614/article/details/39853259
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

 

[回到顶部(go to top)](https://www.cnblogs.com/littlehann/p/6558575.html#_labelTop)

# 4. 压缩感知

```
https://www.cnblogs.com/Eufisky/p/7798767.html#commentform
https://blog.csdn.net/yq_forever/article/details/55271952
https://www.cnblogs.com/AndyJee/p/4988623.html
```

 

[回到顶部(go to top)](https://www.cnblogs.com/littlehann/p/6558575.html#_labelTop)

# 5. 主成分回归（principle component regression，PCR）

正如文章前面讨论的，主成分分析经常用于应用学习算法之前的预处理步骤。当学习算法是线性回归时，由此产生的模型称为**主成分回归（principal component regression）**。

主成份回归可以**解决变量间共线性的问题**。它使用从数据抽提出的主成份进行回归，一般来说是选择前面的几个主成份。 



## 0x1：主成份的组成

PCA的主成分是通过选择原始属性在新的正交基上的投影向量的前k个来实现，和主成分分析PCA不同的是，**PCR的主成分是基于****原始属性的线性组合**。

通俗的理解就是可以将PCR理解为一个2层的线性回归模型，第一层的线性回归的输出结果作为第二层的输入属性，在第二层，根据各个输入属性的相关度权重选择k个属性，进行最终的线性回归。

实际上，如果所有的属性分量都被使用，而不是”主要的“部分，其结果与在原始输入数据上应用最小二乘回归的结果是一样的。使用少部分而不是全部属性分量集合的结果是一个削弱了的回归。

**Relevant Link:**

```
https://www.jianshu.com/p/d090721cf501?from=timeline
```

 

[回到顶部(go to top)](https://www.cnblogs.com/littlehann/p/6558575.html#_labelTop)

# 5. 偏最小二乘回归（partial least-squares regression）

偏最小二乘不同于主成分分析的是，在构建坐标系统时，和预测属性一样，它考虑类属性。其思想是计算派生的方向，这些方向和有高方差一样，是和类有强关联的。这在为有监督学习寻找一个尽可能小的转换属性集时将很有益处。



## 0x1：偏最小二乘计算方法

有一种迭代方法用于计算偏最小二乘方向，且仅仅涉及点积运算。

- **量纲归一化**：从输入属性开始，所有属性被标准化为拥有零均值和单位方差
- **初始偏差计算**：用于第一个偏最小二乘方向的属性系数是通过每一个属性向量和类向量之间以此进行点积运算得到的，点积结果代表方向上的偏差。
- **残差驱动的启发式属性选择**：用同样的方法找到第二个方向，但是，此时的原始属性要被替换为，该属性原始值与用上一轮迭代选定的单变量属性回归所得的预测值之间的差值，这个单变量属性回归使用的是上一轮选定的属性（上一轮的残差）作为属性预测的单一预测因子。这些差值被称为残差（redidual）。每次的残差（新属性）都代表针对当残差的一个方向上的修正，然后再通过点积度量依然存在的方向上的偏差
- 用同样的方式继续运行此流程以得到其余的方向，用前一次迭代所得的残差作为属性形成输入来找到当前偏最小二乘的方向。
- **PLSR结束**：整个流程结束后得到的所有属性，就是不断修正方向以靠近目标类方向的属性集合

用一个例子来说明这个过程：

![img](https://img2018.cnblogs.com/blog/532548/201911/532548-20191128113848595-1808317817.png)

CPU性能数据中的前5个实例

任务是：**要依据其他两种属性找到一种新的表达方式，用于表示目标属性PRP**。

- 第一个偏最小二乘方向的属性系数是通过在属性和类属性之间依次进行点积运算得到的，表a）列出了原始属性值
  - prp和chmin之间的点积是-0.4472
  - prp和chmax之间的点积是22.981
- 因此，第一个偏最小二乘方向为：pls1 = -0.4472*chmin + 22.981*chmax，表b）列出了第一个偏最小二乘方向
- 接下来的步骤是准备输入数据，用以找到第二个偏最小二乘方向。为此，pls1依次回归到chmin和chmax，由pls1得到线性方程，用以单独预测这些属性中的每一个属性。这些系数通过计算pls1与待求解属性之间的点积得到，且用pls1与它自身的点积来划分所得的结果。由此产生的单变量（一元）回归方程为：
  - chmin = 0.0483 * pls1
  - chmax = 0.0444 * pls1
- 表c）列出的是准备用于寻找第二个偏最小二乘方向的CPU数据。chmin和chmax的原始值被残差所替代。残差是指原始值与之前给出的相应的单变量（一元）回归方程的输入之间的差值（目标值prp仍然一样，只是属性系数变了）。整个过程重复地使用这些数据作为输入产生第二个偏最小二乘方向，即：
  - pls2 = -23.6002*chmin + -0.4593*chmax
- 在最后的偏最小二乘方向确定之后，属性的残差都为0。这反应了一个事实，正如主成分分析一样，所有方向的全集担负了原始数据的所有方差

上面的过程理解起来可能有些抽象，可以将其和神经网络的神经元学习过程进行类比：

- 将每个原始属性都看做是一个独立的神经元
- 每个神经元都是自己的独立的系数
- 所有神经元共同决定了最终的输出，同样，最终预测结果与目标值的差距，这个“错误责任”也要分担给所有的神经元
- 每个神经元都根据自己系数情况，根据一元回归的方式来度量自己的“偏离度”，并根据这个偏离度来进行对应的修改
- 经过不断地迭代，最终所有神经元都彼此协调，整体得到了一个适配目标值的局部最优结果

当把偏最小二乘方向作为输入用于线性回归时，结果模型称为**偏最小二乘回归模型（partial least-squares regression model）**。和主成分回归一样，若使用所有的方向，其结果与在原始数据上应用线性回归所得的结果是一样的。

**Relevant Link:** 

```
https://blog.csdn.net/qq_19600291/article/details/83823994
https://www.cnblogs.com/duye/p/9031511.html
```

 

[回到顶部(go to top)](https://www.cnblogs.com/littlehann/p/6558575.html#_labelTop)

# **7. 降维在其他领域的应用**



## 0x1: PHP SSDEEP模糊化HASH

看完了Random Projection的理论，突然想到以前用过的PHP里面的SSDEEP模糊化HASH本质上也是一种数据降维方法，我们来看看SSDEEP的概念定义

```
模糊化HASH也叫"局部不敏感HASH"，如果用于对比的两个数据集之间的区别只在一个很小的区域内，则模糊化HASH会忽略这种差别，的趋向于得到一个相等的Fuzzy HASH值
```

**LSH**算法的基本思想是利用一个hash函数把集合中的元素映射成hash值，使得相似度越高的元素hash值相等的概率也越高。**LSH**算法使用的关键是针对某一种相似度计算方法，找到一个具有以上描述特性的hash函数。**LSH**所要求的hash函数的准确数学定义比较复杂，以下给出一种通俗的定义方式：

```
对于集合S，集合内元素间相似度的计算公式为sim(*,*)。如果存在一个hash函数h(*)满足以下条件：存在一个相似度s到概率p的单调递增映射关系，使得S中的任意两个满足sim(a,b)>=s的元素a和b，h(a)=h(b)的概率大于等于p。那么h(*)就是该集合的一个LSH算法hash函数
```

一般来说在最近邻搜索中，元素间的关系可以用相似度或者距离来衡量。如果用距离来衡量，那么距离一般与相似度之间存在单调递减的关系。以上描述如果使用距离来替代相似度需要在单调关系上做适当修改

**Relevant Link:**

```
http://www.cnblogs.com/GarfieldEr007/p/5479401.html
```

 https://www.cnzz.com/stat/website.php?web_id=1000401968&method=online)