## Scipy
### 简介
SciPy 是一个用于数学、科学和工程的开源库，其集成了统计、优化、线性代数、傅立叶变换、信号和图像处理，ODE 求解器等模块，是使用 Python 进行科学计算的重要工具之一
```python
import scipy
scipy.__version__
```
查看版本

### 常量模块
为了方便科学计算，SciPy 提供了一个叫` scipy.constants `模块，该模块下包含了常用的物理和数学常数及单位。你可以通过前面给出的链接来查看这些常数和单位

例如，数学中的圆周率和黄金分割常数
```python
from scipy import constants
constants.pi

constants.golden
```
物理学经常会用到的真空中的光速、普朗克系数等
```python
constants.c, constants.speed_of_light

constants.h, constants.Planck
```
### 线性代数
线性代数应该是科学计算中最常涉及到的计算方法之一，
SciPy 中提供的详细而全面的线性代数计算函数。
这些函数基本都放置在模块 `scipy.linalg` 下方。其中，又大致分为：
+ 基本求解方法
+ 特征值问题
+ 矩阵分解
+ 矩阵函数
+ 矩阵方程求解
+ 特殊矩阵构造等几个小类

例： 
+ scipy.linalg.inv： 求给定矩阵的逆
```python
import numpy as np
from scipy import linalg

linalg.inv(np.matrix([[1, 2], [3, 4]]))
```
+ scipy.linalg.svd: 奇异值分解
```python
U, s, Vh = linalg.svd(np.random.randn(5, 4))
U, s, Vh

(array([[-0.13476522, -0.41944012, -0.12939582,  0.86778084,  0.19005644],
        [ 0.09402103, -0.5720548 ,  0.53978602, -0.29806686,  0.53263548],
        [ 0.83400225, -0.35443886, -0.09784915,  0.0334136 , -0.41002734],
        [-0.33732217, -0.54739487, -0.65042082, -0.39427473, -0.08984878],
        [ 0.40453972,  0.26749814, -0.50918177, -0.03928762,  0.70991744]]),
 array([2.98624503, 2.23788305, 1.11504619, 0.70143008]),
 array([[-0.58977443, -0.66726298,  0.01640267,  0.45459563],
        [-0.61013586,  0.43969283,  0.6370153 , -0.16916251],
        [ 0.05177474,  0.52678209, -0.08998367,  0.84363674],
        [ 0.5265166 , -0.28970841,  0.76540538,  0.23022578]]))
```
最终返回酉矩阵 U 和 Vh，以及奇异值 s
+ scipy.linalg.lstsq: 最小二乘法求解函数

### 插值函数
插值，是数值分析领域中通过已知的、离散的数据点，在范围内推求新数据点的过程或方法。
SciPy 提供的 `scipy.interpolate` 模块下方就包含了大量的数学插值方法，涵盖非常全面

过程
+ 先给出一组x, y的值
```python
from matplotlib import pyplot as plt
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])

plt.scatter(x, y)
```
+ 接下来，我们想要在上方两个点与点之间再插入一个值。怎样才能最好地反映数据的分布趋势呢？这时，就可以用到线性插值的方法
```python
from scipy import interpolate

xx = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])  # 两点之间的点的 x 坐标
f = interpolate.interp1d(x, y)  # 使用原样本点建立插值函数
yy = f(xx)  # 映射到新样本点

plt.scatter(x, y)
plt.scatter(xx, yy, marker='*')
```
### 图像处理
有趣的是，SciPy 集成了大量针对图像处理的函数和方法。当然，一张彩色图片是由 RGB 通道组成，而这实际上就是一个多维数组。所以，SciPy 针对图像的处理的模块 `scipy.ndimage`，实际上也是针对多维数组的处理过程，你可以完成卷积、滤波，转换等一系列操作

在正式了解 scipy.ndimage 模块之前，我们先使用 scipy.misc 模块中的  face 方法导入一张浣熊的示例图片。scipy.misc 是一个杂项模块，包含了一些无法被准确归类的方法
```python
from scipy import misc

face = misc.face()
face
```
`face` 默认是图片的 RGB 数组，我们可以对其进行可视化还原
```python
plt.imshow(face)
```
是一张浣熊的图片

接下来，我们尝试` scipy.ndimage `中的一些图像处理方法。例如，对图片进行高斯模糊处理
```python
from scipy import ndimage
plt.imshow(ndimage.gaussian_filter(face, sigma = 5))
```
针对图像进行旋转变换
```python
plt.imshow(ndimage.rotate(face, 45))
```
或者对图像执行卷积操作，首先随机定义一个卷积核 k，然后再执行卷积
```python
k = np.random.randn(2, 2, 3)
plt.imshow(ndimage.convolve(face, k))
```
### 优化方法
最优化，是应用数学的一个分支，一般我们需要最小化（或者最大化）一个目标函数，而找到的可行解也被称为最优解。最优化的思想在工程应用领域非常常见。举例来讲，机器学习算法都会有一个目标函数，我们也称之为损失函数。而找到损失函数最小值的过程也就是最优化的过程。我们可能会用到最小二乘法，梯度下降法，牛顿法等最优化方法来完成。

Markdown Code      
SciPy 提供的 scipy.optimize 模块下包含大量写好的最优化方法。例如上面我们用过的 scipy.linalg.lstsq 最小二乘法函数在 scipy.optimize 模块下也有一个很相似的函数 scipy.optimize.least_squares。这个函数可以解决非线性的最小二乘法问题
### 信号处理
 信号处理（英语：Signal processing）是指对信号表示、变换、运算等进行处理的过程，其在计算机科学、药物分析、电子学等学科中应用广泛。几十年来，信号处理在诸如语音与数据通信、生物医学工程、声学、声呐、雷达、地震、石油勘探、仪器仪表、机器人、日用电子产品以及其它很多的这样一些广泛的领域内起着关键的作用。

SciPy 中关于信号处理的相关方法在 scipy.signal 模块中，其又被划分为：卷积，B-样条，滤波，窗口函数，峰值发现，光谱分析等 13 个小类，共计百余种不同的函数和方法。所以说，信号处理是 SciPy 中十分重要的模块之一

### 统计函数
统计理论应用广泛，尤其是和计算机科学等领域形成的交叉学科，为数据分析、机器学习等提供了强大的理论支撑。SciPy 自然少不了针对统计分析的相关函数，集中在 scipy.stats 模块中。

scipy.stats 模块包含大量概率分布函数，主要有连续分布、离散分布以及多变量分布。除此之外还有摘要统计、频率统计、转换和测试等多个小分类。基本涵盖了统计应用的方方面面
+ norm.rvs 返回随机变量
+ norm.pdf 返回概率密度函数
+ norm.cdf 返回累计分布函数
+ norm.sf 返回残存函数
+ norm.ppf 返回分位点函数
+ norm.isf 返回逆残存函数
+ norm.stats 返回均值，方差，（费舍尔）偏态，（费舍尔）峰度
+ 以及 norm.moment 返回分布的非中心矩





