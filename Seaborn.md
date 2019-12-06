## Seaborn 
### 简介
Seaborn 是以 Matplotlib 为核心的高阶绘图库，无需经过复杂的自定义即可绘制出更加漂亮的图形，非常适合用于数据可视化探索
Seaborn 具有如下特点：

+ 内置数个经过优化的样式效果。
+ 增加调色板工具，可以很方便地为数据搭配颜色。
+ 单变量和双变量分布绘图更为简单，可用于对数据子集相互比较。
+ 对独立变量和相关变量进行回归拟合和可视化更加便捷。
+ 对数据矩阵进行可视化，并使用聚类算法进行分析。
+ 基于时间序列的绘制和统计功能，更加灵活的不确定度估计。
+ 基于网格绘制出更加复杂的图像集合

### 快速优化图形
当我们使用 Matplotlib 绘图时，默认的图像样式算不上美观。此时，就可以使用 Seaborn 完成快速优化。下面，我们先使用 Matplotlib 绘制一张简单的图像
```python
import matplotlib.pyplot as plt
%matplotlib inline

x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
y_bar = [3, 4, 6, 8, 9, 10, 9, 11, 7, 8]
y_line = [2, 3, 5, 7, 8, 9, 8, 10, 6, 7]

plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='y')
```
使用 Seaborn 完成图像快速优化的方法非常简单。只需要将 Seaborn 提供的样式声明代码 sns.set() 放置在绘图前即可
```python
import seaborn as sns

sns.set()  # 声明使用 Seaborn 样式

plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='y')
```
相比于 Matplotlib 默认的纯白色背景，Seaborn 默认的浅灰色网格背景看起来的确要细腻舒适一些。而柱状图的色调、坐标轴的字体大小也都有一些变化

sns.set() 的默认参数为：
```python
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
```
+ context='' 参数控制着默认的画幅大小，分别有 {paper, notebook, talk, poster} 四个值。其中，poster > talk > notebook > paper。
+ style='' 参数控制默认样式，分别有 {darkgrid, whitegrid, dark, white, ticks}，你可以自行更改查看它们之间的不同。
+ palette='' 参数为预设的调色板。分别有 {deep, muted, bright, pastel, dark, colorblind} 等，你可以自行更改查看它们之间的不同。
+ 剩下的 font='' 用于设置字体，font_scale= 设置字体大小，color_codes= 不使用调色板而采用先前的 'r' 等色彩缩写

### Seaborn绘图API
Seaborn 的绘图方法大致分类 6 类，分别是：
+ 关联图
+ 类别图
+ 分布图
+ 回归图
+ 矩阵图
+ 组合图

而这 6 大类下面又包含不同数量的绘图函数

#### 关联图

关联性分析|介绍
--|:--:
relplot|绘制关系图
scatterplot|多维度分析散点图
lineplot|多维度分析线形图

#### 类别图
与关联图相似，类别图的 Figure-level 接口是 `catplot`，其为 categorical plots 的缩写
```python
iris = sns.load_dataset("iris")  //导入数据集  自带的
iris.head()
```
```python
sns.catplot(x = "sepal_length", y = "species", data = iris)
```
`kind="swarm"` 可以让散点按照 beeswarm 的方式防止重叠，可以更好地观测数据分布

`hue= `参数可以给图像引入另一个维度
--------

##### 绘制箱线图
```python
sns.catplot(x="sepal_length", y="species", kind="box", data=iris)
```
##### 绘制小提琴图
```python
sns.catplot(x="sepal_length", y="species", kind="violin", data=iris)
```
##### 绘制增强箱线图
```python
sns.catplot(x="species", y="sepal_length", kind="boxen", data=iris)
```
##### 绘制点线图
```python
sns.catplot(x="sepal_length", y="species", kind="point", data=iris)
```
##### 绘制条形图
```python
sns.catplot(x="sepal_length", y="species", kind="bar", data=iris)
```
##### 绘制计数条形图
```python
sns.catplot(x="species", kind="count", data=iris)
```

------------------------------

### 分布图
分布图主要是用于可视化变量的分布情况，一般分为单变量分布和多变量分布。当然这里的多变量多指二元变量，更多的变量无法绘制出直观的可视化图形
Seaborn 提供的分布图绘制方法一般有这几个： jointplot，pairplot，distplot，kdeplot。接下来，我们依次来看一下这些绘图方法的使用

+ Seaborn 快速查看单变量分布的方法是 `distplo`t。默认情况下，该方法将会绘制直方图并拟合核密度估计图

```python
sns.distplot(iris["sepal_length"])
```
distplot 提供了参数来调整直方图和核密度估计图，例如设置 kde=False 则可以只绘制直方图，或者 hist=False 只绘制核密度估计图。当然，kdeplot 可以专门用于绘制核密度估计图，其效果和 distplot(hist=False) 一致，但 kdeplot 拥有更多的自定义设置

+ jointplot 主要是用于绘制二元变量分布图。例如，我们探寻 sepal_length 和 sepal_width 二元特征变量之间的关系
jointplot 并不是一个 Figure-level 接口，但其支持 kind= 参数指定绘制出不同样式的分布图。例如，绘制出核密度估计对比图

### 回归图
接下来，我们继续介绍回归图，回归图的绘制函数主要有：`lmplot` 和 `regplot`

regplot 绘制回归图时，只需要指定自变量和因变量即可，regplot 会自动完成线性回归拟合
```python
sns.regplot(x = "sepal_length", y = "sepal_width", data = iris)
```

lmplot 同样是用于绘制回归图，但 lmplot 支持引入第三维度进行对比，例如我们设置 hue="species"
```python
sns.lmplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
```
### 矩阵图
矩阵图中最常用的就只有 2 个，分别是：heatmap 和 clustermap。

意如其名，heatmap 主要用于绘制热力图
```python
import numpy as np

sns.heatmap(np.random.rand(10, 10))
```

















