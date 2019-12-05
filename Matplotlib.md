## Matplotlib
### 简介
Matplotlib 是支持 Python 语言的开源绘图库，因为其支持丰富的绘图类型、简单的绘图方式以及完善的接口文档，深受 Python 工程师、科研学者、数据工程师等各类人士的喜欢
### 简单图形绘制
使用 Matplotlib 提供的面向对象 API，需要导入 pyplot 模块，并约定简称为 plt
```python
from matplotlib import pyplot as plt
```
+ plt.plot()--折线图
```python
plt.plot([1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
```
当然，如果你需要自定义横坐标值，只需要传入两个列表即可。如下方代码，我们自定义横坐标刻度从 2 开始
```python
plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         [1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
```

其他图形绘制的函数

方法|含义
--|:--:
matplotlib.pyplot.angle_spectrum|绘制电子波谱图
matplotlib.pyplot.bar|绘制柱状图
matplotlib.pyplot.barh|绘制直方图
matplotlib.pyplot.broken_barh|绘制水平直方图
matplotlib.pyplot.contour|绘制等高线图
matplotlib.pyplot.errorbar|绘制误差线
matplotlib.pyplot.hexbin|绘制六边形图案
matplotlib.pyplot.hist|绘制柱形图
matplotlib.pyplot.hist2d|绘制水平柱状图
matplotlib.pyplot.pie|绘制饼状图
matplotlib.pyplot.quiver|绘制量场图
matplotlib.pyplot.scatter|散点图
matplotlib.pyplot.specgram|绘制光谱图

例： 结合numpy画sin图像
```python
import numpy as np  # 载入数值计算模块

# 在 -2PI 和 2PI 之间等间距生成 1000 个值，也就是 X 坐标
X = np.linspace(-2*np.pi, 2*np.pi, 1000)
# 计算 y 坐标
y = np.sin(X)

# 向方法中 `*args` 输入 X，y 坐标
plt.plot(X, y)
```
但值得注意的是，pyplot.plot 在这里绘制的正弦曲线，实际上不是严格意义上的曲线图，而在两点之间依旧是直线。这里看起来像曲线是因为样本点相互挨得很近

+ 柱状图
```python
plt.bar([1, 2, 3], [1, 2, 3])
```
+ 散点图
```python
# X,y 的坐标均有 numpy 在 0 到 1 中随机生成 1000 个值
X = np.random.ranf(1000)
y = np.random.ranf(1000)
# 向方法中 `*args` 输入 X，y 坐标
plt.scatter(X, y)
```
+ 饼状图 matplotlib.pyplot.pie(*args, **kwargs) 在有限列表以百分比呈现时特别有用，你可以很清晰地看出来各类别之间的大小关系，以及各类别占总体的比例
```python
plt.pie([1, 2, 3, 4, 5])
```

+ 量场图 matplotlib.pyplot.quiver(*args, **kwargs) 就是由向量组成的图像，在气象学等方面被广泛应用。从图像的角度来看，量场图就是带方向的箭头符号

```python
X, y = np.mgrid[0:10, 0:10]
plt.quiver(X, y)
```
+ 等高线图 matplotlib.pyplot.contourf(*args, **kwargs) 是工程领域经常接触的一类图，它的绘制过程稍微复杂一些
```python
# 生成网格矩阵
x = np.linspace(-5, 5, 500)
y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(x, y)
# 等高线计算公式
Z = (1 - X / 2 + X ** 3 + Y ** 4) * np.exp(-X ** 2 - Y ** 2)

plt.contourf(X, Y, Z)
```

### 定义图形样式
+ 线性图
我们已经知道了，线形图通过 matplotlib.pyplot.plot(*args, **kwargs) 方法绘出。其中，args 代表数据输入，而 kwargs 的部分就是用于设置样式参数了

常见参数|含义
--|:--:
alpha=|设置线型的透明度，从 0.0 到 1.0
color=|设置线型的颜色
fillstyle=|设置线型的填充样式
linestyle=|设置线型的样式
linewidth=|设置线型的宽度
marker=|设置标记点的样式
……|……

例：重新绘制三角函数
```python
# 在 -2PI 和 2PI 之间等间距生成 1000 个值，也就是 X 坐标
X = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
# 计算 sin() 对应的纵坐标
y1 = np.sin(X)
# 计算 cos() 对应的纵坐标
y2 = np.cos(X)

# 向方法中 `*args` 输入 X，y 坐标
plt.plot(X, y1, color='r', linestyle='--', linewidth=2, alpha=0.8)
plt.plot(X, y2, color='b', linestyle='-', linewidth=2)
```
+ 散点图
参数|含义
--|:--：
s=|散点大小
c=|散点颜色
marker=|散点样式
cmap=|定义多类别散点的颜色
alpha=|点的透明度
edgecolors=|散点边缘颜色

```python
# 生成随机数据
x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)
size = np.random.normal(50, 60, 10)

plt.scatter(x, y, s=size, c=colors) 
```

+ 饼状图

```python
label = 'Cat', 'Dog', 'Cattle', 'Sheep', 'Horse'  # 各类别标签
color = 'r', 'g', 'r', 'g', 'y'  # 各类别颜色
size = [1, 2, 3, 4, 5]  # 各类别占比
explode = (0, 0, 0, 0, 0.2)  # 各类别的偏移半径
# 绘制饼状图
plt.pie(size, colors=color, explode=explode,
        labels=label, shadow=True, autopct='%1.1f%%')
# 饼状图呈正圆
plt.axis('equal')
```

### 定义图形位置
在图形的绘制过程中，你可能需要调整图形的位置，或者把几张单独的图形拼接在一起。此时，我们就需要引入 plt.figure 图形对象了

例： 
```python
x = np.linspace(0, 10, 20)  # 生成数据
y = x * x + 2

fig = plt.figure()  # 新建图形对象
axes = fig.add_axes([0.5, 0.5, 0.8, 0.8])  # 控制画布的左, 下, 宽度, 高度
axes.plot(x, y, 'r')
```
上面的绘图代码中，你可能会对 figure 和 axes 产生疑问。Matplotlib 的 API 设计的非常符合常理，在这里，figure 相当于绘画用的画板，而 axes 则相当于铺在画板上的画布。我们将图像绘制在画布上，于是就有了 plot，set_xlabel 等操作

根据这个，可以实现大图套小图
```python
fig = plt.figure()  # 新建画板
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # 大画布
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])  # 小画布

axes1.plot(x, y, 'r')  # 大画布
axes2.plot(y, x, 'g')  # 小画布
```
 Matplotlib 中，还有一种添加画布的方式，那就是 plt.subplots()，它和 axes 都等同于画布
 
```python
 fig, axes = plt.subplots()
axes.plot(x, y, 'r')
```
 
 借助于 plt.subplots()，我们就可以实现子图的绘制，也就是将多张图按一定顺序拼接在一起。
 
 ```python
 fig, axes = plt.subplots(nrows=1, ncols=2)  # 子图为 1 行，2 列
for ax in axes:
    ax.plot(x, y, 'r')
 ```
 通过设置 plt.subplots 的参数，可以实现调节画布尺寸和显示精度
 ```python
 fig, axes = plt.subplots(
    figsize=(16, 9), dpi=50)  # 通过 figsize 调节尺寸, dpi 调节显示精度
axes.plot(x, y, 'r')
 ```
 
 
### 规范绘图方法
 
 首先，任何图形的绘制，都建议通过 plt.figure() 或者 plt.subplots() 管理一个完整的图形对象。而不是简单使用一条语句，例如 plt.plot(...) 来绘图。
#### 添加图标题、图例
绘制包含图标题、坐标轴标题以及图例的图形，举例如下：
 
```python
fig, axes = plt.subplots()

axes.set_xlabel('x label')  # 横轴名称
axes.set_ylabel('y label')
axes.set_title('title')  # 图形名称

axes.plot(x, x**2)
axes.plot(x, x**3)
axes.legend(["y = x**2", "y = x**3"], loc=0)  # 图例
```

#### 线型、颜色、透明度
在 Matplotlib 中，你可以设置线的颜色、透明度等其他属性
```python
fig, axes = plt.subplots()

axes.plot(x, x+1, color="red", alpha=0.5)
axes.plot(x, x+2, color="#1155dd")
axes.plot(x, x+3, color="#15cc55")
```
而对于线型而言，除了实线、虚线之外，还有很多丰富的线型可供选择

```python
fig, ax = plt.subplots(figsize=(12, 6))

# 线宽
ax.plot(x, x+1, color="blue", linewidth=0.25)
ax.plot(x, x+2, color="blue", linewidth=0.50)
ax.plot(x, x+3, color="blue", linewidth=1.00)
ax.plot(x, x+4, color="blue", linewidth=2.00)

# 虚线类型
ax.plot(x, x+5, color="red", lw=2, linestyle='-')
ax.plot(x, x+6, color="red", lw=2, ls='-.')
ax.plot(x, x+7, color="red", lw=2, ls=':')

# 虚线交错宽度
line, = ax.plot(x, x+8, color="black", lw=1.50)
line.set_dashes([5, 10, 15, 10])

# 符号
ax.plot(x, x + 9, color="green", lw=2, ls='--', marker='+')
ax.plot(x, x+10, color="green", lw=2, ls='--', marker='o')
ax.plot(x, x+11, color="green", lw=2, ls='--', marker='s')
ax.plot(x, x+12, color="green", lw=2, ls='--', marker='1')

# 符号大小和颜色
ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
ax.plot(x, x+15, color="purple", lw=1, ls='-',
        marker='o', markersize=8, markerfacecolor="red")
ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8,
        markerfacecolor="yellow", markeredgewidth=2, markeredgecolor="blue")
```
#### 画布网格、坐标轴范围
有些时候，我们可能需要显示画布网格或调整坐标轴范围。设置画布网格和坐标轴范围。这里，我们通过指定 axes[0] 序号，来实现子图的自定义顺序排列。
```python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 显示网格
axes[0].plot(x, x**2, x, x**3, lw=2)
axes[0].grid(True)

# 设置坐标轴范围
axes[1].plot(x, x**2, x, x**3)
axes[1].set_ylim([0, 60])
axes[1].set_xlim([2, 5])
```
除了折线图，Matplotlib 还支持绘制散点图、柱状图等其他常见图形。下面，我们绘制由散点图、梯步图、条形图、面积图构成的子图
```python
n = np.array([0, 1, 2, 3, 4, 5])

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

axes[0].scatter(x, x + 0.25*np.random.randn(len(x)))
axes[0].set_title("scatter")

axes[1].step(n, n**2, lw=2)
axes[1].set_title("step")

axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")

axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5)
axes[3].set_title("fill_between")
```











