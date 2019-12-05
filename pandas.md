## pandas
### pandas简介
Pandas 是非常著名的开源数据处理库，我们可以通过它完成对数据集进行快速读取、转换、过滤、分析等一系列操作。除此之外，Pandas 拥有强大的缺失数据处理与数据透视功能，可谓是数据预处理中的必备利器
### 数据类型
Pandas 的数据类型主要有以下几种，它们分别是：Series（一维数组），DataFrame（二维数组），Panel（三维数组），Panel4D（四维数组），PanelND（更多维数组）。其中 Series 和 DataFrame 应用的最为广泛，几乎占据了使用频率 90% 以上
#### series
` Series `是 Pandas 中最基本的一维数组形式。其可以储存整数、浮点数、字符串等类型的数据。Series 基本结构如下：

pandas.Series(data=None, index=None)
其中，`data `可以是字典，或者NumPy 里的 ndarray 对象等。index 是数据索引，索引是 Pandas 数据结构中的一大特性，它主要的功能是帮助我们更快速地定位数据。
```python
import pandas as pd
s = pd.Series({'a': 10, 'b': 20, 'c': 30})
s

a    10
b    20
c    30
dtype: int64
```
`type(s)`: 查看类型

由于 Pandas 基于 NumPy 开发。那么 NumPy 的数据类型 ndarray 多维数组自然就可以转换为 Pandas 中的数据。而 Series 则可以基于 NumPy 中的一维数据转换
```python
import numpy as np
s = pd.Series(np.random.randn(5))
s
```
#### DataFrame
它仿佛是由多个 Series 拼合而成。它和 Series 的直观区别在于，数据不但具有行索引，且具有列索引

基本数据结构
```python
pandas.DataFrame(data=None, index=None, columns=None)
```
DataFrame 可以由以下多个类型的数据构建：

+ 一维数组、列表、字典或者 Series 字典。
+ 二维或者结构化的 numpy.ndarray。
+ 一个 Series 或者另一个 DataFrame

```python
df = pd.DataFrame({'one': pd.Series([1, 2, 3]), 'two': pd.Series([4, 5, 6])})
df
```
### 数据读取
基本格式
```python
pandas.read_数据集()
```
例： 
```python
df = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/906/los_census.csv")
df
```
--------------
为什么要将数据转换为 Series 或者 DataFrame 结构？
```
因为 Pandas 针对数据操作的全部方法都是基于 Pandas 支持的数据结构设计的。也就是说，只有 Series 或者 DataFrame 才能使用 Pandas 提供的方法和函数进行处理

```
--------------
### 基本操作
+ `head()`: 读取前几条
+ `tail()`: 读取后面几条
+ `describe()`: 相当于对数据集进行概览，会输出该数据集每一列数据的计数、最大值、最小值等
+ `.values `将 DataFrame 转换为 NumPy 数组
+ `.index`: 查看索引
+ `.columns`: 查看列名
+ `.shape`: 查看形状

### 数据选择
#### 基于索引数字选择
当我们新建一个 DataFrame 之后，如果未自己指定行索引或者列对应的标签，那么 Pandas 会默认从 0 开始以数字的形式作为行索引，并以数据集的第一行作为列对应的标签。其实，这里的「列」也有数字索引，默认也是从 0 开始，只是未显示出来。

所以，我们首先可以基于数字索引对数据集进行选择。这里用到的 Pandas 中的 `.iloc `方法。该方法可以接受的类型有：

+ 整数。例如：5
```python
df.iloc[:3]  //获取df的前三行数据
```
```python
df.iloc[5]   //选取特定的一行
```
> 那么选择多行，是不是 df.iloc[1, 3, 5] 这样呢？
答案是错误的。df.iloc[] 的 [[行]，[列]] 里面可以同时接受行和列的位置，如果你直接键入 df.iloc[1, 3, 5] 就会报错

+ 整数构成的列表或数组。例如：[1, 2, 3]
```python
df.iloc[[1, 3, 5], [1, 2]]  //获取1， 3， 5行并且1， 2列
```
```python
df.iloc[:, 1:4]  //选取1~4列
```
+ 布尔数组。
+ 可返回索引值的函数或参数

#### 基于标签名称选择
除了根据数字索引选择，还可以直接根据标签对应的名称选择。这里用到的方法和上面的 iloc 很相似，少了个 i 为 df.loc[]。

`df.loc[]` 可以接受的类型有：

+ 单个标签。例如：2 或 'a'，这里的 2 指的是标签而不是索引位置。
+ 列表或数组包含的标签。例如：['A', 'B', 'C']。
+ 切片对象。例如：'A':'E'，注意这里和上面切片的不同支持，首尾都包含在内。
+ 布尔数组。
+ 可返回标签的函数或参数

```python
df.loc[0:2]  //选择前三行
```
```python
df.loc[[0, 2, 4]]  //选择1， 3， 5行
```
```python
df.loc[:, 'Total Population':'Total Males']  //根据列名选择
```
#### 数据删减

+ `DataFrame.drop` 可以直接去掉数据集中指定的列和行。一般在使用时，我们指定 labels 标签参数，然后再通过 axis 指定按列或按行删除即可。当然，你也可以通过索引参数删除数据，具体查看官方文档

```python
df.drop(labels = ['Median Age', 'Total Males'], axis = 1)
```
+ `DataFrame.drop_duplicates` 则通常用于数据去重，即剔除数据集中的重复值。使用方法非常简单，指定去除重复值规则，以及 axis 按列还是按行去除即可。

+ 除此之外，另一个用于数据删减的方法 DataFrame.dropna 也十分常用，其主要的用途是删除缺少值，即数据集中空缺的数据列或行

#### 数据填充
##### 检测缺失值
Pandas 为了更方便地检测缺失值，将不同类型数据的缺失均采用 NaN 标记。这里的 NaN 代表 Not a Number，它仅仅是作为一个标记。例外是，在时间序列里，时间戳的丢失采用 NaT 标记
Pandas 中用于检测缺失值主要用到两个方法，分别是：isna() 和 notna()，故名思意就是「是缺失值」和「不是缺失值」。默认会返回布尔值用于判断

```python
df = pd.DataFrame(np.random.rand(9, 5), columns=list('ABCDE'))
# 插入 T 列，并打上时间戳
df.insert(value=pd.Timestamp('2017-10-1'), loc=0, column='Time')
# 将 1, 3, 5 列的 1，3，5 行置为缺失值
df.iloc[[1, 3, 5, 7], [0, 2, 4]] = np.nan
# 将 2, 4, 6 列的 2，4，6 行置为缺失值
df.iloc[[2, 4, 6, 8], [1, 3, 5]] = np.nan
df
```
通过 isna() 或 notna() 中的一个即可确定数据集中的缺失值

+ fillna(): 填充缺失值
```python
df.fillna(0)  //全部填充为0
```
除了直接填充值，我们还可以通过参数，将缺失值前面或者后面的值填充给相应的缺失值。例如使用缺失值前面的值进行填充
```python
df.fillna(method='pad')
```
或者是后面的值
```python
df.fillna(method='bfill')
```
##### 插值填充
插值是数值分析中一种方法。简而言之，就是借助于一个函数（线性或非线性），再根据已知数据去求解未知数据的值。插值在数据领域非常常见，它的好处在于，可以尽量去还原数据本身的样子。

我们可以通过 interpolate() 方法完成线性插值



