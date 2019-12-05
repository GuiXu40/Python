## Numpy
### 简介
如果你使用 Python 语言进行科学计算，那么一定会接触到 NumPy。NumPy 是支持 Python 语言的数值计算扩充库，其拥有强大的多维数组处理与矩阵运算能力。除此之外，NumPy 还内建了大量的函数，方便你快速构建数学模型

**支持的数据类型**

类型|解释
--|:--:
bool|布尔类型，1 个字节，值为 True 或 False。
int|整数类型，通常为 int64 或 int32 。
intc|与 C 里的 int 相同，通常为 int32 或 int64。
intp|用于索引，通常为 int32 或 int64。
int8|字节（从 -128 到 127）
int16|整数（从 -32768 到 32767）
int32|整数（从 -2147483648 到 2147483647）
int64|整数（从 -9223372036854775808 到 9223372036854775807）
uint8|无符号整数（从 0 到 255）
uint16|无符号整数（从 0 到 65535）
uint32|无符号整数（从 0 到 4294967295）
uint64|无符号整数（从 0 到 18446744073709551615）
float|float64 的简写。
float16|半精度浮点，5 位指数，10 位尾数
float32|单精度浮点，8 位指数，23 位尾数
float64|双精度浮点，11 位指数，52 位尾数
complex|complex128 的简写。
complex64|复数，由两个 32 位浮点表示。
complex128|复数，由两个 64 位浮点表示。

上面提到的这些数值类型都被归于 dtype（data-type） 对象的实例

可以用 `numpy.dtype(object, align, copy) `来指定数值类型

例：
```python
import numpy as np
a = np.array([1.1, 2.2, 3.3], dtype = np.float64)
a, a.dtype
```
> (array([1.1, 2.2, 3.3]), dtype('float64'))

`.astype() `方法在不同的数值类型之间相互转换
```python
a.astype(int).dtype
```

### NumPy数组生成
在 Python 内建对象中，数组有三种形式：

+ 列表：[1, 2, 3]
+ 元组：(1, 2, 3, 4, 5)
+ 字典：{A:1, B:2}

NumPy 中，`ndarray` 类具有六个参数，它们分别为：

+ shape：数组的形状。
+ dtype：数据类型。
+ buffer：对象暴露缓冲区接口。
+ offset：数组数据的偏移量。
+ strides：数据步长。
+ order：{'C'，'F'}，以行或列为主排列顺序

使用NumPy创建多维数组的方法：

+ 从 Python 数组结构列表，元组等转换。
+ 使用 np.arange、np.ones、np.zeros 等 NumPy 原生方法。
+ 从存储空间读取数组。
+ 通过使用字符串或缓冲区从原始字节创建数组。
+ 使用特殊函数，如 random

#### 列表或元组转换
```python
numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)
```
参数|解释
--|:--:
object|列表、元组等。
dtype|数据类型。如果未给出，则类型为被保存对象所需的最小类型。
copy|布尔类型，默认 True，表示复制对象。
order|顺序。
subok|布尔类型，表示子类是否被传递。
ndmin|生成的数组应具有的最小维数

例：
```python
np.array([[1, 2, 3], [4, 5, 6]])
np.array([(1, 2), (3, 4), (5, 6)])
```

#### arange方法创建
`arange() `的功能是在给定区间内创建一系列均匀间隔的值。方法如下：
```python
numpy.arange(start, stop, step, dtype=None)
```
例：
```python
# 在区间 [3, 7) 中以 0.5 为步长新建数组
np.arange(3, 7, 0.5, dtype='float32')
```
#### linspace 方法创建
`linspac`e方法也可以像arange方法一样，创建数值有规律的数组。`linspace `用于在指定的区间内返回间隔均匀的值。其方法如下：
```python
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```
+ start：序列的起始值。
+ stop：序列的结束值。
+ num：生成的样本数。默认值为50。
+ endpoint：布尔值，如果为真，则最后一个样本包含在序列内。
+ retstep：布尔值，如果为真，返回间距。
+ dtype：数组的类型。

例：
```python
np.linspace(0, 10, 10, endpoint=True)

np.linspace(0, 10, 10, endpoint=False)
```
#### ones 方法创建
`numpy.ones` 用于快速创建数值全部为 1 的多维数组。其方法如下
```python
numpy.ones(shape, dtype=None, order='C')
```
+ shape：用于指定数组形状，例如（1， 2）或 3。
+ dtype：数据类型。
+ order：{'C'，'F'}，按行或列方式储存数组。
```python
np.ones((2, 3), order = 'F')
```

#### zeros 方法创建
`zeros `方法和上面的 ones 方法非常相似，不同的地方在于，这里全部填充为 0。zeros 方法和 ones 是一致的。
```python
numpy.zeros(shape, dtype=None, order='C')
```
+ shape：用于指定数组形状，例如（1， 2）或3。
+ dtype：数据类型。
+ order：{'C'，'F'}，按行或列方式储存数组。

#### eye 方法创建
`numpy.eye` 用于创建一个二维数组，其特点是k 对角线上的值为 1，其余值全部为0。方法如下：
```python
numpy.eye(N, M=None, k=0, dtype=<type 'float'>)
```
+ N：输出数组的行数。
+ M：输出数组的列数。
+ k：对角线索引：0（默认）是指主对角线，正值是指上对角线，负值是指下对角线

```python
np.eye(5, 5, 3, dtype = 'int')
```
#### 从已知数据创建
我们还可以从已知数据文件、函数中创建 ndarray。NumPy 提供了下面 5 个方法：

+ frombuffer（buffer）：将缓冲区转换为 1 维数组。
+ fromfile（file，dtype，count，sep）：从文本或二进制文件中构建多维数组。
+ fromfunction（function，shape）：通过函数返回值来创建多维数组。
+ fromiter（iterable，dtype，count）：从可迭代对象创建 1 维数组。
+ fromstring（string，dtype，count，sep）：从字符串中创建 1 维数组

### ndarray数组属性
+ ndarray.T转置数组
+ ndarray.dtype 用来输出数组包含元素的数据类型
+ ndarray.imag 用来输出数组包含元素的虚部
+ ndarray.real用来输出数组包含元素的实部
+ ndarray.size用来输出数组中的总包含元素数# ndarray.itemsize输出一个数组元素的字节数
+ ndarray.nbytes用来输出数组的元素总字节数
+ ndarray.ndim用来输出数组尺寸
+ ndarray.shape用来输出数组维数组
+ ndarray.strides用来遍历数组时，输出每个维度中步进的字节数组

### 数组基本操作
#### 重设形状
reshape 可以在不改变数组数据的同时，改变数组的形状。其中，numpy.reshape() 等效于  ndarray.reshape()。reshape 方法非常简单：
```python
numpy.reshape(a, newshape)
```
```
np.arange(10).reshape((5, 2))

a = np.arange(10)
np.reshape(a, (2, 5))
```
#### 数组展开
ravel 的目的是将任意形状的数组扁平化，变为 1 维数组。ravel 方法如下：
```python
numpy.ravel(a, order='C')
```
其中，a 表示需要处理的数组。order 表示变换时的读取顺序，默认是按照行依次读取，当 order='F' 时，可以按列依次读取排序

```python
np.ravel(a, order = 'F')
np.ravel(a, order = 'C')
```
#### 轴移动
moveaxis 可以将数组的轴移动到新的位置。其方法如下：
```python
numpy.moveaxis(a, source, destination)
```
+ a：数组。
+ source：要移动的轴的原始位置。
+ destination：要移动的轴的目标位置。

`swapaxes` 可以用来交换数组的轴。其方法如下：
```python
numpy.swapaxes(a, axis1, axis2)
```
+ a：数组。
+ axis1：需要交换的轴 1 位置。
+ axis2：需要与轴 1 交换位置的轴 1 位置

#### 数组转置
transpose 类似于矩阵的转置，它可以将 2 维数组的横轴和纵轴交换。其方法如下：
```python
numpy.transpose(a, axes=None)
```
其中：

+ a：数组。
+ axis：该值默认为 none，表示转置。如果有值，那么则按照值替换轴
```python
a = np.arange(4).reshape(2, 2)
np.transpose(a)
```
#### 维度改变
atleast_xd 支持将输入数据直接视为 x维。这里的 x 可以表示：1，2，3。方法分别为：
```python
numpy.atleast_1d()
numpy.atleast_2d()
numpy.atleast_3d()
```
例
```python
print(np.atleast_1d([1, 2, 3]))
print(np.atleast_2d([4, 5, 6]))
print(np.atleast_3d([7, 8, 9]))
```
#### 类型转换
在 NumPy 中，还有一系列以 as 开头的方法，它们可以将特定输入转换为数组，亦可将数组转换为矩阵、标量，ndarray 等。如下：
     
+ asarray(a，dtype，order)：将特定输入转换为数组。
+ asanyarray(a，dtype，order)：将特定输入转换为 ndarray。
+ asmatrix(data，dtype)：将特定输入转换为矩阵。
+ asfarray(a，dtype)：将特定输入转换为 float 类型的数组。
+ asarray_chkfinite(a，dtype，order)：将特定输入转换为数组，检查 NaN 或 infs。
+ asscalar(a)：将大小为 1 的数组转换为标量

#### 数组连接
`concatenate` 可以将多个数组沿指定轴连接在一起。其方法为：
```python
numpy.concatenate((a1, a2, ...), axis=0)
```
其中：

+ (a1, a2, ...)：需要连接的数组。
+ axis：指定连接轴(不理解这里的轴是什么意思-->相当于维度)

例
```python
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[7, 8], [9, 10]])
c = np.array([[11, 12]])

np.concatenate((a, b, c), axis=0)
```
这里，我们可以尝试沿着横轴连接。但要保证连接处的维数一致，所以这里用到了 .T 转置
```python
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[7, 8, 9]])

np.concatenate((a, b.T), axis=1)
```

#### 数组堆叠
在 NumPy 中，以下方法可用于数组的堆叠：

+ stack(arrays，axis)：沿着新轴连接数组的序列。
+ column_stack()：将 1 维数组作为列堆叠到 2 维数组中。
+ hstack()：按水平方向堆叠数组。
+ vstack()：按垂直方向堆叠数组。
+ dstack()：按深度方向堆叠数组

#### 拆分
split 及与之相似的一系列方法主要是用于数组的拆分，列举如下：

+ split(ary，indices_or_sections，axis)：将数组拆分为多个子数组。
+ dsplit(ary，indices_or_sections)：按深度方向将数组拆分成多个子数组。
+ hsplit(ary，indices_or_sections)：按水平方向将数组拆分成多个子数组。
+ vsplit(ary，indices_or_sections)：按垂直方向将数组拆分成多个子数组

```python
a = np.arange(10)
np.split(a, 5)
```
高维度也是可以拆分的
```python
a = np.arange(10).reshape(2, 5)
np.split(a, 2)
```
#### 删除
首先是 delete 删除：

+ `delete(arr，obj，axis)`：沿特定轴删除数组中的子数组
```python
a = np.arange(12).reshape(3, 4)
np.delete(a, 2, 1)
```
#### 数组插入
再看一看 insert插入，用法和 delete 很相似，只是需要在第三个参数位置设置需要插入的数组对象：

+ insert(arr，obj，values，axis)：依据索引在特定轴之前插入值
```python
a = np.arange(12).reshape(3, 4)
b = np.arange(4)

np.insert(a, 2, b, 0)
```
#### 附加
`append `的用法也非常简单。只需要设置好需要附加的值和轴位置就好了。它其实相当于只能在末尾插入的 insert，所以少了一个指定索引的参数。

+ append(arr，values，axis)：将值附加到数组的末尾，**并返回 1 维数组**
```python
a = np.arange(6).reshape(2, 3)
b = np.arange(3)

np.append(a, b)
```
#### 重设尺寸
+ resize(a，new_shape)：对数组尺寸进行重新设定
```python
a = np.arange(10)
a.resize(2, 5)
a
```
------------
reshape 在改变形状时，不会影响原数组，相当于对原数组做了一份拷贝。而 resize 则是对原数组执行操作

-------------

#### 翻转数组
在 NumPy 中，我们还可以对数组进行翻转操作：

+ fliplr(m)：左右翻转数组。
+ flipud(m)：上下翻转数组

```python
a = np.arange(16).reshape(4, 4)
print(a)
print(np.fliplr(a))
print(np.flipud(a))
```
### NumPy 随机数
NumPy 的随机数功能非常强大，主要由 numpy.random 模块完成

`numpy.random.rand(d0, d1, ..., dn)` 方法的作用为：指定一个数组，并使用 [0, 1) 区间随机数据填充，这些数据均匀分布

```python
np.random.rand(2, 5)
```
```python
np.random.randn(1, 10)  // 返回标准正态分布
```








