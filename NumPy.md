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

#### zeros 方法创建¶
`zeros `方法和上面的 ones 方法非常相似，不同的地方在于，这里全部填充为 0。zeros 方法和 ones 是一致的。
```python
numpy.zeros(shape, dtype=None, order='C')
```
+ shape：用于指定数组形状，例如（1， 2）或3。
+ dtype：数据类型。
+ order：{'C'，'F'}，按行或列方式储存数组。





