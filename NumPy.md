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
+ randint(low, high, size, dtype) 方法将会生成 [low, high) 的随机整数。注意这是一个半开半闭区间

```python
np.random.randint(2, 5, (2, 5))
```
+ random_sample(size) 方法将会在 [0, 1) 区间内生成指定 size 的随机浮点数

```python
np.random.random_sample([2, 5])
```
与 numpy.random.random_sample 类似的方法还有：

+ numpy.random.random([size])
+ numpy.random.ranf([size])
+ numpy.random.sample([size])

+ choice(a, size, replace, p) 方法将会给定的数组里随机抽取几个值，该方法类似于随机抽样

```python
np.random.choice(10, 5)
```
#### 概率分布
概率密度分布
除了上面介绍的 6 种随机数生成方法，NumPy 还提供了大量的满足特定概率密度分布的样本生成方法。它们的使用方法和上面非常相似

+ numpy.random.beta(a，b，size)：从 Beta 分布中生成随机数。
+ numpy.random.binomial(n, p, size)：从二项分布中生成随机数。
+ numpy.random.chisquare(df，size)：从卡方分布中生成随机数。
+ numpy.random.dirichlet(alpha，size)：从 Dirichlet 分布中生成随机数。
+ numpy.random.exponential(scale，size)：从指数分布中生成随机数。
+ numpy.random.f(dfnum，dfden，size)：从 F 分布中生成随机数。
+ numpy.random.gamma(shape，scale，size)：从 Gamma 分布中生成随机数。
+ numpy.random.geometric(p，size)：从几何分布中生成随机数。
+ numpy.random.gumbel(loc，scale，size)：从 Gumbel 分布中生成随机数。
+ numpy.random.hypergeometric(ngood, nbad, nsample, size)：从超几何分布中生成随机数。
+ numpy.random.laplace(loc，scale，size)：从拉普拉斯双指数分布中生成随机数。
+ numpy.random.logistic(loc，scale，size)：从逻辑分布中生成随机数。
+ numpy.random.lognormal(mean，sigma，size)：从对数正态分布中生成随机数。
+ numpy.random.logseries(p，size)：从对数系列分布中生成随机数。+
+ numpy.random.multinomial(n，pvals，size)：从多项分布中生成随机数。
+ numpy.random.multivariate_normal(mean, cov, size)：从多变量正态分布绘制随机样本。
+ numpy.random.negative_binomial(n, p, size)：从负二项分布中生成随机数。
+ numpy.random.noncentral_chisquare(df，nonc，size)：从非中心卡方分布中生成随机数。
+ numpy.random.noncentral_f(dfnum, dfden, nonc, size)：从非中心 F 分布中抽取样本。
+ numpy.random.normal(loc，scale，size)：从正态分布绘制随机样本。
+ numpy.random.pareto(a，size)：从具有指定形状的 Pareto II 或 Lomax 分布中生成随机数。
+ numpy.random.poisson(lam，size)：从泊松分布中生成随机数。
+ numpy.random.power(a，size)：从具有正指数 a-1 的功率分布中在 0，1 中生成随机数。
+ numpy.random.rayleigh(scale，size)：从瑞利分布中生成随机数。
+ numpy.random.standard_cauchy(size)：从标准 Cauchy 分布中生成随机数。
+ numpy.random.standard_exponential(size)：从标准指数分布中生成随机数。
+ numpy.random.standard_gamma(shape，size)：从标准 Gamma 分布中生成随机数。
+ numpy.random.standard_normal(size)：从标准正态分布中生成随机数。
+ numpy.random.standard_t(df，size)：从具有 df 自由度的标准学生 t 分布中生成随机数。
+ numpy.random.triangular(left，mode，right，size)：从三角分布中生成随机数。
+ numpy.random.uniform(low，high，size)：从均匀分布中生成随机数。
+ numpy.random.vonmises(mu，kappa，size)：从 von Mises 分布中生成随机数。
+ numpy.random.wald(mean，scale，size)：从 Wald 或反高斯分布中生成随机数。
+ numpy.random.weibull(a，size)：从威布尔分布中生成随机数。
+ numpy.random.zipf(a，size)：从 Zipf 分布中生成随机数

### 数学函数
#### 三角函数
首先, 看一看 NumPy 提供的三角函数功能。这些方法有：

+ numpy.sin(x)：三角正弦。
+ numpy.cos(x)：三角余弦。
+ numpy.tan(x)：三角正切。
+ numpy.arcsin(x)：三角反正弦。
+ numpy.arccos(x)：三角反余弦。
+ numpy.arctan(x)：三角反正切。
+ numpy.hypot(x1,x2)：直角三角形求斜边。
+ numpy.degrees(x)：弧度转换为度。
+ numpy.radians(x)：度转换为弧度。
+ numpy.deg2rad(x)：度转换为弧度。
+ numpy.rad2deg(x)：弧度转换为度

#### 双曲函数
在数学中，双曲函数是一类与常见的三角函数类似的函数。双曲函数经常出现于某些重要的线性微分方程的解中，使用 NumPy 计算它们的方法为：

+ numpy.sinh(x)：双曲正弦。
+ numpy.cosh(x)：双曲余弦。
+ numpy.tanh(x)：双曲正切。
+ numpy.arcsinh(x)：反双曲正弦。
+ numpy.arccosh(x)：反双曲余弦。
+ numpy.arctanh(x)：反双曲正切

#### 数值修约
数值修约, 又称数字修约, 是指在进行具体的数字运算前, 按照一定的规则确定一致的位数, 然后舍去某些数字后面多余的尾数的过程。比如, 我们常听到的「4 舍 5 入」就属于数值修约中的一种。

+ numpy.around(a)：平均到给定的小数位数。
+ numpy.round_(a)：将数组舍入到给定的小数位数。
+ numpy.rint(x)：修约到最接近的整数。
+ numpy.fix(x, y)：向 0 舍入到最接近的整数。
+ numpy.floor(x)：返回输入的底部(标量 x 的底部是最大的整数 i)。
+ numpy.ceil(x)：返回输入的上限(标量 x 的底部是最小的整数 i).
+ numpy.trunc(x)：返回输入的截断值

#### 求和、求积、差分
下面这些方法用于数组内元素或数组间进行求和、求积以及进行差分。

+ numpy.prod(a, axis, dtype, keepdims)：返回指定轴上的数组元素的乘积。
+ numpy.sum(a, axis, dtype, keepdims)：返回指定轴上的数组元素的总和。
+ numpy.nanprod(a, axis, dtype, keepdims)：返回指定轴上的数组元素的乘积, 将 NaN 视作 1。
+ numpy.nansum(a, axis, dtype, keepdims)：返回指定轴上的数组元素的总和, 将 NaN 视作 0。
+ numpy.cumprod(a, axis, dtype)：返回沿给定轴的元素的累积乘积。
+ numpy.cumsum(a, axis, dtype)：返回沿给定轴的元素的累积总和。
+ numpy.nancumprod(a, axis, dtype)：返回沿给定轴的元素的累积乘积, 将 NaN 视作 1。
+ numpy.nancumsum(a, axis, dtype)：返回沿给定轴的元素的累积总和, 将 NaN 视作 0。
+ numpy.diff(a, n, axis)：计算沿指定轴的第 n 个离散差分。
+ numpy.ediff1d(ary, to_end, to_begin)：数组的连续元素之间的差异。
+ numpy.gradient(f)：返回 N 维数组的梯度。
+ numpy.cross(a, b, axisa, axisb, axisc, axis)：返回两个(数组）向量的叉积。
+ numpy.trapz(y, x, dx, axis)：使用复合梯形规则沿给定轴积分

#### 指数和对数
如果你需要进行指数或者对数求解，可以用到以下这些方法。

+ numpy.exp(x)：计算输入数组中所有元素的指数。
+ numpy.log(x)：计算自然对数。
+ numpy.log10(x)：计算常用对数。
+ numpy.log2(x)：计算二进制对数

#### 算术运算
当然，NumPy 也提供了一些用于算术运算的方法，使用起来会比 Python 提供的运算符灵活一些，主要是可以直接针对数组。

+ numpy.add(x1, x2)：对应元素相加。
+ numpy.reciprocal(x)：求倒数 1/x。
+ numpy.negative(x)：求对应负数。
+ numpy.multiply(x1, x2)：求解乘法。
+ numpy.divide(x1, x2)：相除 x1/x2。
+ numpy.power(x1, x2)：类似于 x1^x2。
+ numpy.subtract(x1, x2)：减法。
+ numpy.fmod(x1, x2)：返回除法的元素余项。
+ numpy.mod(x1, x2)：返回余项。
+ numpy.modf(x1)：返回数组的小数和整数部分。
+ numpy.remainder(x1, x2)：返回除法余数

#### 矩阵和向量积
求解向量、矩阵、张量的点积等同样是 NumPy 非常强大的地方。

+ numpy.dot(a, b)：求解两个数组的点积。
+ numpy.vdot(a, b)：求解两个向量的点积。
+ numpy.inner(a, b)：求解两个数组的内积。
+ numpy.outer(a, b)：求解两个向量的外积。
+ numpy.matmul(a, b)：求解两个数组的矩阵乘积。
+ numpy.tensordot(a, b)：求解张量点积。
+ numpy.kron(a, b)：计算 Kronecker 乘积

### 数组索引和切片
#### 数组切片
NumPy 里面针对Ndarray的数组切片和 Python 里的list 切片操作是一样的。其语法为：

+ Ndarray[start:stop:step]
+ [start:stop:step] 分布代表 [起始索引:截至索引:步长]。

对于多维数组，我们只需要用逗号 , 分割不同维度即可

#### 排序、搜索、计数
最后，再介绍几个 NumPy 针对数组元素的使用方法，分别是排序、搜索和计数。

我们可以使用 numpy.sort方法对多维数组元素进行排序。其方法为：
```python
numpy.sort(a, axis=-1, kind='quicksort', order=None)
```
其中：

+ a：数组。
+ axis：要排序的轴。如果为None，则在排序之前将数组铺平。默认值为 -1，沿最后一个轴排序。
+ kind：{'quicksort'，'mergesort'，'heapsort'}，排序算法。默认值为 quicksort







