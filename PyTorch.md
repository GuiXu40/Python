## PyTorch
### 简介
PyTorch 是由 Facebook 主导开发的深度学习框架，因其高效的计算过程以及良好的易用性被诸多大公司和科研人员所喜爱。本次实验中，我们将学习 PyTorch 的基础语法，了解 Autograd 自动求导机制，并最终利用 PyTorch 构建可用于图像分类任务的人工神经网络
### 张量
`Tensors`（张量）与 NumPy 中的 Ndarrays 多维数组类似，但是在 PyTorch 中 Tensors 可以使用 GPU 进行计算
使用 `torch.empty `可以返回填充了未初始化数据的张量。张量的形状由可变参数大小定义

```python
import torch
torch.empty(5, 3)
```
`torch.rand()`: 创建一个随机初始化的矩阵
```python
torch.rand(5, 3)
```
创建一个 0 填充的矩阵，指定数据类型为 long
```python
torch.zeros(5, 3, dtype = torch.long)
```
创建 Tensor 并使用现有数据初始化
```python
x = torch.tensor([5.5, 3])
x
```
根据现有张量创建新张量。这些方法将重用输入张量的属性，除非设置新的值进行覆盖
```python
x = x.new_ones(5, 3, dtype = torch.double)
x
```

### 操作
+ 加法
```python
y = torch.rand(5, 3)
x + y
```
2.
```python
torch.add(x, y)
```
提供输出 Tensor 作为参数

3.
```python
y.add_(x)  # 将 x 加到 y
y
```
> 任何以下划线结尾的操作都会用结果替换原变量。例如：x.copy_(y), x.t_(), 都会改变 x

`torch.view` 可以改变张量的维度和大小

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # size -1 从其他维度推断

x.size(), y.size(), z.size()

(torch.Size([4, 4]), torch.Size([16]), torch.Size([2, 8]))
```
如果张量只有一个元素，使用 .item() 来得到 Python 数据类型的数值
```python
x = torch.randn(1)

x, x.item()
```

### NumPy 转换
将 PyTorch 张量转换为 NumPy 数组（反之亦然）是一件轻而易举的事。PyTorch 张量和 NumPy 数组将共享其底层内存位置，改变一个也将改变另一个。

将 PyTorch 张量转换为 NumPy 数组
```python
a = torch.ones(5)
a

b = a.numpy()
b

a.add_(1)
a, b
```
NumPy 数组转换成 PyTorch 张量时，可以使用 from_numpy 完成
```python
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
a, b
```
### CUDA 张量
CUDA 张量是能够在 GPU 设备中运算的张量。使用 .to 方法可以将 Tensor 移动到 GPU 设备中：
```python
# is_available 函数判断是否有 GPU 可以使用
if torch.cuda.is_available():
    device = torch.device("cuda")          # torch.device 将张量移动到指定的设备中
    y = torch.ones_like(x, device=device)  # 直接从 GPU 创建张量
    x = x.to(device)                       # 或者直接使用 .to("cuda") 将张量移动到 cuda 中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # .to 也会对变量的类型做更改
    
tensor([1.4566], device='cuda:0')
tensor([1.4566], dtype=torch.float64)
```

