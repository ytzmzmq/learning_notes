# PyTorch 基础指南

> 基于 PyTorch 入门学习的实践整理（第 1 章：张量与基本运算），分为"张量基础"与"元素级运算"两大部分。运行环境：torch 2.13（CPU 版）。

---

# 一、张量基础（Tensors）

## 1.1 什么是张量

张量（Tensor）是 PyTorch 的基本单元，指任何 n 维度的数字数组——向量和矩阵也属于张量。使用 `torch.tensor()` 从 Python 列表创建，每个内层 `[]` 表示一行：

```python
import torch

test = torch.tensor([[3, 4], [5, 6]])
print(test)
# tensor([[3, 4],
#         [5, 6]])
print(type(test))   # <class 'torch.Tensor'>
```

## 1.2 张量的数据类型

创建张量时会根据列表元素自动推断类型：整数列表得到 `torch.int64`，浮点数列表得到 `torch.float32`。

```python
a = torch.tensor([1.0, 2.0, 4.0, 8.0])   # torch.float32
c = torch.tensor([4, -4, 0, 2])          # torch.int64
```

## 1.3 常用张量创建函数

**empty：未初始化的张量**

```python
example = torch.empty(3, 4)   # 3 行 4 列
```
`empty` 只分配内存不填值，内容是内存中遗留的**任意数据**，不能假设它是 0。默认使用 32 位浮点数（float32）。

**zeros / ones：全 0 / 全 1 张量**

```python
print(torch.zeros(3, 4))   # 全部为 0
print(torch.ones(3, 4))    # 全部为 1
```

**rand：0-1 均匀分布的随机张量**

```python
torch.manual_seed(32)        # 随机种子确保结果可重复
example_random = torch.rand(3, 4)
print(example_random)
```
`manual_seed` 设置相同的种子后，`rand` 生成的随机数序列完全一致，实验可复现。不设置种子则每次运行结果不同。

---

# 二、元素级运算

元素级（element-wise）运算指对张量中的每一个元素分别执行同样的操作，要求参与运算的张量形状相容。

## 2.1 算术运算

`+` `-` `*` `/` 都是逐元素进行的：

```python
a = torch.tensor([1.0, 2.0, 4.0, 8.0])
b = torch.tensor([1.0, 0.5, 0.25, 0.125])

print(a + b)   # tensor([2.0000, 2.5000, 4.2500, 8.1250])，逐元素相加
print(a - b)   # tensor([0.0000, 1.5000, 3.7500, 7.8750])
print(a * b)   # tensor([1., 1., 1., 1.])
print(a / b)   # tensor([ 1.,  4., 16., 64.])
```

注意：`*` 是逐元素相乘，不是矩阵乘法（矩阵乘法用 `torch.matmul` 等，后续章节补充）。

## 2.2 常用激活函数

**sigmoid：把任意值映射到 (0,1) 区间**

```python
print(torch.sigmoid(a))                  # tensor([0.7311, 0.8808, 0.9820, 0.9997])
print(torch.sigmoid(torch.tensor(0.)))   # tensor(0.5000)
```
输入越大输出越接近 1，越小越接近 0，`sigmoid(0) = 0.5`。输出值有时可以视为概率。

**relu（修正线性单元）：max(x, 0)**

```python
c = torch.tensor([4, -4, 0, 2])
print(torch.relu(c))   # tensor([4, 0, 0, 2])
```
任何正数原样保留，负数一律变成 0。

---

# 附录：常用函数与语法速查

## A. 张量创建函数速查

| 函数 | 作用 | 默认类型 |
|------|------|---------|
| `torch.tensor(列表)` | 从列表创建张量 | 整数→int64，浮点→float32 |
| `torch.empty(行, 列)` | 未初始化张量（值是任意的） | float32 |
| `torch.zeros(行, 列)` | 全 0 张量 | float32 |
| `torch.ones(行, 列)` | 全 1 张量 | float32 |
| `torch.rand(行, 列)` | 0-1 均匀分布随机张量 | float32 |
| `torch.manual_seed(n)` | 设置随机种子，保证可复现 | — |

## B. 元素级操作速查

| 操作 | 含义 |
|------|------|
| `+` `-` `*` `/` | 逐元素加减乘除 |
| `torch.sigmoid(x)` | 映射到 (0,1)，可视为概率 |
| `torch.relu(x)` | max(x, 0)，负数归零 |

## C. 易错点备忘

- `torch.empty()` 的值是未初始化的内存数据，不能假设为 0，想要全 0 用 `torch.zeros()`。
- `*` 是逐元素相乘而非矩阵乘法，形状不匹配会报错。
- `torch.rand` 每次运行结果不同，需要复现实验时先调用 `torch.manual_seed(种子)`。
- 整数列表创建的是 int64 张量，浮点列表是 float32，混合运算前注意类型。
- `torch.tensor(0)` 创建的是 0 维标量张量，与单元素向量不同。
