# 数据科学实践指南

> 基于《数据科学实践与Python应用》第 2 章"向量"的学习实践整理，分为"NumPy 数组与向量运算"与"matplotlib 向量的几何可视化"两大部分。运行环境：numpy 2.3、matplotlib 3.10。

---

# 一、NumPy 数组与向量

## 1.1 创建向量（np.array）

Python 列表用 `[]` 直接书写；NumPy 用 `np.array()` 构建数组。数组的形状决定它是向量、行向量还是列向量：

```python
import numpy as np

asList = [1, 2, 3]                  # 普通列表
asArray = np.array([1, 2, 3])       # 一维数组
rowVec = np.array([[1, 2, 3]])      # 行向量：1 行 3 列
colVec = np.array([[1], [2], [3]])  # 列向量：3 行 1 列
```

## 1.2 观察形状（shape）

`np.shape()` 或数组的 `.shape` 属性都可以查看形状；返回的元组按"行数, 列数"给出：

```python
print(np.shape(asList))   # (3,)，列表也能被 np.shape 检查
print(asArray.shape)      # (3,)，一维
print(rowVec.shape)       # (1, 3)
print(colVec.shape)       # (3, 1)
```

一维数组 `(3,)` 既不是行向量也不是列向量，它没有"行"这一维。

## 1.3 向量加法

`+` 对数组是**逐元素相加**，只有维数（形状）匹配才能相加：

```python
v = np.array([4, 5, 6])
w = np.array([7, 8, 9])
print(v + w)   # [11 13 15]
```

## 1.4 广播运算（broadcasting）

行向量加列向量时触发广播：两个向量的每个元素之间重复同一个运算，生成一个矩阵（形状取两维的最大值）：

```python
v1 = np.array([[1, 2, 3]])        # (1, 3) 行向量
w1 = np.array([[4], [5]])         # (2, 1) 列向量
print(v1 + w1)
# [[5 6 7]
#  [6 7 8]]    结果形状 (2, 3)
```

## 1.5 标量与数组的运算

标量乘列表是**重复列表**，标量乘 NumPy 数组是**逐元素乘法**；标量加数组时，标量会加到每一个元素上：

```python
s = 2
a = [1, 2, 3]
b = np.array(a)
print(s * a)   # [1, 2, 3, 1, 2, 3]，列表被重复
print(s * b)   # [2 4 6]，逐元素相乘
print(s + b)   # [3 4 5]，标量加到每个元素
```

## 1.6 维数与幅度

向量的**维数（dimensionality）**用 `len()` 获得；向量的**几何长度/幅度（magnitude）**用 `np.linalg.norm()` 获得：

```python
v2 = np.array([1, 2, 5, 6, 7, 9, 11])
print(len(v2))               # 7，维数
print(np.linalg.norm(v2))    # 17.804493814764857，幅度（欧氏长度）
```

## 1.7 向量的点积（np.dot）

点积用 `np.dot()` 计算，**只有数组或行向量 × 列向量**才能算点积，结果形状为 (1,1)：

```python
v3 = np.array([[1, 4, 5, 2]])          # 行向量 (1, 4)
w3 = np.array([[-2], [4], [5], [-3]])  # 列向量 (4, 1)
print(np.dot(v3, w3))
# [[33]]    = 1×(-2) + 4×4 + 5×5 + 2×(-3)
```

验证点积的分配律 `v·(w+u) = v·w + v·u`：

```python
v4 = np.array([1, 4, 5, 3])
w4 = np.array([5, 6, 2, -4])
u4 = np.array([-2, 4, 5, -1])
res1 = np.dot(v4, w4 + u4)
res2 = np.dot(v4, w4) + np.dot(v4, u4)
print(res1, res2)   # 63 63，两边相等
```

## 1.8 Hadamard 乘法

**Hadamard 乘法**指相同维数的向量对应元素分别相乘，用普通 `*` 即可：

```python
v5 = np.array([[1, 4, 5, 2]])
w5 = np.array([7, 2, 0, -3])
print(v5 * w5)   # [[ 7  8  0 -6]]，逐元素相乘
```

## 1.9 外积（np.outer）

外积由一个行向量和一个列向量创建矩阵，每一项为对应行的元素乘对应列的元素。`np.outer(a, b)` 会把两个输入都展平成 1 维，生成 `len(a) × len(b)` 的矩阵：

```python
res3 = np.outer(v5, w5)
print(res3)
# [[  7   2   0  -3]      第 i 行 = v5 的第 i 个元素 × w5
#  [ 28   8   0 -12]
#  [ 35  10   0 -15]
#  [ 14   4   0  -6]]
```

## 1.10 案例：自定义向量范数算法（练习 2）

向量范数即几何长度：元素平方求和再开方。NumPy 支持逐元素运算，不需要写循环——`v ** 2` 自动平方每个元素，再配合 `np.sum()` 求和、`np.sqrt()` 开方：

```python
def norm_calculate(v):
    return np.sqrt(np.sum(v ** 2))

u1 = np.array([3, 0, 4, -4, 2, -2])
print(norm_calculate(u1))     # 7.0
print(np.linalg.norm(u1))     # 7.0，与官方函数一致
```

**批量验证**：用随机向量在 1~100 维上逐一对比自定义算法与 `np.linalg.norm()`，只要有一次不一致就把标志位置 False 并 `break`。浮点数比较用 `np.isclose()` 而不是 `==`：

```python
np.random.seed(44)                 # 固定随机种子，保证可复现
dimensions = range(1, 101)
all_passed = True                  # 状态标志位
for dim in dimensions:
    v_1d = np.random.randn(dim)    # 生成长度为 dim 的随机一维数组
    val_custom = norm_calculate(v_1d)
    val_numpy = np.linalg.norm(v_1d)
    if not np.isclose(val_custom, val_numpy):
        print(f"测试失败！维度 {dim}, 自定义={val_custom}, 官方={val_numpy}")
        all_passed = False
        break
if all_passed:
    print("完美！1~100 维测试结果完全一致！")
```

练习目的是练习数组下标和把公式转成代码；实际项目中直接用 `np.linalg.norm()` 更简单。

## 1.11 案例：单位向量（练习 3）

单位向量是与原向量方向相同、长度为 1 的向量：把向量除以它自己的范数。

```python
def unit_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm

v1 = np.array([3, 2, -3, 1, -1])
res1 = unit_vector(v1)
print(res1)          # [ 0.61237244  0.40824829 -0.61237244  0.20412415 -0.20412415]
print(np.sum(res1 ** 2))   # 1.0，单位向量各元素平方和为 1
```

**输入零向量会怎样？** 范数为 0，除以 0 时 NumPy 不抛异常，而是发出 RuntimeWarning 并返回全 `nan` 的向量：

```python
print(unit_vector(np.zeros((4, 1))))
# RuntimeWarning: invalid value encountered in divide
# [[nan] [nan] [nan] [nan]]
```
健壮写法是在函数里判断 `if norm == 0:` 并做处理（如返回原向量或抛出异常）。

## 1.12 案例：任意幅度的向量（练习 4）

在单位向量的基础上再乘以目标幅度，就得到方向与输入向量相同、幅度为给定值的向量：

```python
def create_mag_vector(vector, magnitude):
    norm = np.linalg.norm(vector)
    return magnitude * vector / norm

v1 = np.array([1, 0, 0])
print(create_mag_vector(v1, 3))     # [3. 0. 0.]

m = 10
v2 = np.array([0, 3, 4])            # 原幅度为 5
res1 = create_mag_vector(v2, m)
print(res1, np.linalg.norm(res1))   # [0. 6. 8.] 10.0，方向不变、幅度变为 10
```

## 1.13 案例：for 循环实现行向量转列向量（练习 5）

不用 `np.transpose()` 或 `v.T`，用循环逐元素搬运。关键点是 `shape` 的维度含义与二维数组的 `a[i, j]` 索引：

```python
def transpose_vector(rowVec):
    dimension = rowVec.shape[1]     # shape[0] 是行数（垂直方向），shape[1] 是列数（水平方向）
    colVec = np.zeros((dimension, 1), dtype=rowVec.dtype)
    for i in range(dimension):
        colVec[i, 0] = rowVec[0, i] # 行向量第 0 行第 i 列 → 列向量第 i 行第 0 列
    return colVec

v1 = np.array([[1, 2, 3, 4, 5]])    # 行向量 (1, 5)
print(transpose_vector(v1))         # (5, 1) 列向量 [[1][2][3][4][5]]
```

注意 `np.zeros()` 不指定 dtype 时默认创建**浮点数组**；这里用 `dtype=rowVec.dtype` 继承原数组类型，否则整数向量转置后会变成浮点数。

## 1.14 案例：向量与自身的点积 = 范数的平方（练习 6）

一个有趣的事实：`v·v` 等于范数的平方（元素平方和正是点积本身）：

```python
v1 = np.random.randn(10)            # 随机向量
res1 = np.dot(v1, v1)               # 8.715997589354338
res2 = (np.linalg.norm(v1)) ** 2    # 8.71599758935434
```
两者在浮点精度内相等（末位可能有微小差异，比较时应用 `np.isclose()` 而非 `==`）。

---

# 二、向量的几何可视化（matplotlib）

## 2.1 导入与全局设置

```python
import numpy as np
import matplotlib.pyplot as plt

# 以下两行用于 Jupyter 环境的显示优化，普通脚本可省略：
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg')  # 矢量格式输出
# plt.rcParams.update({'font.size': 14})                           # 全局字号
```

## 2.2 画向量箭头（plt.arrow）

`plt.arrow(x, y, dx, dy, ...)` 的前四个参数表示：从起点 `(x, y)` 开始，沿 X 轴延伸 `dx`、沿 Y 轴延伸 `dy`。

```python
v = np.array([1, 2])
w = np.array([4, -6])
vPlusW = v + w      # (5, -4)
vMinusW = v - w     # (-3, 8)

plt.figure(figsize=(6, 6))   # 创建画布：宽 6 英寸、高 6 英寸

a1 = plt.arrow(0, 0,              # 起点为原点
               v[0], v[1],        # 延伸分量为 v
               head_width=.3,     # 箭头头部宽度
               width=.1,          # 箭身粗细
               color='k',         # 黑色（'k' 代表 black）
               length_includes_head=True)   # 总长含箭头尖，终点精确停在向量末端
a2 = plt.arrow(v[0], v[1], w[0], w[1],      # w 从 v 的终点出发（三角形法则）
               head_width=.3, width=.1,
               color=[.5, .5, .5],          # RGB 灰度：中灰色
               length_includes_head=True)
a3 = plt.arrow(0, 0, vPlusW[0], vPlusW[1],  # 合向量从原点出发
               head_width=.3, width=.1,
               color=[.8, .8, .8],          # 浅灰色
               length_includes_head=True)
```

颜色既可用单字母（`'k'` 黑等），也可用 `[r, g, b]` 灰度值列表（如 `[.5,.5,.5]` 中灰、`[.8,.8,.8]` 浅灰）。

## 2.3 美化与输出

```python
plt.grid(linestyle='--', linewidth=.5)   # 背景网格：虚线、粗细 0.5
plt.axis('square')                       # 横纵比强制 1:1，几何图形不失真
plt.axis([-6, 6, -6, 6])                 # 显示范围 [xmin, xmax, ymin, ymax]
plt.legend([a1, a2, a3], ['v', 'w', 'v+w'])   # 图例：箭头对象与标签一一对应
plt.title('Vectors v, w, and v+w')       # 标题
plt.savefig('练习2_1a.png', dpi=300)     # 保存为 300 dpi 的 PNG
plt.show()                               # 在屏幕/Notebook 中显示
```

## 2.4 案例：v+w 与 v−w 的几何效果

以 v=(1,2)、w=(4,-6) 为例（对应教材图 2-2）：

- **v+w**：把 w 的起点平移到 v 的终点，从原点指向 w 终点的向量即为 v+w=(5,-4)（平行四边形/三角形法则）。
- **v−w**：w 从原点画出，从 w 的终点画 v−w，其终点落在 v 上——因为 w+(v−w)=v。两图共用 2.2/2.3 的画法，只差第二个箭头的起点和图例标签（`['v', 'w', 'v-w']`）。

---

# 附录：常用函数与语法速查

## A. NumPy 向量运算速查

| 操作 | 写法 | 说明 |
|------|------|------|
| 创建数组 | `np.array([...])` | 一维 `(n,)`、行 `(1,n)`、列 `(n,1)` |
| 查看形状 | `a.shape` / `np.shape(a)` | 返回 (行, 列) 元组 |
| 加法 | `v + w` | 逐元素，形状需匹配 |
| 广播 | `(1,n) + (n,1)` → `(n,n)` | 元素两两重复运算 |
| 标量运算 | `s * b` / `s + b` | 作用到每个元素 |
| 维数 | `len(v)` | 元素个数 |
| 幅度 | `np.linalg.norm(v)` | 欧氏长度 |
| 点积 | `np.dot(v, w)` | 行×列，结果为 (1,1) |
| Hadamard | `v * w` | 对应元素相乘 |
| 外积 | `np.outer(v, w)` | 展平后生成 len(v)×len(w) 矩阵 |
| 自定义范数 | `np.sqrt(np.sum(v ** 2))` | 等价于 `np.linalg.norm(v)` |
| 单位向量 | `v / np.linalg.norm(v)` | 零向量会得到全 nan |
| 任意幅度向量 | `m * v / np.linalg.norm(v)` | 同方向、幅度为 m |
| 范数平方 | `np.dot(v, v)` == `norm(v)**2` | 向量与自身点积 |
| 保类型零数组 | `np.zeros(shape, dtype=v.dtype)` | 默认是 float64 |
| 浮点比较 | `np.isclose(a, b)` | 代替 `==` 判断浮点数 |
| 随机数组 | `np.random.randn(n)` | n 个标准正态随机数 |

## B. matplotlib 绘图速查

| 函数 | 作用 |
|------|------|
| `plt.figure(figsize=(w,h))` | 创建画布（英寸） |
| `plt.arrow(x, y, dx, dy, ...)` | 画向量箭头 |
| `head_width` / `width` | 箭头宽 / 箭身粗细 |
| `length_includes_head=True` | 箭头尖计入向量终点 |
| `color='k'` / `[r,g,b]` | 单字母色 / RGB 灰度 |
| `plt.grid(linestyle, linewidth)` | 背景网格 |
| `plt.axis('square')` | 横纵比 1:1 |
| `plt.axis([xmin,xmax,ymin,ymax])` | 显示范围 |
| `plt.legend([对象], [标签])` | 图例 |
| `plt.title(...)` | 标题 |
| `plt.savefig(名, dpi=300)` | 保存到文件 |
| `plt.show()` | 显示图像 |

## C. 易错点备忘

- 一维数组 `(3,)` 与行向量 `(1,3)`、列向量 `(3,1)` 形状不同，做点积/广播时行为不同。
- 只有维数匹配才能相加；不匹配时可能触发广播而不是报错，注意结果形状。
- 标量 × 列表是重复列表，标量 × 数组才是逐元素相乘。
- `np.dot` 要求行向量 × 列向量（或两个一维等长数组），形状不对会报错。
- `np.outer` 会先把两个输入都展平成 1 维再计算。
- `plt.arrow` 不加 `length_includes_head=True` 时，箭头尖会超出向量真实终点。
- 画几何图务必 `plt.axis('square')`，否则横纵比例失真、角度看起来不对。
- `plt.axis('square')` 与 `plt.axis([...范围])` 是两次独立调用，范围以后一次为准。
- `savefig` 要在 `show` 之前调用，某些环境下 show 后画布会被清空。
- `matplotlib_inline` 只在 Jupyter 中可用，普通 .py 脚本运行时要去掉相关两行。
- `v ** 2` 是逐元素平方，NumPy 里对整组数据运算不需要写 for 循环。
- 单位向量 = 向量除以自身范数；零向量范数为 0，除法会得 nan 并告警（不报错），需先判断零向量。
- 比较两个浮点结果是否相等要用 `np.isclose(a, b)`，直接用 `==` 可能因精度误差误判。
- 随机测试要 `np.random.seed(固定值)` 才能保证结果可复现。
- `np.zeros()` 默认创建 float64 数组，要保留原类型必须传 `dtype=原数组.dtype`。
- `shape[0]` 是行数、`shape[1]` 是列数；二维数组索引 `a[i, j]` 是第 i 行第 j 列。
- `np.dot(v, v)` 等于范数的平方，但浮点结果末位可能有微小差异，比较用 `np.isclose()`。
- 任意幅度向量 = 目标幅度 × 单位向量，即 `m * v / norm(v)`；对零向量同样会得到 nan。
