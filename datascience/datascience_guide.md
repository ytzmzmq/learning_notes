# 数据科学实践指南

> 基于《数据科学实践与Python应用》第 2 章"向量"的学习实践整理（已完结：正文 + 练习 2-1 至 2-10），分为"NumPy 数组与向量运算"与"matplotlib 向量的几何可视化"两大部分，文末附第 2 章练习汇总。运行环境：numpy 2.3、matplotlib 3.10。

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

## 1.15 案例：点积的交换律（练习 7）

点积是可交换的（commutative）：a·b = b·a，即 aTb = bTa。

**一维向量**直接用 `np.dot` 验证，两种顺序的差为 0：

```python
v1 = np.random.randn(5)
v2 = np.random.randn(5)
print(np.dot(v1, v2) - np.dot(v2, v1))   # 0.0
```

**列向量 (5,1)** 不能直接 `np.dot`——矩阵乘法要求前者的列数等于后者的行数。此时用 Hadamard 乘法逐元素相乘再求和来计算点积：

```python
u1 = np.random.randn(5, 1)   # 5 行 1 列的列向量
u2 = np.random.randn(5, 1)
aTb = np.sum(u1 * u2)        # Hadamard 乘法 + 求和
bTa = np.sum(u2 * u1)
print(aTb - bTa)             # 0.0
```

原理：点积就是对应元素乘积的和，加法满足交换律，所以 aTb = Σaᵢbᵢ = Σbᵢaᵢ = bTa。

## 1.16 案例：向量的垂直投影（练习 8）

问题：已知向量 a 和 b，在 a 方向上找一点 βa，使它与 b 最接近。思路：b−βa 与 a 垂直（点积为 0），即 a·b − β(a·a) = 0，解得：

**β = a·b / a·a**

```python
a = np.array([1, 2])
b = np.array([3, 1])

beta = np.dot(a, b) / np.dot(a, a)   # 1.0
proj_point = beta * a                # [1. 2.]，投影点
c = b - beta * a                     # [ 2. -1.]，b 到投影点的垂直分量
```
几何图（a、b、b−βa 三个箭头加垂线虚线）见 2.5 的绘图技巧。

## 1.17 案例：正交分解（练习 9）

把任意向量 t 沿 r 方向分解为平行分量与垂直分量：

- 平行分量：t∥r = βr，其中 β = t·r / r·r（与 1.16 的投影同理）
- 垂直分量：t⊥r = t − t∥r

```python
np.random.seed(33)
t = np.random.randn(2)      # [-0.31885351 -1.60298056]
r = np.random.randn(2)      # [-1.53521787 -0.57040089]

beta = np.dot(t, r) / np.dot(r, r)   # 0.5233854305606638
t_para_r = beta * r                  # [-0.80351067 -0.29853952]
t_perp_r = t - t_para_r              # [ 0.48465716 -1.30444104]
```

两项验证：

```python
# 验证 1：两个分量的和还原为原向量 t
print(np.allclose(t, t_para_r + t_perp_r))   # True

# 验证 2：两分量相互正交（点积接近机器精度 0）
print(np.dot(t_para_r, t_perp_r))            # -5.55e-17
```

## 1.18 案例：完整性检验（练习 2-10）

这道题的核心启发：**"程序没有报错"不等于"数学逻辑正确"**。要设计基于理论几何约束（垂直度、尺度不变性等）的测试用例（Sanity-check）来保障算法严谨性。

**题目中的 Bug**：正交分解的正确公式分母是参考向量自身的点积 r·r：

β = t·r / r·r

手写时容易把分母错写成目标向量自身的点积 t·t：

β_bug = t·r / t·t

**陷阱：加和检验会失效。** 很多人以为检查 t∥ + t⊥ = t 就够了——不行！因为代码中 t⊥ 是用 `t - t_para` 算出来的，无论 beta 多么离谱，代数恒等式 t∥ + (t − t∥) ≡ t 永远成立，加和检验永远发现不了这个 Bug。

**有效的完整性检验：**

1. **正交性检验**（最直接）：两个分量必须垂直，t∥·t⊥ = 0（且 t⊥·r = 0）。分母写错时点积显著非 0。
2. **尺度缩放检验**：投影只取决于 r 的方向、与 r 的长度无关——把 r 放大为 2r，投影向量应保持不变。错误公式下投影会跟着放大（β_bug 变 2 倍 → t∥ 变 2 倍）。

完整解答代码（已验证）：

```python
import numpy as np

# 1. 生成两个随机向量
np.random.seed(42)
t = np.random.randn(3)
r = np.random.randn(3)

# 2. 正确的实现
beta_correct = np.dot(t, r) / np.dot(r, r)  # 正确：分母是 r·r
t_para_correct = beta_correct * r
t_perp_correct = t - t_para_correct

# 3. 缺陷实现（Bug）
beta_bug = np.dot(t, r) / np.dot(t, t)  # 错误：分母写成了 t·t
t_para_bug = beta_bug * r
t_perp_bug = t - t_para_bug

# 检验 1：分量之和是否等于 t（陷阱检验——两种实现误差都是 0，无法发现 Bug）
print(np.linalg.norm(t - (t_para_correct + t_perp_correct)))   # 0.0
print(np.linalg.norm(t - (t_para_bug + t_perp_bug)))           # 0.0

# 检验 2：两分量的点积是否为 0（正交性检验，能捕获 Bug）
print(np.dot(t_para_correct, t_perp_correct))   # 8.67e-19，机器精度内为 0
print(np.dot(t_para_bug, t_perp_bug))           # -1.5077，显著不为 0！

# 检验 3：t_perp 是否垂直于参考向量 r
print(np.dot(t_perp_correct, r))   # 1.73e-17，垂直
print(np.dot(t_perp_bug, r))       # -1.6215，不垂直
```

尺度不变性验证：把 r 换成 2r 后，正确实现的投影不变；缺陷实现的投影直接放大（实测 β_bug·2r ≈ [5.66, -0.87, -0.87]）。

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

## 2.5 中文与标注绘图技巧（练习 8-9 的图像）

**显示中文与负号**：matplotlib 默认字体不含中文，需设为 Windows 自带黑体（SimHei）；负号显示为方块时关闭 unicode_minus：

```python
plt.rcParams['font.sans-serif'] = ['SimHei']   # 全局中文字体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示坐标轴负号 '-'
```

**用 plt.plot 画线段**：`plt.plot([x1, x2], [y1, y2], 'k--', linewidth=1.2)` 画从 (x1,y1) 到 (x2,y2) 的黑色虚线——第一个列表是 x 序列、第二个是 y 序列。练习 8 用它画 b 到投影点的垂线，练习 9 用粗虚线画出两个分解分量。

**用 plt.text 在坐标处标注文字**：

```python
plt.text(a[0]+.1, a[1], 'a', fontsize=18)            # 在向量端点旁标 a
plt.text(x, y, r'(b-βa)', fontsize=18)               # r 前缀保留原始字符
```

**图例混用与取 [0]**：`plt.arrow` 返回箭头对象，但 `plt.plot` 返回的是**列表**，放进 legend 前要取 `[0]`：

```python
c = plt.plot([0, t_para_r[0]], [0, t_para_r[1]], 'k--', linewidth=3)
d = plt.plot([0, t_perp_r[0]], [0, t_perp_r[1]], 'k--', linewidth=3)
plt.legend([a, b, c[0], d[0]], [r'$t$', r'$r$', r'$t_{\parallel r}$', r'$t_{\perp r}$'])
```
图例标签支持 LaTeX 风格（如 `r'$t_{\parallel r}$'` 显示下标）。

**保存到子目录**：`plt.savefig('figures/练习2_8.png', dpi=300)` 可存入子目录，但 `figures/` 目录必须已存在，否则报错。

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
| 整体比较 | `np.allclose(a, b)` | 逐元素 isclose 后取与 |
| 垂直投影 | `β = np.dot(a,b)/np.dot(a,a)` | 投影点为 `β*a` |
| 正交分解 | `t∥ = βr`，`t⊥ = t − t∥` | t∥⊥t⊥，和为 t |
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
| `plt.rcParams['font.sans-serif']=['SimHei']` | 中文字体（黑体） |
| `plt.rcParams['axes.unicode_minus']=False` | 正常显示负号 |
| `plt.plot([x1,x2],[y1,y2],'k--')` | 画线段/虚线，返回列表 |
| `plt.text(x, y, '标签', fontsize=n)` | 在坐标处标注文字 |

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
- `np.dot` 不能直接用于两个同形状列向量 `(n,1)`（矩阵乘法维度不匹配），要用 Hadamard 乘法 + `np.sum` 计算点积。
- `plt.plot` 的返回值是列表，传给 `plt.legend` 前要取 `[0]`；`plt.arrow` 返回单个对象不用取。
- `savefig` 存到子目录（如 `figures/xx.png`）时，该目录必须已存在，否则报 FileNotFoundError。
- 中文标注要设 `font.sans-serif=['SimHei']`，坐标轴负号变方块时加 `axes.unicode_minus=False`。
- 正交分量的点积是接近机器精度的极小数（如 -5.55e-17），不是精确的 0，判断正交用 `np.allclose`/阈值比较。
- 完整性检验要选独立的理论约束：由 `t − t∥` 算出的 t⊥ 做加和检验恒成立（假阳性），正交性、尺度不变性这类检验才能真正捕获公式 Bug。

---

# 第 2 章练习汇总（练习 2-1 至 2-10）

## 练习 2-1：向量加法与减法的几何效果

> 创建图 2-2 的代码，其中向量 v 为 (1,2)，向量 w 为 (4,-6)，分别展示 v+w 和 v−w 的几何效果。

答案要点与代码见 2.4；v+w=(5,−4)、v−w=(−3,8)，成图见 `练习2_1a.png` / `练习2_1b.png`。

## 练习 2-2：自定义向量范数算法

> 写一个计算向量范数的算法（元素平方求和再开方）。使用随机的向量和不同的维数与排列方式验证，结果都应与 `np.linalg.norm()` 相同。

答案要点与代码见 1.10：`np.sqrt(np.sum(v ** 2))`，配 `np.random.seed` + `np.isclose` 做 1~100 维批量验证。

## 练习 2-3：单位向量

> 写一个函数，输入一个向量，输出与该向量方向相同的单位向量。若输入零向量会如何？

答案要点与代码见 1.11：`v / np.linalg.norm(v)`，单位向量元素平方和为 1；零向量除法返回全 nan 并发 RuntimeWarning。

## 练习 2-4：任意幅度的向量

> 写一个函数，输入一个向量和一个要求的幅度，输出方向与输入向量相同、但幅度为给定幅度的向量。

答案要点与代码见 1.12：`magnitude * vector / norm`（例：[0,3,4] 取幅度 10 → [0,6,8]）。

## 练习 2-5：for 循环实现转置

> 写一个 for 循环将行向量转换为列向量，不要使用 `np.transpose()` 或 `v.T`。

答案要点与代码见 1.13：`shape[1]` 取列数、`np.zeros((n,1), dtype=原类型)` 建目标数组、循环内 `colVec[i, 0] = rowVec[0, i]`。

## 练习 2-6：向量与自身的点积 = 范数的平方

> 可以使用向量与其自身的点积来计算范数的平方，用 Python 确认。

答案要点与代码见 1.14：`np.dot(v, v)` 与 `np.linalg.norm(v) ** 2` 在浮点精度内相等。

## 练习 2-7：验证点积满足交换律

> 演示点积是可交换的（aTb = bTa），并理解为什么。

答案要点与代码见 1.15：一维用 `np.dot` 直接对比；列向量 (n,1) 不能直接 `np.dot`，用 Hadamard 乘法 + `np.sum`。原理：点积是对应元素乘积的和，加法可交换。

## 练习 2-8：向量的垂直投影

> 已知向量 a 和 b，在向量 a 方向上找到一点使其与向量 b 最接近，并作图。

答案要点与代码见 1.16（数学）与 2.5（作图）：β = a·b / a·a，投影点 βa，垂直分量 b − βa ⊥ a。

## 练习 2-9：正交分解

> 从两个随机给定的数值向量 t 和 r 开始，重复图 2-8 的过程；验证两个分量的和是 t，且 t∥r 与 t⊥r 相互正交。

答案要点与代码见 1.17：t∥r = βr（β = t·r / r·r），t⊥r = t − t∥r；用 `np.allclose` 验证加和、点积 ≈ 0（机器精度）验证正交。

## 练习 2-10：完整性检验（Sanity-checking）

> 把练习 2-9 中 β 的分母故意写错（`np.dot(t, t)` 而非 `np.dot(r, r)`），然后思考：用什么检验代码能一眼看出结果错了？

核心结论见 1.18：
- **陷阱**：加和检验 t∥+t⊥=t 永远通过（代数恒等式），发现不了 Bug。
- **有效检验 1（正交性）**：t∥·t⊥ 与 t⊥·r 在正确代码下 ≈ 0（机器精度），缺陷代码下显著非 0（实测 -1.5077 / -1.6215）。
- **有效检验 2（尺度不变性）**：投影只与 r 的方向有关，与 r 长度无关——r 放大 2 倍投影应不变；缺陷代码的投影随之放大。
- 完整解答代码见 1.18。核心启发：编写科学计算/机器学习代码时，"程序没有报错"≠"数学逻辑正确"，要用基于理论几何约束（垂直度、尺度不变性、范数不变性等）的测试用例保障严谨性。
