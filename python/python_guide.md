# Python 基础语法指南

> 基于 Python 学习实践持续整理，涵盖入门第 3 章"数据类型与运算符"（01-18 全部小节）及字符串操作与正则表达式部分，分为"基本语法元素"、"数据类型"、"数据类型转换"、"运算符"、"输入语句"、"字符串操作"与"正则表达式"七大部分。

---

# 一、基本语法元素

## 1.1 字面量

字面量是直接写在代码中的值，不需要任何计算就能被识别。字符串字面量必须用引号包起来：可以是单引号、双引号、三个单引号或三个双引号。

```python
'字符串必须引号包起来'
"可以是单引号、双引号、三个单引号或双引号"
```

引号必须成对出现，否则语法报错。单引号和双引号在功能上完全等价。

## 1.2 变量

变量用于存储数据，通过赋值符号 `=` 把值绑定到名字上，之后可以通过变量名引用数据。

```python
name = '张三'
age = 18
weight = 64.2
print(name, '的年龄是', age, '体重是', weight)
```

**print 输出数据的两种方式**
```python
print("张三的体重是65.2kg")        # 内容全部写死在字符串里，值固定
print("张三的体重是", 65.2, "kg")  # 逗号分隔多个值，可以混入变量
```
逗号方式让 print 依次输出多个值，字符串与数字可以混合出现；而加号拼接只允许字符串之间进行（见 2.4）。

## 1.3 标识符的命名规则

标识符是变量、函数等对象的名字，规则如下：

- 只能由数字、字母、下划线三种字符组成。
- 区分大小写（`name` 与 `Name` 是两个不同的标识符）。
- 不能以数字开头（`1name` 非法，`name1` 合法）。
- 不能使用 Python 关键字（如 `and`、`if`、`class` 等）。
- 尽量不要与内置函数重名（如 `print`、`type`、`all`），否则会覆盖内置函数。

## 1.4 常量

Python 没有真正的常量机制，约定俗成用**全大写的变量名**表示常量，本质仍是变量，只是约定不去修改它。

```python
ADULT_AGE = 18        # 成年年龄
MONTH_IN_YEAR = 12    # 一年月份数
MAX_USERS = 1000      # 最大用户数

print(ADULT_AGE, MONTH_IN_YEAR, MAX_USERS)
```

## 1.5 注释与字符编码

注释分为单行注释、多行注释和文件编码注释三类。

**单行注释（#）**
```python
# name是名字
name = '张三'
print(name)  # print是打印
```
单行注释不能换行，但可以连续写很多个单行注释。

**多行注释（三引号，约定俗成）**
```python
"""
这段字符串在代码执行时被忽略了
可以不太严谨地认为是多行注释
"""
```
用 3 个单引号或 3 个双引号编写多行注释是约定俗成的方法，本质是一个字符串。三个引号的形式还常被用作文档字符串（docstring）。

**文件编码注释**
```python
# coding=utf-8
```
Python 3 默认使用 UTF-8 编码，所以通常不需要写文件编码注释。

---

# 二、数据类型

## 2.1 查看数据类型（type）

使用内置函数 `type()` 查看任意值的数据类型：

```python
print(type("张三"))  # <class 'str'>
print(type(18))      # <class 'int'>
print(type('18'))    # <class 'str'>
print("16.77的类型是", type(16.77))  # <class 'float'>
```

注意：`18` 是整数，`'18'` 是字符串——数字一旦加上引号就不再是数字，不能参与算术运算。

## 2.2 整型（int）

**下划线分组：** 数字很大时，可以用下划线将数字分组，方便阅读，下划线不影响数值本身。

```python
salary = 500_000
print(salary)  # 500000
```

**整数上限：** 整数的上限值取决于执行代码的计算机内存和处理能力，Python 本身不限制整数大小。

```python
import sys

a = 9 ** 9999
b = a + 100
sys.set_int_max_str_digits(0)  # 取消大整数转字符串的位数限制
print(b)
```
超大整数在打印（转字符串）时默认有位数上限，可用 `sys.set_int_max_str_digits(0)` 取消限制。

## 2.3 浮点型（float）

浮点型指带有小数点的数字。

```python
price = 120.0
print(type(price))  # <class 'float'>
```

**科学计数法表示：**

```python
speed_of_sound = 3.4e+2  # 340.0，加号可以省略，e 可以大写为 E
one_ml = 1E-3            # 0.001
print(speed_of_sound, one_ml)
```
`e`/`E` 后面的数字表示 10 的指数次幂。

## 2.4 字符串（str）

### 2.4.1 定义方式

字符串可以使用 `' '`、`" "`、`''' '''`、`""" """` 四种引号定义。单引号和双引号完全等价，**不能换行**；三引号**可以换行**书写多行内容。

```python
message1 = ("XXXX,"
            "XXXXX")   # 单行字符串可以用圆括号拆成多行书写，相邻字符串自动拼接
message2 = '''
XXXX,
XXXXX'''               # 三引号直接保留换行
```

### 2.4.2 格式化输出

三种写法：

```python
name = '张三'
age = 18
weight = 65.42
gender = 'male'

# 写法1：加号拼接，只能字符串之间拼接，整数、浮点数不能直接与字符串拼接
info1 = '姓名是' + name + '，性别是' + gender

# 写法2：使用 % 占位符
info2 = '姓名是%s,性别是%s,体重是%f,年龄是%i' % (name, gender, weight, age)

# 写法3：使用 f-string（最推荐）
info3 = f'姓名是{name},性别是{gender},体重是{weight},年龄是{age}'
```

要点：
- `%s` 的占位是"万能"的，任何类型都能放。
- `%f` 默认输出 6 位小数（65.42 → `65.420000`）。
- f-string 直接把变量写进 `{}` 中，最直观高效。

### 2.4.3 占位符的精度控制

`%s`：`%宽度.精度s`，正负号代表左右对齐，宽度不够用空格补齐；精度表示最多取几个字符。

```python
'姓名是%.3s' % name   # 最多取 3 个字符
```

`%m.nf`：宽度 m、精度 n（n 默认 6），小数位不够补 0，超出部分四舍五入截断。

```python
'体重是%.1f' % weight   # 65.4
'%6.4i' % 18            # '  0018'：先补 0 到 4 位，再按宽度 6 右对齐补空格
```

### 2.4.4 转义字符

转义字符通常以反斜杠 `\` 开头：

| 转义字符 | 作用 |
|---------|------|
| `\'` | 输出单引号 `'` |
| `\"` | 输出双引号 `"` |
| `\n` | 换行 |
| `\\` | 输出反斜杠 `\` |
| `\b` | 删除（回退）前一个字符 |
| `\r` | 光标回到本行开头，覆盖输出 |
| `\t` | 光标跳转到下一制表位 |

```python
print('可以用\'输出,\n可以用\"输出"')
print("D:\\nice")          # 输出 D:\nice
print('张三\t李四\t12')    # 制表位对齐
```

注意：
- 一个制表位的宽度因环境而异（PyCharm 下是 4 个字符位，cmd 是 8 位），可用字符串的 `expandtabs()` 方法限制。
- 中文通常占两个英文字符的位置，对齐时需留意。

## 2.5 布尔类型（bool）

布尔类型只有两个值：`True` 和 `False`，常用于表示比较运算和逻辑运算的结果。

```python
a = True
b = False
c = 5 > 3
print(type(a), a)   # <class 'bool'> True
print(type(b), b)   # <class 'bool'> False
print(type(c), c)   # <class 'bool'> True
```

**布尔类型是 int 类型的子类型：** 本质是用 1 表示 True、0 表示 False，因此布尔值可以直接参与算术运算。

```python
print(int(True), int(False))   # 1 0
print(4 + True)                # 5
```

**bool() 转换规则：**
- 数字：除 0 以外的所有数（整数、浮点数、正负均可）转为 True，0 转为 False。
- 字符串：除空字符串以外的任何字符串转为 True，空字符串 `''` 转为 False。

```python
print(bool(1), bool(0))          # True False
print(bool(34e6), bool(-42.4))   # True True
print(bool('hello'), bool(""))   # True False
```

## 2.6 进制表示与转换

**进制表示：** 整数可以用前缀 `0b`（二进制）、`0o`（八进制）、`0x`（十六进制）书写，输出时自动转为十进制显示。

```python
num1 = 0b110010
num2 = 0o1063
num3 = 0x1FD
print(num1, num2, num3)  # 50 563 509
print(str(num2))         # '563'
print(num3 > 123)        # True，可以和十进制数正常比较
```

**十进制转其他进制：** `bin()`、`oct()`、`hex()` 返回的都是**字符串**，分别带 `0b`、`0o`、`0x` 前缀。

```python
print(bin(28))   # 0b11100
print(oct(28))   # 0o34
print(hex(28))   # 0x1c
print(type(bin(28)))   # <class 'str'>
```

**其他进制转十进制：** 用 `int(字符串, 进制)`，第二个参数指明字符串的进制，结果是整型。

```python
print(int('0b11100', 2), int('0o34', 8), int('0x1c', 16))  # 28 28 28
```

---

# 三、数据类型转换

## 3.1 转换为字符串（str）

任何类型都可以用 `str()` 转成字符串类型：

```python
result1 = str(18)      # '18'
result2 = str(1.8e3)   # '1800.0'
```

## 3.2 转换为整型（int）

规则：
- 字符串转整型时，数字间不能有空格（字符串两侧可以有空格）。
- 数字 + 字符的组合不可以（如 `'12a'`）。
- 带小数点的字符串不可以（如 `'5.6'` 会报错，需先 `float()` 再 `int()`）。
- 浮点数转整型直接截断小数部分（不四舍五入）。

```python
result3 = int(' 79 ')   # 79，两侧空格允许
result4 = int(15.6)     # 15，截断小数
```

## 3.3 转换为浮点型（float）

规则：
- 允许整数、浮点数，以及字符串形式的整数、小数（字符串两侧可以有空格）。
- 小数内部不能有空格，不能有非数字字符。
- 不允许不合法的数字格式（如 `'5.13.21'` 报错）。

```python
result5 = float(18)       # 18.0
result6 = float(' 5.7 ')  # 5.7
```

## 3.4 转换为布尔型（bool）

任何类型都可以用 `bool()` 转成布尔型。核心规则：只有 `0`、`0.0`、空字符串 `''` 这类"空/零"值转为 False，其余一律 True（详见 2.5）。

```python
print(bool(34e6))     # True，除 0 外的数都是 True
print(bool(-42.4))    # True
print(bool('hello'))  # True，非空字符串
print(bool(""))       # False，空字符串
```

---

# 四、运算符

## 4.1 算术运算符

| 运算符 | 含义 | 示例 | 结果 |
|-------|------|------|------|
| `+` | 加 | `6 + 3` | 9 |
| `-` | 减 | `6 - 3` | 3 |
| `*` | 乘 | `6 * 3` | 18 |
| `/` | 除（结果一定是浮点型） | `6 / 3` | 2.0 |
| `//` | 取整除 | `25 // 4` | 6 |
| `%` | 取余 | `25 % 3` | 1 |
| `**` | 指数运算 | `3 ** 5` | 243 |

```python
print(6 / 3, 25 // 4, 25 % 3, 3 ** 5)  # 2.0 6 1 243
```

## 4.2 赋值运算符和复合赋值运算符

赋值运算符为 `=`，复合赋值运算符把运算和赋值合并为一步。

```python
age = 18
age = age + 1   # 普通写法
print(age)      # 19

age2 = 18
age2 += 1       # 加法赋值，等价于 age2 = age2 + 1
print(age2)     # 19

price = 20
age2 += price   # 后面也可以接变量，等价于 age2 = age2 + price
print(age2)     # 39
```

其他复合赋值运算符：

```python
age3 = 18
age3 -= 1          # 17

price2 = 20
price2 *= 0.8      # 16.0

price3 = 20
num = 5
price3 /= num      # 4.0（/ 的结果是浮点型）

all_num = 46
num = 3
all_num //= num    # 15

seconds = 432
minutes = 60
seconds %= minutes # 12（432 秒换算后余 12 秒）

a = 2
a **= 3            # 8
```

## 4.3 比较运算符

比较运算的结果是布尔值（True / False）。

**== 与 !=**

`==` 判断左右两侧是否相等，要求数据类型一致；类型不一致**不会报错**，直接返回 False。`!=` 判断是否不等。

```python
a, b, c, d = 5, 5, 6, "5"
print(a == b)   # True
print(a == c)   # False
print(a == d)   # False，int 与 str 类型不同，不报错直接判 False
print(a != b)   # False
print(a != d)   # True
```

**>、<、>=、<=**

左右两侧必须是同一类型，否则报错（TypeError）。

```python
print(a > c)    # False
print(a < c)    # True
print(a >= c)   # False
print(a <= c)   # True
# print(a > d)  # TypeError: '>' not supported between instances of 'int' and 'str'
```

**字符串间的比较**

比较运算符同样适用于字符串。规则：依次比较每个字符的 Unicode 编码，一旦区分出大小就不再比较后面的字符；前几位一致且一侧先用完时，长度更长的一方更大。

```python
msg1 = "ytzmzmq"
msg2 = 'ytzmzmj'
print(msg1 != msg2)        # True

print('abcdx' > "xabcd")   # False，首位 a < x 已分出结果
print("abcdef" > "abcd")   # True，前缀相同，长的大
```

内置函数 `ord()` 查看字符的 Unicode 编码，`chr()` 把编码转回字符：

```python
print(ord("a"))        # 97
print(ord('我'))       # 25105
print(chr(97))         # a
print(chr(25105))      # 我
```

## 4.4 逻辑运算符

`and`、`or`、`not` 三个逻辑运算符用于组合或取反条件。

**and：判断两侧是否都为 True**

```python
print(True and True)      # True
print(True and False)     # False
print(1 > 7 and 5 > 3)    # False
```

**or：判断两侧是否至少有一个为 True**

```python
print(True or False)      # True
print(False or False)     # False
print(1 > 7 or 5 > 3)     # True
```

**not：对单个值取反，结果一定是布尔值**

```python
print(not 3 > 2)     # False
print(not "你好")    # False（非空字符串转布尔为 True，再取反）
```

**逻辑短路：** and 左侧为假、or 左侧为真时，结果已确定，右侧不再执行（即使右侧会报错也不会触发）。

```python
print(False and 3 / 0)   # False，右侧不执行，不报除零错误
print(2 > 1 or 4 / 0)    # True，右侧不执行，不报除零错误
```

**返回的是参与计算的值本身：** and/or 的结果不一定是布尔值。非布尔值参与运算时先转布尔判断真假，但最终返回原值。规则：
- `x and y`：左侧为假返回左侧，否则返回右侧。
- `x or y`：左侧为真返回左侧，否则返回右侧。

```python
print(2 - 2 and 3 + 4)   # 0（左侧 0 为假，返回左侧 0）
print('' and 8 / 2)      # 空字符串（左侧为假，返回左侧）
print('a' and 3 * 2)     # 6（左侧为真，返回右侧）
print(2 - 2 or 3 + 4)    # 7（左侧为假，返回右侧）
print('你好' or 8 / 2)   # 你好（左侧为真，返回左侧）
```

---

# 五、输入语句

`input()` 用于从键盘接收用户输入。**返回结果永远是字符串类型**，即使输入的是数字。

```python
age = input()
print(type(age))                          # <class 'str'>
print(f'张三的年龄是{age}')
print(f'张三明年的年龄是{int(age) + 1}')   # 参与运算前要先 int() 转整型
```

**带提示信息：** 给 `input()` 传入一个字符串参数，会先打印提示再等待输入。

```python
name = input('请输入你的姓名：')
print(f'{name}你明年的年龄是{int(age) + 1}')
```

f-string 的花括号内可以放任何表达式（如 `int(age) + 1`），但字符串不能直接和整数相加，所以要先用 `int()` 转换（见第三部分）。

---

# 六、字符串操作

## 6.1 索引

用方括号取出字符串中的单个字符。正向索引从 0 开始；负数索引从末尾倒数，`-1` 是最后一个字符。

```python
name = "ytzmzmq ytjszmq"

a = name[0]    # 'y'，第一位
b = name[1]    # 't'
c = name[-1]   # 'q'，最后一位
```

## 6.2 切片

**基础切片 [start:end]：** 取索引 start 到 end 之间的字符。极其重要的法则是"包头不包尾"（左闭右开区间）——包含 start，不包含 end。

```python
d = name[3:7]   # 'mzmq'，即索引 3、4、5、6 四个字符
```

**步长切片 [::step]：** step 指每隔多少个位置取一个。

```python
e = name[::3]   # 'ymqtz'，每隔两个取一个
```

步长设为负数（如 `[::-1]`）时从右往左反向截取，常用于反转字符串：

```python
print(name[::-1])   # 'qmzsjt yqmzmzty'
```

**完整切片 [start:end:step]：** start:end 规定范围（依然包头不包尾），step 规定该范围内的步长。

```python
f = name[0:5:2]   # 'yzz'，索引 0-4 范围内隔一个取一个
```

## 6.3 长度

`len()` 返回字符串长度，从 1 开始计数（而索引从 0 开始）。

```python
print(len(name))   # 15
```

## 6.4 拼接与复制

```python
statement = name + ":ytjszmj"   # + 拼接字符串
name_3x = 3 * name              # * 把字符串复制 3 份
```

## 6.5 常用字符串方法

```python
a = name.upper()                            # 全部转为大写
b = name.replace("ytzmzmq", "杨韬太强了")    # 替换：旧字符串在前，新的在后
c = name.find("yt")                         # 查找：返回第一次出现的索引
d = name.find("YT")                         # -1，不存在（区分大小写）
```

**split()：** 在指定分隔符处分割字符串，返回列表。语法 `.split(分隔符, 最大分割次数)`，分隔符默认为空白字符，最大分割次数默认无限制。

```python
print(name.split())          # ['ytzmzmq', 'ytjszmq']，按空白分割
print(name.split('yt', 2))   # ['', 'zmzmq ', 'jszmq']，最多分 2 次
print(name.split("yt", 1))   # ['', 'zmzmq ytjszmq']，最多分 1 次
```

---

# 七、正则表达式

正则表达式是一种用于匹配和处理字符串的工具。Python 通过 `re` 模块提供 search、split、findall、sub 等功能。

## 7.1 导入模块与原始字符串

```python
import re

sentence_1 = "Yangtao is the most ambitious guy."
pattern = r"Yangtao"
print(pattern, type(pattern))   # Yangtao <class 'str'>
```
模式前的字母 `r` 代表 raw string（原始字符串）。只要模式里出现反斜杠 `\`（如 `\d`、`\w`、`\b` 等正则专属符号），就必须加 `r`（或者写双反斜杠）。

## 7.2 search()：查找第一个匹配

`search()` 扫描整个字符串，寻找第一个与 pattern 匹配的子串；`match.group()` 用于获取匹配到的部分。

```python
result = re.search(pattern, sentence_1)
if result:
    print("Yangtao found!", result.group())   # Yangtao found! Yangtao
else:
    print("Yangtao not found!")
```

## 7.3 特殊序列

| 序列 | 匹配内容 | 对应大写（取反） |
|------|---------|----------------|
| `\d` | 任何数字字符（0-9） | `\D` 任何非数字字符 |
| `\w` | 任何单词字符（a-z、A-Z、0-9、下划线） | `\W` 任何非单词字符 |
| `\s` | 任何空白字符（空格、制表符、换行等） | `\S` 任何非空白字符 |
| `\b` | 单词边界 | `\B` 非单词边界 |

模式里写几个符号就匹配几位，位数必须够：

```python
Sentence_2 = "Yangtao scores 430 in his exam."

match_2 = re.search(r"\d\d\d", Sentence_2)
print(match_2.group())   # '430'
# 若是 \d\d 则只输出 '43'；若是 \d\d\d\d 则匹配不到
```

## 7.4 findall()：查找所有匹配

`findall()` 找出字符串中所有符合模式的位置，返回列表。

```python
print(re.findall(r"\W", Sentence_2))   # [' ', ' ', ' ', ' ', ' ', '.']
```

## 7.5 re.split()：按正则分割

```python
print(re.split(r"\s", Sentence_2))   # ['Yangtao', 'scores', '430', 'in', 'his', 'exam.']
```

## 7.6 re.sub()：按正则替换

```python
pattern_4 = r"yangtao"
replacement = "Wang Zhetao"
new_string = re.sub(pattern_4, replacement, Sentence_2, flags=re.IGNORECASE)
new_string2 = re.sub(pattern_4, replacement, Sentence_2)
print(new_string)    # 'Wang Zhetao scores 430 in his exam.'
print(new_string2)   # 原句不变
```
`re.sub` 默认严格区分大小写，正则引擎只会寻找完全一模一样的词；传入 `flags=re.IGNORECASE` 才不区分大小写。

---

# 附录：常用关键字与语法速查

## A. 数据类型速查

| 类型 | 关键字 | 示例 |
|------|--------|------|
| 整型 | `int` | 18、500_000 |
| 浮点型 | `float` | 65.42、3.4e+2 |
| 字符串 | `str` | '张三'、"abc" |
| 布尔型 | `bool` | True、False |

查看类型：`type(值)`。

## B. 运算符一览

| 类别 | 运算符 |
|------|--------|
| 算术 | `+` `-` `*` `/` `//` `%` `**` |
| 赋值 | `=` |
| 复合赋值 | `+=` `-=` `*=` `/=` `//=` `%=` `**=` |
| 比较 | `==` `!=` `>` `<` `>=` `<=` |
| 逻辑 | `and` `or` `not` |

## C. 类型转换规则对照

| 函数 | 目标类型 | 允许 | 不允许 |
|------|---------|------|--------|
| `str()` | 字符串 | 任何类型 | — |
| `int()` | 整型 | 纯整数字符串（两侧可带空格）、浮点数（截断小数） | 数字间有空格、数字+字符、带小数点的字符串 |
| `float()` | 浮点型 | 整数、浮点数、字符串形式的整数/小数（两侧可带空格） | 小数内部有空格、含非数字、非法格式如 5.13.21 |
| `bool()` | 布尔型 | 任何类型（0/0.0/空字符串 → False，其余 → True） | — |

## D. 占位符速查

| 占位符 | 含义 |
|--------|------|
| `%s` | 万能占位，任何类型 |
| `%i` | 整数 |
| `%f` | 浮点数，默认 6 位小数 |
| `%宽度.精度s` | 控制宽度与字符个数，正负号控制对齐 |
| `%m.nf` | 宽度 m、小数位 n，不够补 0，四舍五入 |

## E. 转义字符速查

| 转义字符 | 作用 |
|---------|------|
| `\'` `\"` | 输出引号本身 |
| `\n` | 换行 |
| `\\` | 输出反斜杠 |
| `\b` | 删除前一个字符 |
| `\r` | 光标回到行首覆盖输出 |
| `\t` | 制表位 |

## F. 字符串操作速查

| 语法 / 方法 | 含义 |
|------------|------|
| `s[0]` / `s[-1]` | 正向索引 / 负向索引（-1 为末位） |
| `s[a:b]` | 切片，包头不包尾 |
| `s[a:b:n]` | 带步长切片 |
| `s[::-1]` | 反转字符串 |
| `len(s)` | 长度，从 1 计数 |
| `+` / `*` | 拼接 / 复制 |
| `s.upper()` | 转大写 |
| `s.replace(旧, 新)` | 替换 |
| `s.find(子串)` | 返回首次出现的索引，找不到返回 -1 |
| `s.split(分隔符, 最大次数)` | 分割返回列表 |

## G. 正则表达式速查

| 函数 / 语法 | 含义 |
|------------|------|
| `re.search(p, s)` | 找第一个匹配，返回 match 对象 |
| `m.group()` | 获取匹配到的内容 |
| `re.findall(p, s)` | 所有匹配，返回列表 |
| `re.split(p, s)` | 按模式分割 |
| `re.sub(p, 替换, s, flags=)` | 按模式替换 |
| `r"..."` | 原始字符串，模式含 `\` 时必须加 r |
| `flags=re.IGNORECASE` | 不区分大小写 |
| `\d` `\w` `\s` | 数字 / 单词字符 / 空白（大写取反） |

## H. 易错点备忘

- 字符串必须用成对的引号包裹，忘写或写错引号是最常见的语法错误。
- 标识符不能以数字开头，不能用关键字，避免与内置函数（如 `all`、`type`）重名。
- `+` 不能拼接字符串和整数/浮点数，要么先用 `str()` 转换，要么用 f-string。
- `'18'` 是字符串不是数字，不能参与算术运算。
- `int('5.6')`、`int('1 2')` 都会报 ValueError；`float('5.13.21')` 同理。
- `int(15.6)` 是截断小数得 15，不是四舍五入。
- `/` 的结果一定是浮点型（`6 / 3` 得 `2.0`）；`//` 才是取整除。
- 比较时 `==` 两侧类型不同不报错（直接 False），但 `>` `<` 类型不同会报 TypeError。
- Python 没有真正的常量，全大写命名只是约定，值依然可以被修改。
- 大整数打印前可能需要 `sys.set_int_max_str_digits(0)` 解除位数限制。
- 布尔是 int 的子类型，`True` 即 1、`False` 即 0，参与算术运算不会报错（如 `4 + True` 得 5），容易埋下隐患。
- `and`/`or` 返回的不一定是布尔值，而是参与计算的某个值本身；判断结果时别想当然。
- `and`/`or` 有短路特性，右侧可能根本不执行，依赖右侧副作用要小心。
- 二进制前缀是 `0b`、八进制 `0o`、十六进制 `0x`（是字母 o 不是数字 0）；`bin()`/`oct()`/`hex()` 返回的是**字符串**，要参与运算需先 `int(x, 进制)` 转回整数。
- `input()` 的返回值永远是字符串，直接和数字做算术会报 TypeError，必须先 `int()`/`float()` 转换。
- 切片"包头不包尾"：`s[3:7]` 不包含索引 7 的字符。
- 字符串索引从 0 开始，`len()` 从 1 计数，两者别混淆。
- `find()` 区分大小写，找不到返回 -1 而不是报错。
- `split()` 的第二参数是最大分割次数，剩余部分整体保留在最后一个元素里。
- 正则模式含反斜杠（`\d`、`\w` 等）必须加 `r` 前缀，否则反斜杠会被转义。
- `re.sub` 默认区分大小写，要忽略大小写需传 `flags=re.IGNORECASE`。
- 正则模式的位数必须够：`\d\d\d` 匹配不到四位数字。
