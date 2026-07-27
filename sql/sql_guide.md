# SQL 使用指南

> 基于 MySQL 实践操作整理，分为"基础知识及修改表"与"约束"两大部分。

---

# 一、基础知识及修改表

## 1.1 数据库操作

**创建数据库（指定字符集）**
```sql
create database test default character set utf8;
```
创建名为 test 的数据库，默认字符集 utf8。

**查看所有数据库**
```sql
show databases;
```
列出当前 MySQL 实例中所有数据库。

**删除数据库**
```sql
drop database test;
```
删除指定数据库及其所有内容，不可恢复。

**切换/使用数据库**
```sql
use bjsxt;
```
将当前会话的默认数据库切换为 bjsxt，后续操作均在此库下执行。

## 1.2 表的基本操作

**创建表**
```sql
create table employees(employee_id int, employee_name varchar(10), salary float(8,2));
```
在当前库中创建表，定义列名及数据类型。varchar(n) 为变长字符串，float(m,d) 为浮点数（总位数 m，小数位 d）。

**查看当前库中所有表**
```sql
show tables;
```

**删除表**
```sql
drop table employees;
```
删除指定表的结构和数据。

**表重命名**
```sql
alter table employees rename emp;
```
将表 employees 重命名为 emp。

## 1.3 列的修改操作

**修改列名及类型（change column）**
```sql
alter table emp change column employee_name name varchar(10);
```
将列 employee_name 改名为 name，同时重新指定类型。change column 必须同时写旧列名、新列名和完整类型定义。

**修改列类型（modify）**
```sql
alter table emp modify name varchar(40);
```
仅修改列的数据类型，不改列名。

**添加列**
```sql
alter table emp add column commission_pct float(4,2);
```
在表末尾新增一列。

**删除列**
```sql
alter table emp drop column commission_pct;
```
删除指定列及其所有数据。

## 1.4 查看表结构信息

**查看索引/键信息**
```sql
show keys from depts;
```
显示表上所有索引信息，包括 PRIMARY KEY 和 UNIQUE 约束对应的索引。注意：NOT NULL 是列级约束，不产生索引，因此不会出现在此输出中。

**查看列定义（含 NOT NULL 信息）**
```sql
desc depts;
-- 或
show columns from depts;
```
输出中 `Null` 列显示 `NO` 即表示该列有非空约束。

---

# 二、约束（Constraints）

约束是施加在表数据上的规则，用于保证数据的完整性和准确性。

## 2.1 约束类型概览

| 约束类型 | 关键字 | 核心规则 |
|---------|--------|---------|
| 主键约束 | PRIMARY KEY | 不可为空、不可重复，保障唯一性，一表仅一个 |
| 外键约束 | FOREIGN KEY | 允许空值和重复，非空值必须参照表中已有，保障一致性 |
| 唯一性约束 | UNIQUE | 非空值不可重复，允许多个 NULL，一表可多个 |
| 非空约束 | NOT NULL | 列中不允许出现空值 |
| 检查约束 | CHECK | 自定义条件，MySQL 8.0.16 前不生效 |

## 2.2 主键约束（PRIMARY KEY）

**概念：**
- 一般每张表都要求设置一个主键，保障每条记录的唯一性。
- 单一主键：由一列充当主键。
- 联合主键：由多列组合充当主键，组合值必须唯一。
- 主键自增长（AUTO_INCREMENT）：仅适用于单一主键（或唯一键）列，且必须为整数类型（int）。插入数据时无需手动指定该列值，MySQL 自动递增分配。

**为已有表添加主键约束**
```sql
alter table emp add primary key(employee_id);
```

**设置主键自增长**
```sql
alter table emp modify employee_id int auto_increment;
```
前提：该列已是主键且为 int 类型。

**去掉自增长（删除主键前的必要步骤）**
```sql
alter table emp modify employee_id int;
```
将列改回普通 int，移除 auto_increment 属性。

**删除主键约束**
```sql
alter table emp drop primary key;
```
注意：若主键列带有 auto_increment，必须先去掉自增长再 drop primary key，否则会报错。

## 2.3 外键约束（FOREIGN KEY）

**概念：**
- 外键一般参照另一张表的主键列。
- 添加外键时需自定义约束名，命名惯例为"当前表名_fk"（如 emp_fk）。
- 一张表允许有多个外键约束（分别参照不同的表或列）。

**添加外键约束**
```sql
alter table emp add constraint emp_fk foreign key(dept_id) references departments(department_id);
```
语法：`add constraint 约束名 foreign key(本表列) references 参照表(参照列)`。

**删除外键约束**
```sql
alter table emp drop foreign key emp_fk;
```
通过约束名删除外键。删除后该列仍保留，只是不再受外键约束限制。

## 2.4 唯一性约束（UNIQUE）

**概念：**
- 添加唯一性约束时需自定义约束名，命名惯例为"表名_uk"（如 emp_uk）。
- 在 Navicat 中可通过"索引"面板查看唯一性约束。
- 允许存在多个 NULL 值，但一旦有非空值则不允许重复。

**添加唯一性约束**
```sql
alter table emp add constraint emp_uk unique(name);
```
语法：`add constraint 约束名 unique(列名)`。

**删除唯一性约束**
```sql
alter table emp drop key emp_uk;
```
注意：删除唯一性约束用的是 `drop key`（而非 drop constraint 或 drop unique），因为 MySQL 中 UNIQUE 本质上是通过唯一索引实现的。

## 2.5 非空约束（NOT NULL）

**概念：**
- 列中不允许出现空值，插入或更新时必须提供有效值。
- 不能对已经存在空值的列添加非空约束（需先处理已有 NULL 数据）。

**添加非空约束**
```sql
alter table emp modify salary float(8,2) not null;
```
语法：`modify 列名 类型 not null`，在列定义中直接追加 not null。

**去掉非空约束**
```sql
alter table emp modify salary float(8,2) null;
```
将列改回允许空值。

## 2.6 检查约束（CHECK）

用户自定义的条件约束，插入/更新的数据必须满足该条件。MySQL 8.0.16 之前的版本不支持 CHECK 约束（语法会被解析但被忽略），8.0.16 起才真正生效。

## 2.7 创建表时直接定义约束

建表时可以在列定义中直接内联约束，无需事后再 alter：
```sql
create table depts(
    department_id int primary key auto_increment,
    department_name varchar(30) unique,
    location_id int not null
);
```
此写法一次性完成主键+自增长、唯一性、非空三种约束的定义，适合新建表时使用。

---

# 三、DML 语言（数据操作）

DML（Data Manipulation Language）用于操作表中的数据：添加（INSERT）、更新（UPDATE）、删除（DELETE）。

## 3.1 插入数据（INSERT）

两种方式：
- **选择插入**：指定要插入的列名，未指定的列取默认值（或 NULL）。
- **完全插入**：按表定义顺序为每一列提供值，不可省略任何列。

完全插入时，若主键为 auto_increment，可用 `default`、`null` 或 `0` 占位，MySQL 会自动分配递增值。

**选择插入（指定列）**
```sql
insert into departments(department_name, location_id) values("market", 1);
```
只插入指定的列，department_id 由自增长自动填充。

**完全插入（default 占位）**
```sql
insert into departments values(default, "development", 2);
```

**完全插入（null 占位）**
```sql
insert into departments values(null, "human", 3);
```

**完全插入（0 占位）**
```sql
insert into departments values(0, "teaching", 4);
```
三种占位方式效果相同，均触发自增长分配。

## 3.2 列默认值与 default 占位

建表或添加列时可通过 `default` 关键字设定列的默认值。插入数据时，若某列使用 `default` 占位（或选择插入时省略该列），MySQL 自动填入预设的默认值。

**建表时设置列默认值**
```sql
create table emp3(
    emp_id int primary key auto_increment,
    name varchar(10),
    address varchar(50) default "unknown"
);
```

**添加列时设置默认值**
```sql
alter table emp3 add column job_id int default 0;
```

**选择插入（省略的列自动取默认值）**
```sql
insert into emp3(name) values("admin");
```
emp_id 自增长，address 取 "unknown"，job_id 取 0。

**完全插入（多列用 default 占位）**
```sql
insert into emp3 values(default, "oldlu", default, default);
```
所有需要默认值的列统一用 `default` 占位，MySQL 分别填入各自的预设值。

## 3.3 更新数据（UPDATE）

重要原则：UPDATE 必须带 `where` 条件，否则表中所有行都会被更新。

**更新单列**
```sql
update emp3 set address = "Beijing" where emp_id = 1;
```

**同时更新多列（逗号分隔）**
```sql
update emp3 set address = "Beijing", job_id = 2 where emp_id = 1;
```
set 后面用逗号分隔多个"列 = 值"赋值对。若新值与原值相同，则 Changed 为 0（不产生实际修改）。

## 3.4 删除数据（DELETE / TRUNCATE）

重要原则：DELETE 不带 `where` 条件会删除表中所有数据（逐行删除）。

**带条件删除**
```sql
delete from emp3 where emp_id = 1;
```

**清空整张表（TRUNCATE）**
```sql
truncate table emp3;
```

**DELETE 与 TRUNCATE 的区别：**

| 对比项 | DELETE | TRUNCATE |
|--------|--------|----------|
| 速度 | 逐行删除，较慢 | 整体删除，较快 |
| 日志 | 写服务器 log（可回滚） | 不写 log（不可回滚） |
| 自增值 | 继续累加 | 重置为初始值 |
| 条件 | 可带 where 删除部分行 | 只能清空全表 |

---

# 附录：常用关键字与语法速查

## A. 操作动词（按功能分类）

| 功能 | 关键字 | 示例片段 |
|------|--------|---------|
| 创建 | `create` | create database / create table |
| 删除结构 | `drop` | drop database / drop table / drop column / drop primary key / drop foreign key / drop key |
| 修改结构 | `alter` | alter table ... rename / add / modify / change / drop |
| 插入数据 | `insert` | insert into 表(列) values(值) |
| 更新数据 | `update` | update 表 set 列=值 where 条件 |
| 删除数据 | `delete` | delete from 表 where 条件 |
| 清空表 | `truncate` | truncate table 表名 |
| 查看 | `show` | show databases / show tables / show keys from 表名 |
| 查看列定义 | `desc` | desc 表名（等价于 show columns from） |
| 切换 | `use` | use 数据库名 |

## B. ALTER TABLE 子操作对照

| 目的 | 语法模式 |
|------|---------|
| 重命名表 | `alter table 旧名 rename 新名` |
| 添加列 | `alter table 表 add column 列名 类型` |
| 删除列 | `alter table 表 drop column 列名` |
| 改列名+类型 | `alter table 表 change column 旧列名 新列名 类型` |
| 只改类型/属性 | `alter table 表 modify 列名 新类型 [约束]` |
| 添加主键 | `alter table 表 add primary key(列名)` |
| 删除主键 | `alter table 表 drop primary key` |
| 添加外键 | `alter table 表 add constraint 名 foreign key(列) references 表(列)` |
| 删除外键 | `alter table 表 drop foreign key 约束名` |
| 添加唯一约束 | `alter table 表 add constraint 名 unique(列)` |
| 删除唯一约束 | `alter table 表 drop key 约束名` |

## C. 常用数据类型

| 类型 | 说明 |
|------|------|
| `int` | 整数（4字节） |
| `float(m,d)` | 浮点数，m=总位数，d=小数位数 |
| `varchar(n)` | 变长字符串，最大 n 个字符 |

## D. 约束相关关键字

| 关键字 | 含义 |
|--------|------|
| `primary key` | 主键，唯一+非空 |
| `auto_increment` | 自增长，仅限 int 类型的主键/唯一列 |
| `foreign key ... references` | 外键，参照另一张表的列 |
| `unique` | 唯一性，非空值不可重复 |
| `not null` / `null` | 非空 / 允许空值 |
| `constraint 名` | 为约束自定义名称（外键、唯一约束需要） |
| `default character set` | 指定数据库/表的默认字符集 |
| `default` | 列默认值 / 插入时占位触发默认值或自增长 |
| `insert into ... values` | 插入数据 |
| `update ... set ... where` | 按条件更新数据 |
| `delete from ... where` | 按条件删除数据 |
| `truncate table` | 清空全表（不可回滚，重置自增） |

## E. 约束命名惯例

| 约束类型 | 命名格式 | 示例 |
|---------|---------|------|
| 外键 | 表名_fk | emp_fk |
| 唯一 | 表名_uk | emp_uk |
| 主键 | 系统自动命名 PRIMARY | — |

## F. 易错点备忘

- `varchar` 不是 `vaechar`；`table` 不是 `talbe`；`column` 不是 `colunm`——拼写错误是 MySQL 报错最常见原因。
- 删除唯一约束用 `drop key`，不是 `drop constraint`。
- 删除主键前必须先去掉 auto_increment。
- `show keys` 看不到 NOT NULL，要用 `desc 表名`。
- 每条语句必须以分号 `;` 结尾。
- UPDATE / DELETE 忘写 `where` 会作用于全表所有行。
- 完全插入时列数必须与表定义一致，否则报 "Column count doesn't match"。
- TRUNCATE 不可回滚，操作前确认是否需要保留数据。
