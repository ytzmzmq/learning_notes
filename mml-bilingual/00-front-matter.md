# 前言与序言（Foreword and Preface）

> [← 返回目录](README.md)

> Machine learning is the latest in a long line of attempts to distill human knowledge and reasoning into a form that is suitable for constructing machines and engineering automated systems. As machine learning becomes more ubiquitous and its software packages become easier to use, it is natural and desirable that the low-level technical details are abstracted away and hidden from the practitioner. However, this brings with it the danger that a practitioner becomes unaware of the design decisions and, hence, the limits of machine learning algorithms.

机器学习（machine learning）是将人类知识与推理提炼成适合构造机器和构建自动化系统的形式的一系列漫长尝试中的最新成果。随着机器学习变得无处不在、其软件包越来越易于使用，底层技术细节被抽象化并对从业者隐藏起来，这既是自然的，也是人们所期望的。然而，这也带来了风险：从业者可能因此觉察不到设计决策，进而觉察不到机器学习算法的局限性。

> The enthusiastic practitioner who is interested to learn more about the magic behind successful machine learning algorithms currently faces a daunting set of pre-requisite knowledge:

渴望进一步了解成功的机器学习算法背后“魔法”的热情从业者，目前面临着一系列令人生畏的预备知识：

- Programming languages and data analysis tools
- Large-scale computation and the associated frameworks
- Mathematics and statistics and how machine learning builds on it

- 编程语言与数据分析工具
- 大规模计算及其相关框架
- 数学与统计学，以及机器学习如何构建于其上

> At universities, introductory courses on machine learning tend to spend early parts of the course covering some of these pre-requisites. For historical reasons, courses in machine learning tend to be taught in the computer science department, where students are often trained in the first two areas of knowledge, but not so much in mathematics and statistics.

在大学里，机器学习导论课程往往在课程的前段花时间讲授其中一些预备知识。出于历史原因，机器学习课程通常由计算机科学系开设，那里的学生往往在前两个知识领域接受过训练，而在数学和统计学方面则相对欠缺。

> Current machine learning textbooks primarily focus on machine learning algorithms and methodologies and assume that the reader is competent in mathematics and statistics. Therefore, these books only spend one or two chapters on background mathematics, either at the beginning of the book or as appendices. We have found many people who want to delve into the foundations of basic machine learning methods who struggle with the mathematical knowledge required to read a machine learning textbook. Having taught undergraduate and graduate courses at universities, we find that the gap between high school mathematics and the mathematics level required to read a standard machine learning textbook is too big for many people.

现有的机器学习教科书主要聚焦于机器学习算法与方法，并假定读者已具备扎实的数学与统计学功底。因此，这些书只用一两章的篇幅介绍背景数学知识，或置于书首，或作为附录。我们发现，许多想要深入探究基础机器学习方法原理的人，都因阅读机器学习教科书所需的数学知识而举步维艰。在大学教授本科生和研究生课程的过程中，我们发现，对许多人而言，高中数学与阅读标准机器学习教科书所需的数学水平之间的鸿沟实在太大。

> This book brings the mathematical foundations of basic machine learning concepts to the fore and collects the information in a single place so that this skills gap is narrowed or even closed.

本书将基础机器学习概念背后的数学基础推到台前，并将这些内容汇集于一处，以缩小乃至弥合这一技能鸿沟。

> This material is published by Cambridge University Press as Mathematics for Machine Learning by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020). This version is free to view and download for personal use only. Not for re-distribution, re-sale, or use in derivative works. © by M. P. Deisenroth, A. A. Faisal, and C. S. Ong, 2024. https://mml-book.com.

本书内容由剑桥大学出版社出版，书名为 Mathematics for Machine Learning，作者为 Marc Peter Deisenroth、A. Aldo Faisal 和 Cheng Soon Ong（2020 年）。本版本仅供个人使用，可免费阅览与下载；不得再分发、转售或用于衍生作品。© M. P. Deisenroth, A. A. Faisal, C. S. Ong, 2024。https://mml-book.com。

#### 为何再写一本机器学习书？（Why Another Book on Machine Learning?）

> Machine learning builds upon the language of mathematics to express concepts that seem intuitively obvious but that are surprisingly difficult to formalize. Once formalized properly, we can gain insights into the task we want to solve. One common complaint of students of mathematics around the globe is that the topics covered seem to have little relevance to practical problems. We believe that machine learning is an obvious and direct motivation for people to learn mathematics.

机器学习借助数学语言来表达那些看似直观显然、却出人意料地难以形式化的概念。一旦正确地形式化，我们便能对想要解决的任务获得洞见。世界各地的数学专业学生有一个普遍的抱怨：所学的主题似乎与实际问题关联甚少。我们相信，机器学习正是促使人们学习数学的一个显而易见且直接的动力。

> This book is intended to be a guidebook to the vast mathematical literature that forms the foundations of modern machine learning. We motivate the need for mathematical concepts by directly pointing out their usefulness in the context of fundamental machine learning problems. In the interest of keeping the book short, many details and more advanced concepts have been left out. Equipped with the basic concepts presented here, and how they fit into the larger context of machine learning, the reader can find numerous resources for further study, which we provide at the end of the respective chapters. For readers with a mathematical background, this book provides a brief but precisely stated glimpse of machine learning. In contrast to other books that focus on methods and models of machine learning (MacKay, 2003; Bishop, 2006; Alpaydin, 2010; Barber, 2012; Murphy, 2012; Shalev-Shwartz and Ben-David, 2014; Rogers and Girolami, 2016) or programmatic aspects of machine learning (Müller and Guido, 2016; Raschka and Mirjalili, 2017; Chollet and Allaire, 2018), we provide only four representative examples of machine learning algorithms. Instead, we focus on the mathematical concepts behind the models themselves. We hope that readers will be able to gain a deeper understanding of the basic questions in machine learning and connect practical questions arising from the use of machine learning with fundamental choices in the mathematical model.

本书旨在成为通往浩瀚数学文献的指南，这些文献构成了现代机器学习的基础。我们通过直接指出数学概念在基本机器学习问题情境中的用处，来论证为什么需要这些概念。为保持本书篇幅精简，许多细节和更深入的内容被略去。读者在掌握了本书介绍的基本概念、并了解它们如何融入机器学习的整体图景之后，可以找到大量可供深入学习的资源，我们在相应章节的末尾给出了这些资源。对于具备数学背景的读者，本书提供了对机器学习的一个简短而表述精确的概览。与那些聚焦于机器学习方法与模型（model）（MacKay, 2003; Bishop, 2006; Alpaydin, 2010; Barber, 2012; Murphy, 2012; Shalev-Shwartz and Ben-David, 2014; Rogers and Girolami, 2016）或机器学习编程方面（Müller and Guido, 2016; Raschka and Mirjalili, 2017; Chollet and Allaire, 2018）的书籍不同，我们只提供四个有代表性的机器学习算法示例，而将重点放在模型本身背后的数学概念上。我们希望读者能够更深入地理解机器学习中的基本问题，并将使用机器学习时产生的实际问题与数学模型中的根本性选择联系起来。

> We do not aim to write a classical machine learning book. Instead, our intention is to provide the mathematical background, applied to four central machine learning problems, to make it easier to read other machine learning textbooks.

我们的目标并不是写一本传统的机器学习教材，而是提供应用于四个核心机器学习问题的数学背景，使读者更容易阅读其他机器学习教科书。

#### 目标读者是谁？（Who Is the Target Audience?）

> As applications of machine learning become widespread in society, we believe that everybody should have some understanding of its underlying principles. This book is written in an academic mathematical style, which enables us to be precise about the concepts behind machine learning. We encourage readers unfamiliar with this seemingly terse style to persevere and to keep the goals of each topic in mind. We sprinkle comments and remarks throughout the text, in the hope that it provides useful guidance with respect to the big picture.

随着机器学习的应用在社会中日趋普遍，我们认为每个人都应当对其背后的基本原理有所了解。本书采用学术性的数学风格写作，这使我们能够精确地阐述机器学习背后的概念。我们鼓励对这种看似简练的风格感到陌生的读者坚持下去，并牢记每个主题的目标。我们在全书各处穿插了评论与评注，希望能为把握整体图景提供有益的指引。

> The book assumes the reader to have mathematical knowledge commonly covered in high school mathematics and physics. For example, the reader should have seen derivatives and integrals before, and geometric vectors in two or three dimensions. Starting from there, we generalize these concepts. Therefore, the target audience of the book includes undergraduate university students, evening learners and learners participating in online machine learning courses.

本书假定读者具备高中数学和物理中通常讲授的数学知识。例如，读者应当见过导数和积分，以及二维或三维的几何向量（vector）。从这里出发，我们对这些概念加以推广。因此，本书的目标读者包括大学本科生、夜校学员以及参加在线机器学习课程的学习者。

> In analogy to music, there are three types of interaction that people have with machine learning:

以音乐作类比，人与机器学习的交互有三种类型：

> **Astute Listener** The democratization of machine learning by the provision of open-source software, online tutorials and cloud-based tools allows users to not worry about the specifics of pipelines. Users can focus on extracting insights from data using off-the-shelf tools. This enables nontech-savvy domain experts to benefit from machine learning. This is similar to listening to music; the user is able to choose and discern between different types of machine learning, and benefits from it. More experienced users are like music critics, asking important questions about the application of machine learning in society such as ethics, fairness, and privacy of the individual. We hope that this book provides a foundation for thinking about the certification and risk management of machine learning systems, and allows them to use their domain expertise to build better machine learning systems.

**敏锐的聆听者（Astute Listener）** 开源软件、在线教程与云端工具的提供推动了机器学习的普及化，使用户无须担心流水线的具体细节，可以专注于使用现成的工具从数据中提取洞见。这使得不熟悉技术的领域专家也能从机器学习中受益。这类似于聆听音乐：用户能够在不同类型的机器学习之间做出选择与甄别，并从中获益。更有经验的用户则如同音乐评论家，会就机器学习在社会中的应用提出重要问题，例如伦理、公平与个人隐私。我们希望本书能为思考机器学习系统的认证与风险管理提供基础，并让他们能够运用自己的领域专长构建更好的机器学习系统。

> **Experienced Artist** Skilled practitioners of machine learning can plug and play different tools and libraries into an analysis pipeline. The stereotypical practitioner would be a data scientist or engineer who understands machine learning interfaces and their use cases, and is able to perform wonderful feats of prediction from data. This is similar to a virtuoso playing music, where highly skilled practitioners can bring existing instruments to life and bring enjoyment to their audience. Using the mathematics presented here as a primer, practitioners would be able to understand the benefits and limits of their favorite method, and to extend and generalize existing machine learning algorithms. We hope that this book provides the impetus for more rigorous and principled development of machine learning methods.

**经验丰富的演奏家（Experienced Artist）** 熟练的机器学习从业者可以将不同的工具和库即插即用地接入分析流水线。典型的从业者当是数据科学家或工程师：他们理解机器学习接口及其应用场景，能够从数据中施展精彩的预测本领。这类似于演奏大师演绎音乐：技艺高超的演奏者能让现有的乐器焕发生命力，为听众带来享受。以本书介绍的数学作为入门，从业者将能理解自己青睐的方法的优点与局限，并对现有的机器学习算法加以扩展与推广。我们希望本书能为机器学习方法更严谨、更有原则的发展提供推动力。

> **Fledgling Composer** As machine learning is applied to new domains, developers of machine learning need to develop new methods and extend existing algorithms. They are often researchers who need to understand the mathematical basis of machine learning and uncover relationships between different tasks. This is similar to composers of music who, within the rules and structure of musical theory, create new and amazing pieces. We hope this book provides a high-level overview of other technical books for people who want to become composers of machine learning. There is a great need in society for new researchers who are able to propose and explore novel approaches for attacking the many challenges of learning from data.

**初出茅庐的作曲家（Fledgling Composer）** 随着机器学习被应用于新的领域，机器学习的开发者需要开发新方法、扩展现有算法。他们往往是研究人员，需要理解机器学习的数学基础，并揭示不同任务之间的联系。这类似于音乐作曲家：他们在音乐理论的规则与结构之内，创作出新颖而令人惊叹的作品。我们希望本书能为那些希望成为机器学习“作曲家”的人提供对其他技术书籍的高层次概览。社会迫切需要能够提出并探索新颖方法、以攻克从数据中学习所面临的诸多挑战的新一代研究人员。

#### 致谢（Acknowledgments）

> We are grateful to many people who looked at early drafts of the book and suffered through painful expositions of concepts. We tried to implement their ideas that we did not vehemently disagree with. We would like to especially acknowledge Christfried Webers for his careful reading of many parts of the book, and his detailed suggestions on structure and presentation. Many friends and colleagues have also been kind enough to provide their time and energy on different versions of each chapter. We have been lucky to benefit from the generosity of the online community, who have suggested improvements via https://github.com, which greatly improved the book.

我们感谢许多阅读过本书早期草稿、并忍受了书中对概念颇为艰涩的阐述的人。凡是他们提出而我们并未强烈反对的意见，我们都尽力予以采纳。我们要特别感谢 Christfried Webers，他仔细阅读了本书的许多部分，并就结构与表述提出了详尽的建议。许多朋友和同事也慷慨地为各章的不同版本付出了时间与精力。我们有幸受益于在线社区的慷慨相助——他们通过 https://github.com 提出了许多改进建议，使本书大为完善。

> The following people have found bugs, proposed clarifications and suggested relevant literature, either via https://github.com or personal communication. Their names are sorted alphabetically.

以下人员或通过 https://github.com、或通过私下交流，发现了书中的错误、提出了澄清建议并推荐了相关文献。他们的名字按字母顺序排列。

- Abdul-Ganiy Usman
- Adam Gaier
- Adele Jackson
- Aditya Menon
- Alasdair Tran
- Aleksandar Krnjaic
- Alexander Makrigiorgos
- Alfredo Canziani
- Ali Shafti
- Amr Khalifa
- Andrew Tanggara
- Angus Gruen
- Antal A. Buss
- Antoine Toisoul Le Cann
- Areg Sarvazyan
- Artem Artemev
- Artyom Stepanov
- Bill Kromydas
- Bob Williamson
- Boon Ping Lim
- Chao Qu
- Cheng Li
- Chris Sherlock
- Christopher Gray
- Daniel McNamara
- Daniel Wood
- Darren Siegel
- David Johnston
- Dawei Chen
- Ellen Broad
- Fengkuangtian Zhu
- Fiona Condon
- Georgios Theodorou
- He Xin
- Irene Raissa Kameni
- Jakub Nabaglo
- James Hensman
- Jamie Liu
- Jean Kaddour
- Jean-Paul Ebejer
- Jerry Qiang
- Jitesh Sindhare
- John Lloyd
- Jonas Ngnawe
- Jon Martin
- Justin Hsi
- Kai Arulkumaran
- Kamil Dreczkowski
- Lily Wang
- Lionel Tondji Ngoupeyou
- Lydia Knüfing
- Mahmoud Aslan
- Mark Hartenstein
- Mark van der Wilk
- Markus Hegland
- Martin Hewing
- Matthew Alger
- Matthew Lee
- Maximus McCann
- Mengyan Zhang
- Michael Bennett
- Michael Pedersen
- Minjeong Shin
- Mohammad Malekzadeh
- Naveen Kumar
- Nico Montali
- Oscar Armas
- Patrick Henriksen
- Patrick Wieschollek
- Pattarawat Chormai
- Paul Kelly
- Petros Christodoulou
- Piotr Januszewski
- Pranav Subramani
- Quyu Kong
- Ragib Zaman
- Rui Zhang
- Ryan-Rhys Griffiths
- Salomon Kabongo
- Samuel Ogunmola
- Sandeep Mavadia
- Sarvesh Nikumbh
- Sebastian Raschka
- Senanayak Sesh Kumar Karri
- Seung-Heon Baek
- Shahbaz Chaudhary
- Shakir Mohamed
- Shawn Berry
- Sheikh Abdul Raheem Ali
- Sheng Xue
- Sridhar Thiagarajan
- Syed Nouman Hasany
- Szymon Brych
- Thomas Bühler
- Timur Sharapov
- Tom Melamed
- Vincent Adam
- Vincent Dutordoir
- Vu Minh
- Wasim Aftab
- Wen Zhi
- Wojciech Stokowiec
- Xiaonan Chong
- Xiaowei Zhang
- Yazhou Hao
- Yicheng Luo
- Young Lee
- Yu Lu
- Yun Cheng
- Yuxiao Huang
- Zac Cranko
- Zijian Cao
- Zoe Nolan

> Contributors through GitHub, whose real names were not listed on their GitHub profile, are:

通过 GitHub 参与贡献、但未在其 GitHub 个人资料中列出真名的贡献者有：

- SamDataMad
- bumptiousmonkey
- idoamihai
- deepakiim
- insad
- HorizonP
- cs-maillist
- kudo23
- empet
- victorBigand
- 17SKYE
- jessjing1995

> We are also very grateful to Parameswaran Raman and the many anonymous reviewers, organized by Cambridge University Press, who read one or more chapters of earlier versions of the manuscript, and provided constructive criticism that led to considerable improvements. A special mention goes to Dinesh Singh Negi, our LATEX support, for detailed and prompt advice about LATEX-related issues. Last but not least, we are very grateful to our editor Lauren Cowles, who has been patiently guiding us through the gestation process of this book.

我们还要非常感谢 Parameswaran Raman 以及由剑桥大学出版社组织的众多匿名审稿人，他们阅读了手稿早期版本中的一章或多章，并提出了建设性的批评意见，使本书得到长足改进。特别要提到的是我们的 LaTeX 支持 Dinesh Singh Negi，他就 LaTeX 相关问题提供了细致而及时的建议。最后同样重要的是，我们衷心感谢我们的编辑 Lauren Cowles，她一直耐心地引领我们走过本书的孕育历程。

#### 符号表（Table of Symbols）

| Symbol | Typical meaning | 中文含义 |
|---|---|---|
| $a, b, c, \alpha, \beta, \gamma$ | Scalars are lowercase | 标量为小写字母 |
| $x, y, z$ | Vectors are bold lowercase | 向量为粗体小写字母 |
| $A, B, C$ | Matrices are bold uppercase | 矩阵为粗体大写字母 |
| $x^\top, A^\top$ | Transpose of a vector or matrix | 向量或矩阵的转置 |
| $A^{-1}$ | Inverse of a matrix | 矩阵的逆 |
| $\langle x, y \rangle$ | Inner product of $x$ and $y$ | $x$ 与 $y$ 的内积 |
| $x^\top y$ | Dot product of $x$ and $y$ | $x$ 与 $y$ 的点积 |
| $B = (b_1, b_2, b_3)$ | (Ordered) tuple | （有序）元组 |
| $B = [b_1, b_2, b_3]$ | Matrix of column vectors stacked horizontally | 由列向量水平堆叠而成的矩阵 |
| $B = \{b_1, b_2, b_3\}$ | Set of vectors (unordered) | 向量的集合（无序） |
| $\mathbb{Z}, \mathbb{N}$ | Integers and natural numbers, respectively | 分别表示整数与自然数 |
| $\mathbb{R}, \mathbb{C}$ | Real and complex numbers, respectively | 分别表示实数与复数 |
| $\mathbb{R}^n$ | n-dimensional vector space of real numbers | $n$ 维实向量空间 |
| $\forall x$ | Universal quantifier: for all $x$ | 全称量词：对任意 $x$ |
| $\exists x$ | Existential quantifier: there exists $x$ | 存在量词：存在 $x$ |
| $a := b$ | $a$ is defined as $b$ | $a$ 定义为 $b$ |
| $a =: b$ | $b$ is defined as $a$ | $b$ 定义为 $a$ |
| $a \propto b$ | $a$ is proportional to $b$, i.e., $a = \text{constant} \cdot b$ | $a$ 与 $b$ 成正比，即 $a = \text{constant} \cdot b$ |
| $g \circ f$ | Function composition: “g after f” | 函数复合：“先 $f$ 后 $g$” |
| $\iff$ | If and only if | 当且仅当 |
| $\implies$ | Implies | 蕴含 |
| $A, C$ | Sets | 集合 |
| $a \in A$ | $a$ is an element of set $A$ | $a$ 是集合 $A$ 中的元素 |
| $\emptyset$ | Empty set | 空集 |
| $A \setminus B$ | $A$ without $B$: the set of elements in $A$ but not in $B$ | $A$ 去除 $B$：属于 $A$ 但不属于 $B$ 的元素构成的集合 |
| $D$ | Number of dimensions; indexed by $d = 1, \ldots, D$ | 维度数；以 $d = 1, \ldots, D$ 索引 |
| $N$ | Number of data points; indexed by $n = 1, \ldots, N$ | 数据点数；以 $n = 1, \ldots, N$ 索引 |
| $I_m$ | Identity matrix of size $m \times m$ | 大小为 $m \times m$ 的单位矩阵 |
| $0_{m,n}$ | | |

> Matrix of zeros of size $m \times n$

大小为 $m \times n$ 的零矩阵

> $1_{m,n}$ | Matrix of ones of size $m \times n$

$1_{m,n}$ | 大小为 $m \times n$ 的全 1 矩阵

> $e_i$ | Standard/canonical vector (where $i$ is the component that is 1)

$e_i$ | 标准向量／典范向量（其中第 $i$ 个分量为 1）

> $\dim$ | Dimensionality of vector space

$\dim$ | 向量空间（vector space）的维度

> $\operatorname{rk}(A)$ | Rank of matrix $A$

$\operatorname{rk}(A)$ | 矩阵 $A$ 的秩（rank）

> $\operatorname{Im}(\Phi)$ | Image of linear mapping $\Phi$

$\operatorname{Im}(\Phi)$ | 线性映射（linear mapping）$\Phi$ 的像（image）

> $\ker(\Phi)$ | Kernel (null space) of a linear mapping $\Phi$

$\ker(\Phi)$ | 线性映射 $\Phi$ 的核（kernel，即零空间）

> $\operatorname{span}[b_1]$ | Span (generating set) of $b_1$

$\operatorname{span}[b_1]$ | $b_1$ 的张成（span，即生成集）

> $\operatorname{tr}(A)$ | Trace of $A$

$\operatorname{tr}(A)$ | $A$ 的迹（trace）

> $\det(A)$ | Determinant of $A$

$\det(A)$ | $A$ 的行列式（determinant）

> $|\cdot|$ | Absolute value or determinant (depending on context)

$|\cdot|$ | 绝对值或行列式（视上下文而定）

> $\|\cdot\|$ | Norm; Euclidean, unless specified

$\|\cdot\|$ | 范数（norm）；未特别说明时指欧几里得范数（Euclidean norm）

> $\lambda$ | Eigenvalue or Lagrange multiplier

$\lambda$ | 特征值（eigenvalue）或拉格朗日乘子（Lagrange multiplier）

> $E_\lambda$ | Eigenspace corresponding to eigenvalue $\lambda$

$E_\lambda$ | 特征值 $\lambda$ 对应的特征空间（eigenspace）

> Symbol | Typical meaning

符号 | 典型含义

> $x \perp y$ | Vectors $x$ and $y$ are orthogonal

$x \perp y$ | 向量 $x$ 与 $y$ 正交（orthogonal）

> $V$ | Vector space

$V$ | 向量空间

> $V^\perp$ | Orthogonal complement of vector space $V$

$V^\perp$ | 向量空间 $V$ 的正交补（orthogonal complement）

> $\sum_{n=1}^{N} x_n$ | Sum of the $x_n$: $x_1 + \ldots + x_N$

$\sum_{n=1}^{N} x_n$ | $x_n$ 的和：$x_1 + \ldots + x_N$

> $\prod_{n=1}^{N} x_n$ | Product of the $x_n$: $x_1 \cdot \ldots \cdot x_N$

$\prod_{n=1}^{N} x_n$ | $x_n$ 的积：$x_1 \cdot \ldots \cdot x_N$

> $\theta$ | Parameter vector

$\theta$ | 参数向量（parameter vector）

> $\frac{\partial f}{\partial x}$ | Partial derivative of $f$ with respect to $x$

$\frac{\partial f}{\partial x}$ | $f$ 关于 $x$ 的偏导数（partial derivative）

> $\frac{df}{dx}$ | Total derivative of $f$ with respect to $x$

$\frac{df}{dx}$ | $f$ 关于 $x$ 的全导数（total derivative）

> $\nabla$ | Gradient

$\nabla$ | 梯度（gradient）

> $f^* = \min_x f(x)$ | The smallest function value of $f$

$f^* = \min_x f(x)$ | $f$ 的最小函数值

> $x^* \in \arg\min_x f(x)$ | The value $x^*$ that minimizes $f$ (note: arg min returns a set of values)

$x^* \in \arg\min_x f(x)$ | 最小化 $f$ 的取值 $x^*$（注意：arg min 返回的是一组值）

> $L$ | Lagrangian

$L$ | 拉格朗日函数（Lagrangian）

> $\mathcal{L}$ | Negative log-likelihood

$\mathcal{L}$ | 负对数似然（negative log-likelihood）

> $\binom{n}{k}$ | Binomial coefficient, $n$ choose $k$

$\binom{n}{k}$ | 二项式系数（binomial coefficient），$n$ 选 $k$

> $V_X[x]$ | Variance of $x$ with respect to the random variable $X$

$V_X[x]$ | $x$ 关于随机变量（random variable）$X$ 的方差（variance）

> $E_X[x]$ | Expectation of $x$ with respect to the random variable $X$

$E_X[x]$ | $x$ 关于随机变量 $X$ 的期望（expectation）

> $\operatorname{Cov}_{X,Y}[x, y]$ | Covariance between $x$ and $y$.

$\operatorname{Cov}_{X,Y}[x, y]$ | $x$ 与 $y$ 之间的协方差（covariance）。

> $X \perp\!\!\!\perp Y \mid Z$ | $X$ is conditionally independent of $Y$ given $Z$

$X \perp\!\!\!\perp Y \mid Z$ | 给定 $Z$ 时，$X$ 与 $Y$ 条件独立（conditional independence）

> $X \sim p$ | Random variable $X$ is distributed according to $p$

$X \sim p$ | 随机变量 $X$ 服从分布 $p$

> $\mathcal{N}(\mu, \Sigma)$ | Gaussian distribution with mean $\mu$ and covariance $\Sigma$

$\mathcal{N}(\mu, \Sigma)$ | 均值为 $\mu$、协方差为 $\Sigma$ 的高斯分布（Gaussian distribution）

> $\operatorname{Ber}(\mu)$ | Bernoulli distribution with parameter $\mu$

$\operatorname{Ber}(\mu)$ | 参数为 $\mu$ 的伯努利分布（Bernoulli distribution）

> $\operatorname{Bin}(N, \mu)$ | Binomial distribution with parameters $N$, $\mu$

$\operatorname{Bin}(N, \mu)$ | 参数为 $N$、$\mu$ 的二项分布（binomial distribution）

> $\mathrm{Beta}(\alpha, \beta)$ | Beta distribution with parameters $\alpha$, $\beta$

$\mathrm{Beta}(\alpha, \beta)$ | 参数为 $\alpha$、$\beta$ 的贝塔分布（Beta distribution）

> **Table of Abbreviations and Acronyms**

**缩略语与首字母缩略词表**

> Acronym | Meaning

缩略语 | 含义

> e.g. | Exempli gratia (Latin: for example)

e.g. | Exempli gratia（拉丁语：例如）

> GMM | Gaussian mixture model

GMM | 高斯混合模型（GMM）

> i.e. | Id est (Latin: this means)

i.e. | Id est（拉丁语：也就是说）

> i.i.d. | Independent, identically distributed

i.i.d. | 独立同分布（independent, identically distributed）

> MAP | Maximum a posteriori

MAP | 最大后验（maximum a posteriori）

> MLE | Maximum likelihood estimation/estimator

MLE | 最大似然估计／估计量（maximum likelihood estimation/estimator）

> ONB | Orthonormal basis

ONB | 标准正交基（orthonormal basis）

> PCA | Principal component analysis

PCA | 主成分分析（PCA）

> PPCA | Probabilistic principal component analysis

PPCA | 概率主成分分析（probabilistic principal component analysis）

> REF | Row-echelon form

REF | 行阶梯形（row-echelon form）

> SPD | Symmetric, positive definite

SPD | 对称正定（symmetric, positive definite）

> SVM | Support vector machine

SVM | 支持向量机（SVM）
