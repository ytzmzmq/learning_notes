# 《数学基础：机器学习的数学》中英对照版

> **Mathematics for Machine Learning** — M. P. Deisenroth, A. A. Faisal, C. S. Ong 著（2024-01-15 草稿，作者在 [mml-book.github.io](https://mml-book.github.io) 免费公开）。
> 本目录为该书的**中英对照个人学习版**：英文原文在上（引用块），中文译文在下，逐段交替。仅供个人学习使用，请勿分发；正式阅读请支持原书与 Cambridge University Press 出版的印刷版。

## 目录

| 部分 | 章节 | 文件 |
|---|---|---|
| 前言 | 前言与序言 | [00-front-matter.md](00-front-matter.md) |
| 第一部分 数学基础 | 第 1 章 引言与动机 | [01-introduction-and-motivation.md](01-introduction-and-motivation.md) |
| | 第 2 章 线性代数 | [02-linear-algebra.md](02-linear-algebra.md) |
| | 第 3 章 解析几何 | [03-analytic-geometry.md](03-analytic-geometry.md) |
| | 第 4 章 矩阵分解 | [04-matrix-decompositions.md](04-matrix-decompositions.md) |
| | 第 5 章 向量微积分 | [05-vector-calculus.md](05-vector-calculus.md) |
| | 第 6 章 概率与分布 | [06-probability-and-distributions.md](06-probability-and-distributions.md) |
| | 第 7 章 连续优化 | [07-continuous-optimization.md](07-continuous-optimization.md) |
| 第二部分 机器学习核心问题 | 第 8 章 当模型遇上了数据 | [08-when-models-meet-data.md](08-when-models-meet-data.md) |
| | 第 9 章 线性回归 | [09-linear-regression.md](09-linear-regression.md) |
| | 第 10 章 用主成分分析降维 | [10-pca-dimensionality-reduction.md](10-pca-dimensionality-reduction.md) |
| | 第 11 章 用高斯混合模型进行密度估计 | [11-gaussian-mixture-models.md](11-gaussian-mixture-models.md) |
| | 第 12 章 用支持向量机进行分类 | [12-svm-classification.md](12-svm-classification.md) |
| 附录 | 全书统一术语表 | [glossary.md](glossary.md) |

另有合并单文件版：[mml-book-bilingual.html](mml-book-bilingual.html)（双击即可在浏览器中阅读，公式由 MathJax 渲染，图片已内嵌）。

## 排版说明

- **英上中下**：每段英文原文以引用块（`>`）呈现，紧随其后的是对应的中文译文。
- **公式**：全部为 LaTeX。行内公式 `$...$`，展示公式 `$$...$$`；原书编号公式以 `\tag{2.3}` 形式保留编号。公式在原书中如何，译文中就如何，未做改写。
- **术语**：全书按统一术语表翻译（见 [glossary.md](glossary.md)）；术语在每章首次出现时标注「中文（English）」。
- **插图**：`figures/` 目录内 105 张插图全部从原 PDF 裁剪导出。原书的页边注释（margin notes，多为术语提示）未逐条译出；少量原书排印笔误在英文引用块中按原样保留。
- **未译部分**：References（参考文献）与 Index（索引）为纯检索性内容，保留原书 PDF 查阅；各章习题（Exercises）已全部译出。

## 阅读工具建议

- **Typora**（所见即所得）或 **Obsidian** / **VS Code**（Markdown 预览）直接打开各章 md 文件即可，公式与图片自动渲染。
- 不想装软件时，直接双击 `mml-book-bilingual.html`。

## 已知局限

- 翻译由 AI 分章完成并经格式校验（公式定界符配对、段落配对、术语一致性），但未逐句人工审校，个别复杂公式的复原可能与原书排版有细微出入；英文引用块始终是权威原文，遇疑问请以原书为准。
- 原书约 417 页 / 14 万英文词，覆盖 Foreword、第 1–12 章正文与全部习题。
