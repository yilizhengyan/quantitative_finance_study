# Module 2　量化风险和回报

**——现代投资组合理论：搭建起分析框架，从头开始**

## 导读

风险与回报同样重要，不能只看风险或回报——这是习以为常却又最容易被忽略的一点。本篇以马科维茨的均值–方差框架为主线：先把可投资资产及其回报、风险、相关性对应到 $\mu$–$\sigma$ 图上的一个点，再依次讨论两资产组合、$N$ 个风险资产与分散化，最后转入最优化问题的求解——无约束情形用梯度与 Hessian，带约束情形用拉格朗日乘子法，由此得到有效前沿、全局最小方差组合与切点组合；第五章转入 Black–Litterman 模型，用市场均衡权重反推先验回报，再以贝叶斯公式并入主观观点；第六章换个方向，用 VaR 与 ES 回答"这个组合到底有多危险"。全篇围绕一条主线展开：**在给定的风险水平下把回报做到最大，等价于在给定的回报目标下把风险做到最小**。

本篇的统一前提如下：

- 资产数 $N \ge 2$，初始可控金额（初始资产）为 $W$；
- 在时间 $T$ 内挣得越多越好，且只可在 $N$ 个可投资资产之间买卖（做多或做空）；
- 决策一旦确定就不再改变，直到一期结束。

## 目录

- **一、可投资资产**
  - （一）资产的分类
  - （二）组合权重的定义
  - （三）风险资产的统计刻画
  - （四）椭圆分布族
- **二、两资产组合**
  - （一）两种投资思路
  - （二）无风险资产
  - （三）分析的基本前提
  - （四）无风险资产与单个风险资产
  - （五）两个风险资产
  - （六）相关系数的影响
  - （七）加入无风险资产：切点组合、CAL 与 CML
- **三、N 个风险资产与分散化**
  - （一）向量表示与组合的矩
  - （二）资产分散化
  - （三）系统性风险与跨市场操作
  - （四）含无风险资产的 N 资产组合
  - （五）效用函数与风险厌恶
  - （六）应用上的困难
- **四、最优化问题的求解**
  - （一）最优化问题的基本要素
  - （二）无约束优化：梯度与 Hessian
  - （三）无约束优化的应用：最优权重
  - （四）约束优化：拉格朗日乘子法
  - （五）有效前沿：给定回报下的最小风险
    - 1. 拉格朗日函数与一阶条件
    - 2. 标量 $A$、$B$、$C$ 与 $AC - B^2 > 0$
    - 3. 最优权重与乘子
  - （六）全局最小方差组合与 $\sigma^2(m)$
  - （七）含无风险资产：切点组合的推导
    - 1. 带单一约束的最优化
    - 2. 切点组合的权重、均值与标准差
- **五、Black–Litterman 模型**
  - （一）贝叶斯视角：先验、似然与后验
  - （二）先验：从市场权重反推均衡回报
  - （三）观点：$\mathbf{P}$、$\mathbf{Q}$ 与 $\boldsymbol{\Omega}$
  - （四）后验：正态–正态共轭
  - （五）回到最优权重
  - （六）三点使用上的提醒
- **六、风险的衡量：VaR 与 ES**
  - （一）风险是什么：先分类，再度量
  - （二）一致风险度量：四条公理
  - （三）VaR：大概率不会超过的那条线
  - （四）时间尺度：√t 法则
  - （五）ES：尾部风险的真正度量
  - （六）三种估计方法：参数法、历史模拟、蒙特卡洛
  - （七）回测与伦敦鲸：模型要验证，治理要跟上
  - （八）组合层面的 VaR 与 ES：协方差矩阵说了算
  - （九）VaR 的局限与本章收尾

---

## 一、可投资资产

投资的第一步，是弄清"可以买什么"。可投资产的种类繁多，风险特征也各不相同；在把它们放进组合之前，先统一分类的口径，再约定权重的算法。

### （一）资产的分类

按资产的性质，可投资产大致分为三类：

- **金融资产**：股票、外汇等；
- **真实资产**：大宗商品等；
- **无形资产**：劳动力等。

另外不要忘了现实中的限制，比如有些品种仅限国内交易等。这些限制不改变定价的逻辑，却会实实在在地压缩可选择的空间。

### （二）组合权重的定义

各个资产在组合中的权重怎么看？用数学表达，即

$$w_i = \frac{\text{Market Value of Investment in Asset } i}{\text{Total Market Value of the Portfolio}}\tag{1}$$

说白了，就是这个对应资产的市值除以总的资产市值。看起来很简单，但是咱们以债券为例：债券面值 100，市场价值却可能小于 100，我们要用的是那个**市场价值**，不能用面值。

### （三）风险资产的统计刻画

假设每一个风险资产都可以用下列三个指标完整描述。记资产为 $i$：

- **预期回报** $\mu_i$：注意是预期回报，不是实际回报；
- **风险** $\sigma_i$：即该资产回报的标准差；
- **相关性** $\rho_{ij}$：任意两个资产 $i$、$j$ 之间的相关系数。

资产组合的风险正是由这三者共同决定的，这也是均值–方差框架的全部输入。

### （四）椭圆分布族

需要说明的是，上述"均值 + 方差 + 相关系数"的描述并非万能，它成立的前提是资产的回报服从**椭圆分布族**（elliptical distribution）。

椭圆分布其实就是正态分布与 $t$ 分布——把它们的密度曲面从上往下看，等高线是一族椭圆，故得此名。它的累积分布函数（CDF），也就是我们熟悉的 $F(x)$ 概率分布（密度）函数，形式上仍然只依赖均值向量与协方差矩阵，因此用 $\mu$、$\sigma$、$\rho$ 三个量就能把分布说清楚。

## 二、两资产组合

我们的资产组合最终是为了挣钱，回报当然是在风险区间内越高越好；而当给定回报目标时，风险越小越好。

### （一）两种投资思路

马科维茨认为，理性人的投资思路本质上只有这两种：

- 给定回报目标，使风险最小；
- 给定风险水平，使回报最大。

量化就是基于这个理论，把金融问题转化为了数学问题。以回报率为 $y$ 轴、风险为 $x$ 轴，此时就是把风险当作投入、回报即为产出，任何一个资产都可以对应到这个图上的一个点上。

这一套思路就是**均值–方差最优化**（mean–variance optimization）：在均值–方差平面上比较所有可行的组合，再把"给定回报下风险最小"或"给定风险下回报最大"的那一类挑出来。

### （二）无风险资产

风险资产的回报固然高，那有没有无风险资产呢？有的兄弟，有的：正常的定期存款或债券一般就是无风险资产（RFA，Risk-Free Asset）。

无风险资产的回报率就是利率 $R$——它的风险必然为 0，且与任何资产的相关系数都是 0（无风险资产与任何资产无关）。

所以在 $\mu$–$\sigma$ 图上，RFA 就是纵轴上的一个点：风险为 0，回报为 $R$。

### （三）分析的基本前提

OK 啊，现在我们可以采用一些工具来分析了。先把分析的基本前提交代清楚，防止有扯皮：

① **总体回报率**：我们计算的都是总体回报率，包含了资本利得等全部收益；我们获得的收益也会进行再投资，以复利形式增长。

② **可以部分投资**：我们可以只投资一部分，未必必须全部投资——这是为了保证充分的分散。

③ **自由借贷**：假设投资者可以自由地按照无风险回报率借贷。

④ **做空无惩罚**：我们在做空某资产时不会有任何惩罚（现实中做空通常是要罚钱或付融券费用的）。

⑤ **无摩擦市场**：市场没有摩擦，比如没有抵押要求、没有税收等。

后三条是典型的理想化假设。现实中的借贷约束、做空成本与税负，都会让结论出现偏差。

### （四）无风险资产与单个风险资产

现在把资产放到 $\mu$–$\sigma$ 图中，从两个资产与 RFA 入手，构建投资组合。

若只有 RFA 和资产 A：设资产 A 的权重为 $w_A$，全部投资时 RFA 的权重为 $1 - w_A$，投资组合记为 $\Pi$，则

$$R_{\Pi} = w_A\, R_A + (1 - w_A)\, R\tag{2}$$

求期望：

$$\mu_{\Pi} = w_A\, \mu_A + (1 - w_A)\, R = R + w_A\left(\mu_A - R\right)\tag{3}$$

这就是"实际收益率 = 无风险收益率 + 超额收益率"嘛。

再看风险。由于 $(1 - w_A) R$ 为常数，对方差没有影响：

$$\sigma_{\Pi}^2 = \operatorname{Var}\left[w_A R_A + (1 - w_A) R\right] = w_A^2 \operatorname{Var}\left[R_A\right] = w_A^2 \sigma_A^2\tag{4}$$

故

$$\sigma_{\Pi} = w_A\, \sigma_A\tag{5}$$

改变 $w_A$ 就会改变资产的配比，组合对应的点在这条连线上移动。

而这条连线对应的方程就是

$$\mu_{\Pi} = R + \frac{\mu_A - R}{\sigma_A}\,\sigma_{\Pi} = R + k\,\sigma_{\Pi}, \qquad k = \frac{\mu_A - R}{\sigma_A}\tag{6}$$

其中 $k$ 就是大名鼎鼎的**夏普比率**，代表着承担一单位风险所获得的超额回报。

那么这条连线的右上方延伸是什么意思？本质是 $w_A > 1$，即又加大了资产 A 的权重——这就是**加杠杆**：我们通过杠杆买入更多的 A，以获取超额收益。左下方延伸呢？那也是加杠杆，只不过是做空资产 A、买入更多的 RFA，这显然是不划算的：对应的收益率甚至低于 RFA 本身的收益率，而且所谓"负的风险"在现实中很难存在。故最多只考虑右上方延伸。

### （五）两个风险资产

对于两个资产 A 和 B，由上面过程有

$$\mu_{\Pi} = w_A\, \mu_A + (1 - w_A)\,\mu_B = \mu_B + w_A\left(\mu_A - \mu_B\right)\tag{7}$$

$$\sigma_{\Pi} = \sqrt{w_A^2 \sigma_A^2 + (1 - w_A)^2 \sigma_B^2 + 2\rho_{AB}\, w_A (1 - w_A)\,\sigma_A \sigma_B}\tag{8}$$

此时化简得到的就不再是直线，而是双曲线了。左右两端的延伸同样是杠杆。

画一条竖线交双曲线于两个点，我们显然是选择上面那个点——同样的风险，收益率更高。

**定义 1（有效前沿）**　对每一个给定的风险水平，预期回报最高的那些组合所构成的集合，称为**有效前沿**。它就是"最好的投资组合"这一说法的确切含义。

先分析这个 $\sigma_{\Pi}$。由于相关系数 $\rho_{AB}$ 的范围是 $[-1, 1]$，代入化简即可得到组合标准差的上下限：

$$\left|w_A \sigma_A - (1 - w_A)\,\sigma_B\right| \;\le\; \sigma_{\Pi} \;\le\; w_A \sigma_A + (1 - w_A)\,\sigma_B\tag{9}$$

（后面一项中的 $1 - w_A$ 也可以写作 $w_B$。）也就是说，两个资产的相关性越小，组合的风险就能越低。

**定义 2（全局最小方差组合）**　开口朝右的双曲线最左端必有一个点对应着最小的风险，这一点的资产配置称为**全局最小方差组合**（The Global Minimum Variance Portfolio，GMV）。其权重为

$$w_A^{G} = \frac{\sigma_B\left(\sigma_B - \rho_{AB}\,\sigma_A\right)}{\sigma_A^2 + \sigma_B^2 - 2\rho_{AB}\,\sigma_A \sigma_B}\tag{10}$$

这个式子的得到其实就是化简求导取最小值，不多赘述。

三种特殊情形值得单独看一看。

**情形一：$\rho_{AB} = 1$。** 图形与前面"RFA + 单个风险资产"没什么区别，都是直线，只是斜率与初始值不同而已。

**情形二：$\rho_{AB} = -1$。** 此时

$$\sigma_{\Pi} = \left|w_A \sigma_A - (1 - w_A)\,\sigma_B\right| = \left|-\sigma_B + w_A\left(\sigma_A + \sigma_B\right)\right|\tag{11}$$

曲线恰好是一条折线，端点落在纵轴上。也就是说，对于完全负相关的资产，我们可以配置出完全无风险的投资组合：

$$w_A = \frac{\sigma_B}{\sigma_A + \sigma_B}, \qquad \sigma_Z = 0, \qquad \mu_Z = \mu_B + \frac{\sigma_B}{\sigma_A + \sigma_B}\left(\mu_A - \mu_B\right)\tag{12}$$

**情形三：$\rho_{AB} = 0$。** 还是双曲线。

### （六）相关系数的影响

相关系数的作用还分两种情形：

- **没有杠杆时**：若回报率相同，图形表明资产相关系数越小，组合的风险越小；
- **有杠杆时**：若回报率相同，图形表明资产相关系数越大，组合的风险越小。

原因在于：long 方与 short 方所做的操作相反，人为地改变了相关性——二者权重一正一负，乘起来之后 $\rho$ 越大，负得越多，风险就越小。

接下来说一下这两个相关资产再与 RFA 配置，直到延伸到多个资产组合。

### （七）加入无风险资产：切点组合、CAL 与 CML

先看权重结构。设风险资产内部的组合记为 Portfolio，其权重为 $w$；RFA 的权重为 $1 - w$；而 Portfolio 内部，A 占 $w_A$、B 占 $1 - w_A$，如表 1 所示。

<p align="center">表1&nbsp;&nbsp;两个风险资产与 RFA 的权重结构</p>

| 资产 | 权重 |
| --- | --- |
| A | $w \cdot w_A$ |
| B | $w \cdot (1 - w_A)$ |
| RFA | $1 - w$ |
| 合计 | $1$ |

分层地看，组合 $\Pi$ 先由"风险资产组合"与"RFA"两部分构成，前者再拆成 A 与 B；每一层的权重各自加总为 1。

我们也可以扩展，这样就会有两个集合：风险资产的集合与 RFA 的集合。此时 RFA 集合与风险资产集合的连线在 $\mu$–$\sigma$ 图中必然是直线。那么这多条直线，哪一个配置是最好的呢？

看直线斜率。因为简单看 $k = (\mu - R)/\sigma$，其中 $\mu$ 是风险资产集合的收益率，$R$ 是 RFA 的收益率，$\sigma$ 是风险资产集合的风险。总收益率越高，还要风险越小，那就是斜率越大越好。

换种说法，夏普比率越高，代表着承担一单位风险所获得的超额回报越多，这肯定是最好的。

图形上，显然是直线与双曲线相切时直线斜率最大。

**定义 3（切点组合）**　RFA 与风险资产集合的连线若恰好与风险资产的有效前沿相切，切点处的那个风险资产组合称为**切点组合**（tangency portfolio）。它是夏普比率最高的风险资产组合。

切点组合中资产 A 的权重表达式为

$$\tilde{w}_A = \frac{\sigma_B\left[\left(\mu_B - R\right)\rho_{AB}\,\sigma_A - \left(\mu_A - R\right)\sigma_B\right]}{-\left(\mu_A - R\right)\sigma_B^2 - \left(\mu_B - R\right)\sigma_A^2 + \left(\mu_A + \mu_B - 2R\right)\rho_{AB}\,\sigma_A \sigma_B}\tag{13}$$

其中 $\tilde{w}_A$ 表示切点组合内部资产 A 所占的权重。

这条切线，对于个人投资者就是大名鼎鼎的 **CAL 线**（Capital Allocation Line），对于市场（千千万万个个人投资者的汇总）就是 **CML 线**（Capital Market Line）。

> 注：只有风险资产时，对应的是有效前沿；RFA 集合与风险资产集合合并，才对应 CAL 或 CML。

当然哈，这里既然有 $\rho$，那也会有相关系数取值上的讨论，同时当配置无风险组合时也能与 Module 1 的内容相呼应。不多赘述，方法显而易见。

## 三、N 个风险资产与分散化

终于扩展到 $N$ 个风险资产了。因为这种需要配置的资产很多，我们用向量（矩阵）表示，方便计算。

### （一）向量表示与组合的矩

首先回到大前提：资产配置权重加和等于 1，即

$$\mathbf{w}^{T}\mathbf{1}_N = 1\tag{14}$$

其中 $\mathbf{w} = (w_1, w_2, \ldots, w_N)^{T}$ 是一个 $N$ 维列向量，$\mathbf{w}^{T}$ 就是它的转置；$\mathbf{1}_N$ 是 $N$ 维单位列向量。

再设

$$\boldsymbol{\mu} = (\mu_1, \mu_2, \ldots, \mu_N)^{T}, \qquad \boldsymbol{\sigma} = (\sigma_1, \sigma_2, \ldots, \sigma_N)^{T}$$

其中 $\mu_i$、$\sigma_i$ 分别是资产 $i$ 的预期回报与风险。于是组合的均值与标准差为

$$\mu_{\Pi} = \mathbf{w}^{T}\boldsymbol{\mu}, \qquad \sigma_{\Pi} = \sqrt{\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}}\tag{15}$$

其中 $\boldsymbol{\Sigma}$ 为协方差矩阵：

$$\boldsymbol{\Sigma} = \begin{pmatrix}
\sigma_1^2 & \rho_{12}\sigma_1\sigma_2 & \cdots & \rho_{1N}\sigma_1\sigma_N \\
\rho_{21}\sigma_2\sigma_1 & \sigma_2^2 & \cdots & \rho_{2N}\sigma_2\sigma_N \\
\vdots & \vdots & \ddots & \vdots \\
\rho_{N1}\sigma_N\sigma_1 & \rho_{N2}\sigma_N\sigma_2 & \cdots & \sigma_N^2
\end{pmatrix}\tag{16}$$

记住协方差矩阵长这个样子。注意它其实是个对称矩阵，计算什么的是很方便的。

对于 $N$ 个风险资产的组合，我们会画出一个面而非两个风险资产的线，更能凸显有效前沿的特点。

### （二）资产分散化

假设市场上所有的资产都是同质化的：预期回报都是 $\mu$，风险都是 $\sigma$，任意两个资产之间的相关系数都是 $\rho$，权重雨露均沾，都是 $w = 1/N$。

显然，组合的回报仍然为 $\mu$，但风险是

$$\sigma_{\Pi}^2 = \left(\rho + \frac{1 - \rho}{N}\right)\sigma^2\tag{17}$$

当 $N \to \infty$ 时，

$$\sigma_{\Pi}^2 \to \rho\,\sigma^2\tag{18}$$

这就是分散资产可以使风险变小的数学证明。

### （三）系统性风险与跨市场操作

现实生活中，配置 30~50 个资产就基本达到了极限。事实上，风险分散会不断逼近某个值，这个风险就是众所周知的**系统性风险**，也就是不可被分散的风险。

但是！由于不同市场的 $\beta$ 不同，也就是系统性风险的大小不同，我们可以通过**跨市场操作**来降低系统性风险。当然啦，全球性风险依然是不可以被分散的。

### （四）含无风险资产的 N 资产组合

配上 RFA 集合，我们设

$$w_0 = 1 - \mathbf{w}^{T}\mathbf{1}_N\tag{19}$$

通过简单的化简就有

$$\mu_{\Pi} = R + \mathbf{w}^{T}\left(\boldsymbol{\mu} - \mathbf{1}_N R\right), \qquad \sigma_{\Pi} = \sqrt{\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}}\tag{20}$$

（这里只有 $R$ 不是粗体，它只是个标量，不是向量或矩阵。）

### （五）效用函数与风险厌恶

回归到马科维茨的理论，我们就两种投资方式。现实中由于风险有更多方法可以控制，而所谓的"追求更高回报"不可控，故更常见的是固定回报率使之风险最小。

假设没有给定的风险与回报率，我们可以将二者结合起来，构造拉格朗日函数，生成一个效用函数：

$$u = \max_{\mathbf{w}}\left\{ E\left[R_p(w_1, \ldots, w_N)\right] - \frac{\lambda}{2}\,\sigma_p^2(w_1, \ldots, w_N) \right\}, \qquad \text{s.t. } \sum_{i=1}^{N} w_i = 1\tag{21}$$

其中 $\lambda$ 是风险厌恶程度系数。现实中这是比较难确定的，一般都是问卷调查（hhh）。

### （六）应用上的困难

应用上，这个计算简直是太麻烦了。原理很简单，我们只能用一些算法或模型来简化优化；但是怎么判断问题是否为线性的、用什么模型或因子来解决问题，都很难弄。

虽然相关系数相对较稳定，但是风险的测量与回报率的测算很好地弥补了这一点。实在不行，就用其他的分析方法吧。

## 四、最优化问题的求解

前面所有的组合问题，最后都落到同一句话上：在一定的条件下，求某个目标的最大值或最小值。本章先把最优化问题的语言讲清楚，再分别处理无约束与带约束两种情形，最后回到有效前沿与切点组合。

### （一）最优化问题的基本要素

简单介绍：最优化就是在一定的条件下求出最小值或最大值。广泛地说，我们要学会用这种理论培养一种合适的组合意识，当然在这里我们就是应用于资产组合。

要素如下：

1. **目标函数**：设 $f$ 为目标函数，则 $\min f(x_1, x_2, \ldots, x_n)$ 中的 $f$ 就是关于决策变量的函数。
2. **约束条件**：$g_k(x_1, x_2, \ldots, x_n)$，$k = 1, 2, 3, \ldots$，要么大于等于、要么小于等于、要么等于某个确定的约束限制 $b_k$。
3. **一些简单的技巧**，方便求解与编程：

$$\max f(x) = -\min\left(-f(x)\right), \qquad \min f(x) = -\max\left(-f(x)\right)$$

$$\max\left(a + b\,f(x)\right) = a + b\,\max f(x), \qquad b > 0\tag{22}$$

OK 啊，举个例子。众所周知，BS 模型那个大长串公式可以表示为

$$V(S, X, R, \sigma, t) = P$$

此时这个 $V$ 就是目标函数 $f$：我们已知目标值 $P$，反求里面的风险 $\sigma$ 等决策变量。

### （二）无约束优化：梯度与 Hessian

我们把决策变量化作向量的形式：$\mathbf{x} = (x_1, x_2, \ldots, x_n)^{T}$（这是个列向量）。

我们找最小值，不就是求一阶导、找到一阶导为 0 的点，再求二阶导判断是最大值还是最小值嘛。那么，利用梯度（一个倒三角符号，写作 $\nabla$）：

$$\nabla f(\mathbf{x}) = Df = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)^{T}\tag{23}$$

（依旧是个列向量。）然后求二阶导，得出 Hessian 矩阵：

$$H f(\mathbf{x}) = D^{2}f = \begin{pmatrix}
\dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \cdots & \dfrac{\partial^2 f}{\partial x_n \partial x_1} \\
\dfrac{\partial^2 f}{\partial x_1 \partial x_2} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \vdots \\
\vdots & \vdots & \ddots & \vdots \\
\dfrac{\partial^2 f}{\partial x_1 \partial x_n} & \cdots & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}\tag{24}$$

长成这样，这显然也是个对称矩阵。

假设对应最值的横坐标为 $\mathbf{x}^{*}$，那么就有

- **极小值**：$\nabla f(\mathbf{x}^{*}) = 0$，$H f(\mathbf{x}^{*}) > 0$；
- **极大值**：$\nabla f(\mathbf{x}^{*}) = 0$，$H f(\mathbf{x}^{*}) < 0$。

万一没有极值，那就看可不可以先用约束条件约束一下再求解，否则就判定为无极值。最后还要看边界，找出边界上的最值。这是简单的数学思路。

### （三）无约束优化的应用：最优权重

我们做一次简单应用。依旧假设有 $n$ 个资产，权重 $w_i$ 等等初始设置都是一样的，也有

$$\mu_{\Pi} = \mathbf{w}^{T}\boldsymbol{\mu}, \qquad \sigma_{\Pi} = \sqrt{\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}}\tag{25}$$

其中 $\boldsymbol{\Sigma}$ 为协方差矩阵。为了方便计算，我们把它拆成相关系数矩阵 $\mathbf{R}$：

$$\mathbf{R} = \begin{pmatrix}
1 & \rho_{12} & \cdots & \rho_{1n} \\
\rho_{21} & 1 & \cdots & \rho_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\rho_{n1} & \cdots & \cdots & 1
\end{pmatrix}\tag{26}$$

和标准差对角矩阵 $\mathbf{S}$：

$$\mathbf{S} = D(\boldsymbol{\sigma}) = \begin{pmatrix}
\sigma_1 & 0 & \cdots & 0 \\
0 & \sigma_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & \cdots & \cdots & \sigma_n
\end{pmatrix}\tag{27}$$

由此推得

$$\boldsymbol{\Sigma} = \mathbf{S}^{T}\mathbf{R}\mathbf{S} = \mathbf{S}\mathbf{R}\mathbf{S}\tag{28}$$

（因为对角矩阵的转置就是它本身。）这样的计算方式是风险度量体系的基本工具。

其实 RFA 与任何资产的相关系数都等于 0，所以只考虑风险资产的配比，RFA 的权重为 $w_0 = 1 - \mathbf{w}^{T}\mathbf{1}_N$。于是

$$\mu_{\Pi} = R + \mathbf{w}^{T}\left(\boldsymbol{\mu} - \mathbf{1}_N R\right), \qquad \sigma_{\Pi} = \sqrt{\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}}$$

接下来用

$$\max_{\mathbf{w}}\; \mu_{\Pi} - \frac{\lambda}{2}\,\sigma_{\Pi}^2\tag{29}$$

来计算，其中 $\lambda$ 为风险厌恶系数，$\mathbf{w}$ 是风险资产权重。把上面的 $\mu_{\Pi}$、$\sigma_{\Pi}$ 代入进去有

$$\max_{\mathbf{w}}\; V(\mathbf{w}) = \left[R + \mathbf{w}^{T}\left(\boldsymbol{\mu} - \mathbf{1}_N R\right)\right] - \frac{\lambda}{2}\,\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}\tag{30}$$

求一阶导，就是

$$\nabla V(\mathbf{w}^{*}) = \frac{\partial V}{\partial \mathbf{w}}(\mathbf{w}^{*}) = \left(\boldsymbol{\mu} - \mathbf{1}_N R\right) - \lambda\,\boldsymbol{\Sigma}\mathbf{w}^{*} = 0\tag{31}$$

（这里用到了矩阵求导法则，见本节末的注。）整理一下就是

$$\mathbf{w}^{*} = \frac{1}{\lambda}\,\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - \mathbf{1}_N R\right)\tag{32}$$

求二阶导：

$$H V(\mathbf{w}^{*}) = -\lambda\,\boldsymbol{\Sigma} < 0\tag{33}$$

这就是求出来了最大值。

> 注：这里用到的两条矩阵求导法则为 $\frac{\partial(\mathbf{a}^{T}\mathbf{w})}{\partial \mathbf{w}} = \mathbf{a}$ 与 $\frac{\partial(\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w})}{\partial \mathbf{w}} = \left(\boldsymbol{\Sigma} + \boldsymbol{\Sigma}^{T}\right)\mathbf{w} = 2\boldsymbol{\Sigma}\mathbf{w}$（$\boldsymbol{\Sigma}$ 对称）。负定性判据则沿用了标量情形的结论：$-\lambda\boldsymbol{\Sigma} < 0$ 意味着 $V$ 在 $\mathbf{w}^{*}$ 处取极大值。

### （四）约束优化：拉格朗日乘子法

带约束的不能用微积分求解，数学证明略。我们可以用拉格朗日方法解决问题。

**第一步：构造拉格朗日函数。**

$$L(\mathbf{x}, \boldsymbol{\lambda}) = f(x_1, x_2, \ldots, x_n) + \sum_{j=1}^{m}\lambda_j\left(g_j(\mathbf{x}) - b_j\right)\tag{34}$$

此处的 $\lambda_j$ 为拉格朗日乘子，每一个都对应一个优化条件。（顺带说一下：这里的 $\mathbf{x}$ 与 $\boldsymbol{\lambda}$ 都是向量，只是因为写成了加和的形式，此处不再统一加粗。）这样我们就把 $n$ 个带约束的决策变量转化为了 $n + m$ 个不带约束的决策变量。此时就会求

$$\min_{\mathbf{x}, \boldsymbol{\lambda}} L(\mathbf{x}, \boldsymbol{\lambda})\tag{35}$$

**第二步：求一阶导。**

$$\frac{\partial L}{\partial x_i}(\mathbf{x}) = \frac{\partial f}{\partial x_i}(\mathbf{x}) + \sum_{j=1}^{m}\lambda_j\,\frac{\partial g_j}{\partial x_i}(\mathbf{x}) = 0, \qquad i = 1, \ldots, n\tag{36}$$

$$\frac{\partial L}{\partial \lambda_j}(\mathbf{x}) = g_j(\mathbf{x}) - b_j = 0, \qquad j = 1, \ldots, m\tag{37}$$

**第三步：求二阶导**，再判断所得驻点是极大值还是极小值。

一阶条件与二阶条件合起来，就是带约束优化问题的标准解法。

### （五）有效前沿：给定回报下的最小风险

#### 1. 拉格朗日函数与一阶条件

举个例子。拿经典的有效前沿面图来看，我们要计算的是：当收益为定值时，最小风险是多少。这就是一个带约束的优化问题。

我们先表示一下各个参数，再代入这套方法框架求解。目标是

$$\min_{\mathbf{w}}\; \frac{1}{2}\sigma_{\Pi}^2 = \frac{1}{2}\,\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}\tag{38}$$

（乘上 $1/2$ 后会方便计算。）约束条件有两个：

1. **回报率**：$\mu_{\Pi} = \mathbf{w}^{T}\boldsymbol{\mu} = m$，要等于我们设定的目标值 $m$；
2. **预算约束**：我们这里的有效前沿图未考虑 RFA，故 $\mathbf{w}^{T}\mathbf{1}_N = 1$。

于是目标函数与这两个约束一起，代入到拉格朗日函数里有

$$L(\mathbf{w}, \lambda, \gamma) = \frac{1}{2}\,\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w} + \lambda\left(m - \mathbf{w}^{T}\boldsymbol{\mu}\right) + \gamma\left(1 - \mathbf{w}^{T}\mathbf{1}_N\right)\tag{39}$$

对 $\mathbf{w}$ 求一阶导：

$$\frac{\partial L}{\partial \mathbf{w}}\left(\mathbf{w}, \lambda, \gamma\right) = \boldsymbol{\Sigma}\mathbf{w} - \lambda\,\boldsymbol{\mu} - \gamma\,\mathbf{1}_N = 0\tag{40}$$

整理一下得到

$$\mathbf{w}^{*} = \boldsymbol{\Sigma}^{-1}\left(\lambda\,\boldsymbol{\mu} + \gamma\,\mathbf{1}_N\right)\tag{41}$$

然后找二阶导：发现它一定等于 $\boldsymbol{\Sigma}$，而 $\boldsymbol{\Sigma}$ 是个正定矩阵，必然大于 0，所以得到了函数的最小值。

接下来把 $\mathbf{w}^{*}$ 代回约束方程，解出乘子 $\lambda$、$\gamma$。

代入回报率约束，得

$$\boldsymbol{\mu}^{T}\boldsymbol{\Sigma}^{-1}\left(\lambda\boldsymbol{\mu} + \gamma\mathbf{1}_N\right) = \lambda\,\boldsymbol{\mu}^{T}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu} + \gamma\,\boldsymbol{\mu}^{T}\boldsymbol{\Sigma}^{-1}\mathbf{1}_N = m\tag{42}$$

再代入预算约束，得

$$\mathbf{1}_N^{T}\boldsymbol{\Sigma}^{-1}\left(\lambda\boldsymbol{\mu} + \gamma\mathbf{1}_N\right) = \lambda\,\mathbf{1}_N^{T}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu} + \gamma\,\mathbf{1}_N^{T}\boldsymbol{\Sigma}^{-1}\mathbf{1}_N = 1\tag{43}$$

#### 2. 标量 $A$、$B$、$C$ 与 $AC - B^2 > 0$

为了方便起见，我们定义如下三个标量：

$$A = \mathbf{1}_N^{T}\boldsymbol{\Sigma}^{-1}\mathbf{1}_N, \qquad B = \boldsymbol{\mu}^{T}\boldsymbol{\Sigma}^{-1}\mathbf{1}_N = \mathbf{1}_N^{T}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}, \qquad C = \boldsymbol{\mu}^{T}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}\tag{44}$$

于是乘子所满足的方程组化为

$$C\lambda + B\gamma = m, \qquad B\lambda + A\gamma = 1\tag{45}$$

解得

$$\lambda = \frac{Am - B}{AC - B^2}, \qquad \gamma = \frac{C - Bm}{AC - B^2}\tag{46}$$

这里要单独解释一下 $AC - B^2 > 0$ 从何而来。

**理由一（Cauchy–Schwarz）。** 令 $\mathbf{u} = \boldsymbol{\Sigma}^{-1/2}\mathbf{1}_N$、$\mathbf{v} = \boldsymbol{\Sigma}^{-1/2}\boldsymbol{\mu}$，则

$$A = \|\mathbf{u}\|^2, \qquad C = \|\mathbf{v}\|^2, \qquad B = \mathbf{u}^{T}\mathbf{v}$$

由 Cauchy–Schwarz 不等式，$\left(\mathbf{u}^{T}\mathbf{v}\right)^2 \le \|\mathbf{u}\|^2\|\mathbf{v}\|^2$，即

$$B^2 \le AC\tag{47}$$

等号成立当且仅当 $\mathbf{u}$ 与 $\mathbf{v}$ 共线，也就是 $\boldsymbol{\mu}$ 与 $\mathbf{1}_N$ 成正比、所有资产的预期回报相同。而风险资产之间的预期回报一般并不相同，故取严格不等号：$AC - B^2 > 0$。

**理由二（正定性）。** 也可以把 $A$、$B$、$C$ 看作矩阵

$$\begin{pmatrix} A & B \\ B & C \end{pmatrix} = \begin{pmatrix} \mathbf{1}_N^{T} \\ \boldsymbol{\mu}^{T} \end{pmatrix}\boldsymbol{\Sigma}^{-1}\begin{pmatrix} \mathbf{1}_N & \boldsymbol{\mu} \end{pmatrix}$$

的四个元素。由于 $\boldsymbol{\Sigma}^{-1}$ 正定，而 $\mathbf{1}_N$ 与 $\boldsymbol{\mu}$ 线性无关，该矩阵正定，其行列式 $AC - B^2 > 0$。

这个不等式有两处用处：其一，它保证式(45)的系数矩阵可逆，乘子 $\lambda$、$\gamma$ 唯一确定；其二，后文全局最小方差组合的方差 $\sigma_g^2 = 1/A$ 与最优权重都以 $AC - B^2$ 为分母，必须不为零。

#### 3. 最优权重与乘子

最终得到

$$\mathbf{w}^{*} = \frac{1}{AC - B^2}\,\boldsymbol{\Sigma}^{-1}\left[\left(A\boldsymbol{\mu} - B\mathbf{1}_N\right)m + \left(C\mathbf{1}_N - B\boldsymbol{\mu}\right)\right]\tag{48}$$

> 注：注意精细度不高时可以用数值解，更快——使用迭代法计算，小数点后 6 位以内都行。

### （六）全局最小方差组合与 $\sigma^2(m)$

最后解一下全局最小方差组合对应的权重 $\mathbf{w}$。此时就是

$$\min_{\mathbf{w}}\; \frac{1}{2}\,\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}, \qquad \text{s.t. } \mathbf{w}^{T}\mathbf{1}_N = 1\tag{49}$$

除了正常求解的方法，我们可以假设有无穷个 $m$，一定会有唯一的一个 $m$ 对应着我们的目标。所以直接就去找那个唯一的 $m$ 的值就好了。

我们可以采用构建函数 $\sigma_{\Pi}^2(m)$ 的方法，使 $\sigma_{\Pi}^2(m)$ 为目标函数、$m$ 为决策变量。先把式(46)的乘子写回 $\mathbf{w}^{*}$：

$$\mathbf{w}^{*}(m) = \boldsymbol{\Sigma}^{-1}\left(\lambda(m)\,\boldsymbol{\mu} + \gamma(m)\,\mathbf{1}_N\right), \qquad \lambda(m) = \frac{Am - B}{AC - B^2}, \quad \gamma(m) = \frac{C - Bm}{AC - B^2}\tag{50}$$

（记住 $A$、$B$、$C$ 只是标量参数！）

再把它代入 $\sigma_{\Pi}^2 = \mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}$。记 $D = AC - B^2$，则

$$\sigma_{\Pi}^2(m) = \left(\lambda\boldsymbol{\mu} + \gamma\mathbf{1}_N\right)^{T}\boldsymbol{\Sigma}^{-1}\left(\lambda\boldsymbol{\mu} + \gamma\mathbf{1}_N\right) = \lambda^2 C + 2\lambda\gamma B + \gamma^2 A$$

代入 $\lambda$、$\gamma$ 并通分：

$$\sigma_{\Pi}^2(m) = \frac{C\left(Am - B\right)^2 + 2B\left(Am - B\right)\left(C - Bm\right) + A\left(C - Bm\right)^2}{D^2}$$

分子按 $m$ 的幂次合并：$m^2$ 项为 $A^2C - 2AB^2 + AB^2 = A\,D$；$m$ 项为 $-2ABC + 2ABC - 2ABC + 2B^3 = -2B\,D$；常数项为 $CB^2 - 2B^2C + AC^2 = C\,D$。故

$$\sigma_{\Pi}^2(m) = \frac{Am^2 - 2Bm + C}{AC - B^2}\tag{51}$$

这就是构建好的函数：它不过是关于 $m$ 的二次函数再除以正常数 $AC - B^2$，图像是一条开口朝上的抛物线。

对 $m$ 求一阶导并令其为零：

$$\frac{d\sigma_{\Pi}^2(m)}{dm} = \frac{2Am - 2B}{AC - B^2} = 0 \quad \Longleftrightarrow \quad m_g = \frac{B}{A}\tag{52}$$

二阶导 $\dfrac{2A}{AC - B^2} > 0$（因 $A > 0$、$AC - B^2 > 0$），故 $m_g$ 确实是最小值点，$m_g$ 就是全局最小方差组合的回报。

拿到 $m_g$ 就好办了。代回式(50)：

$$\lambda(m_g) = \frac{A \cdot \frac{B}{A} - B}{AC - B^2} = 0, \qquad \gamma(m_g) = \frac{C - B \cdot \frac{B}{A}}{AC - B^2} = \frac{AC - B^2}{A\left(AC - B^2\right)} = \frac{1}{A}$$

于是权向量为

$$\mathbf{w}_g = \frac{\boldsymbol{\Sigma}^{-1}\mathbf{1}_N}{A} = \frac{\boldsymbol{\Sigma}^{-1}\mathbf{1}_N}{\mathbf{1}_N^{T}\boldsymbol{\Sigma}^{-1}\mathbf{1}_N}\tag{53}$$

以及方差

$$\sigma_g^2 = \frac{1}{A}\tag{54}$$

顺带验证一下：$\sigma_g^2 = \mathbf{w}_g^{T}\boldsymbol{\Sigma}\mathbf{w}_g = \dfrac{1}{A^2}\,\mathbf{1}_N^{T}\boldsymbol{\Sigma}^{-1}\mathbf{1}_N = \dfrac{A}{A^2} = \dfrac{1}{A}$，与式(54)一致。

### （七）含无风险资产：切点组合的推导

前面的约束优化都没有考虑 RFA。把 RFA 加进来之后，问题就变成：在给定目标回报 $m$ 的前提下，如何配置风险资产与无风险资产，使风险最小。

（下面公式都按前面的约定：无风险利率记作 $R$ 而不是 $r$，转置一律记作 $^{T}$。）

#### 1. 带单一约束的最优化

$$\min_{\mathbf{w}}\; \frac{1}{2}\,\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}, \qquad \text{s.t. } R + \mathbf{w}^{T}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right) = m\tag{55}$$

拉格朗日函数为

$$L(\mathbf{w}, \lambda) = \frac{1}{2}\,\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w} + \lambda\left(m - R - \mathbf{w}^{T}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)\right)$$

对 $\mathbf{w}$ 求一阶条件：

$$\frac{\partial L}{\partial \mathbf{w}} = \boldsymbol{\Sigma}\mathbf{w} - \lambda\left(\boldsymbol{\mu} - R\mathbf{1}_N\right) = 0\tag{56}$$

再检查二阶条件：目标函数的 Hessian 仍然是协方差矩阵，而它是正定的，因此得到最优权重

$$\mathbf{w}^{*} = \lambda\,\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)\tag{57}$$

（这里同样是把式子转置、再在两边同乘 $\boldsymbol{\Sigma}^{-1}$ 得到的。）

把 $\mathbf{w}^{*}$ 代回约束：

$$R + \lambda\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)^{T}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right) = m\tag{58}$$

解出乘子：

$$\lambda = \frac{m - R}{\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)^{T}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}\tag{59}$$

再代回式(57)，得到 $\mathbf{w}^{*}$ 的最终表达式：

$$\mathbf{w}^{*} = \frac{(m - R)\,\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}{\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)^{T}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}\tag{60}$$

#### 2. 切点组合的权重、均值与标准差

**切点组合**（tangency portfolio）是全部投资于风险资产的那个组合。它的权重记为 $\mathbf{w}_t$：

$$\mathbf{w}_t = \frac{\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}{B - AR}\tag{61}$$

这个组合的均值与标准差为

$$m_t = \mathbf{w}_t^{T}\boldsymbol{\mu} = \frac{C - BR}{B - AR}\tag{62}$$

$$\sigma_t = \sqrt{\mathbf{w}_t^{T}\boldsymbol{\Sigma}\mathbf{w}_t} = \sqrt{\frac{C - 2RB + R^2 A}{\left(B - AR\right)^2}}\tag{63}$$

**这些是怎么得到的？** 关键在于切点组合很特殊，它同时满足两件事。

其一，它在"风险资产 + 无风险资产"的新有效前沿（即 CML）上，故其权重满足式(60)：

$$\mathbf{w}_t = \frac{(m_t - R)\,\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}{\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)^{T}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}$$

其二，它又在"纯风险资产"的旧有效前沿（双曲线）上，故其权重也满足式(41)：

$$\mathbf{w}_t = \boldsymbol{\Sigma}^{-1}\left(\lambda\,\boldsymbol{\mu} + \gamma\,\mathbf{1}_N\right)$$

其中 $\lambda$、$\gamma$ 由式(46)给出。

又因为切点组合全部投资于风险资产，它必须满足预算约束：

$$\mathbf{1}_N^{T}\mathbf{w}_t = 1\tag{64}$$

把式(60)代入，得

$$\mathbf{1}_N^{T}\,\frac{(m_t - R)\,\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}{\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)^{T}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)} = 1$$

先化分母。由 $A$、$B$、$C$ 的定义展开：

$$\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)^{T}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right) = C - 2RB + R^2 A\tag{65}$$

于是条件化为

$$\frac{(m_t - R)\,\mathbf{1}_N^{T}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}{C - 2RB + R^2 A} = 1\tag{66}$$

再化分子：$\mathbf{1}_N^{T}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right) = B - AR$。两边同乘 $C - 2RB + R^2 A$，得

$$\left(m_t - R\right)\left(B - AR\right) = C - 2RB + R^2 A\tag{67}$$

把左端展开，并把 $R\left(B - AR\right) = RB - R^2 A$ 移到右端：

$$m_t\left(B - AR\right) = C - 2RB + R^2 A + RB - R^2 A = C - RB\tag{68}$$

这就得到了切点组合的回报：

$$m_t = \frac{C - RB}{B - AR}\tag{69}$$

再把它代回式(60)。分母由式(65)给出，分子中的 $m_t - R$ 由式(67)给出，于是

$$\mathbf{w}_t = \frac{(m_t - R)\,\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}{C - 2RB + R^2 A} = \frac{C - 2RB + R^2 A}{B - AR}\cdot\frac{\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}{C - 2RB + R^2 A} = \frac{\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}{B - AR}$$

即式(61)。至于标准差，把式(61)代入 $\sigma_t^2 = \mathbf{w}_t^{T}\boldsymbol{\Sigma}\mathbf{w}_t$：

$$\sigma_t^2 = \frac{\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)^{T}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right)}{\left(B - AR\right)^2} = \frac{C - 2RB + R^2 A}{\left(B - AR\right)^2}$$

开方即得式(63)。

至此，含无风险资产的最优配置问题全部解完。切点组合的意义在于：**所有投资者（无论风险偏好如何）持有相同的风险资产组合，只是在切点组合与无风险资产之间调整资金比例**——这正是"切线斜率（夏普比率）最大"的直接结果。

我们终于结束了优化求解的全部。

## 五、Black–Litterman 模型

均值–方差那台"机器"到这里已经装好了，但它有一个致命的输入问题：机器把 $\boldsymbol{\mu}$ 当作已知量，而真实的预期回报谁也说不准。更麻烦的是，最优权重里出现了 $\boldsymbol{\Sigma}^{-1}$——输入误差会被它放大，$\boldsymbol{\mu}$ 稍微动一点，权重就可能从"重仓 A"跳到"重仓 B"。这就是所谓的**误差最大化**（error maximization）。

Black–Litterman 模型（下称 BL）就是为了修这个问题。理解它有两条路，最后殊途同归：

1. **贝叶斯视角**：先有一个先验分布，信息披露（主观观点）之后更新出后验分布；
2. **反向推导视角**：不问"从观点到权重"，而是反过来问——市场上现在这个权重，隐含了什么样的预期回报？把它当作先验。

两条路都落在同一个式子上：先得到后验均值 $\boldsymbol{\mu}_{BL}$，再代回第四部分的最优权重公式。

### （一）贝叶斯视角：先验、似然与后验

回忆贝叶斯公式。设 $E$ 是"要估计的东西"（这里指各资产的预期回报），$I$ 是新获得的信息（这里是我们的观点），则

$$P(E \mid I) = \frac{P(I \mid E)\,P(E)}{P(I)}\tag{70}$$

其中 $P(E)$ 是先验，$P(I \mid E)$ 是似然，$P(E \mid I)$ 是后验。这三样在 BL 里都是正态分布：

- **先验**：$\boldsymbol{\mu} \sim N(\boldsymbol{\Pi}, \tau\boldsymbol{\Sigma})$，即市场均衡给出的预期回报，外加一个不确定度；
- **似然**：$\mathbf{P}\boldsymbol{\mu} \sim N(\mathbf{Q}, \boldsymbol{\Omega})$，即我们的观点，同样允许有误差；
- **后验**：仍然是正态，均值落在 $\boldsymbol{\Pi}$ 与观点之间，方差比先验更小。

> 注：正态 × 正态 = 正态，这叫**共轭**（conjugacy）。正因为共轭，BL 才有闭式解——否则就要做数值积分了。

### （二）先验：从市场权重反推均衡回报

先解决"$\boldsymbol{\Pi}$ 从哪来"。BL 的办法是**反向推导**（reverse optimization）：假设市场组合就是代表性投资者的最优组合，那么市场上观察到的权重，必然对应着某一组预期回报。

由第四部分式(32)，最优权重满足

$$\mathbf{w} = \frac{1}{\lambda}\,\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu} - R\mathbf{1}_N\right) = \frac{1}{\lambda}\,\boldsymbol{\Sigma}^{-1}\boldsymbol{\Pi}\tag{71}$$

其中 $\boldsymbol{\Pi} = \boldsymbol{\mu} - R\mathbf{1}_N$ 是**超额回报**向量。为了与 CAPM 保持一致，BL 全程都用超额回报而不是绝对回报。

把 $\mathbf{w}$ 换成市场组合的权重 $\mathbf{w}_{mkt}$，反解就得到

$$\boldsymbol{\Pi} = \lambda_{mkt}\,\boldsymbol{\Sigma}\,\mathbf{w}_{mkt}\tag{72}$$

这就是所谓的**先验市场超额回报**。剩下唯一的问题是：$\lambda_{mkt}$ 到底是什么？

金融学家们给了一个很巧的处理。在式(72)两边左乘 $\mathbf{w}_{mkt}^{T}$：

$$\mathbf{w}_{mkt}^{T}\boldsymbol{\Pi} = \lambda_{mkt}\,\mathbf{w}_{mkt}^{T}\boldsymbol{\Sigma}\mathbf{w}_{mkt} = \lambda_{mkt}\,\sigma_{mkt}^{2}\tag{73}$$

于是

$$\lambda_{mkt} = \frac{\mathbf{w}_{mkt}^{T}\boldsymbol{\Pi}}{\sigma_{mkt}^{2}} = \frac{S_{mkt}}{\sigma_{mkt}}\tag{74}$$

其中 $S_{mkt} = \mathbf{w}_{mkt}^{T}\boldsymbol{\Pi}/\sigma_{mkt}$ 就是**市场组合的夏普比率**（分子是超额回报，分母是标准差）。$S_{mkt}$ 在实务里常直接取 $0.5$ 左右（标普 500 的长期夏普大致就在这个量级），$\sigma_{mkt}$ 由市场组合的波动率估出来，$\lambda_{mkt}$ 就有了。

有了 $\boldsymbol{\Pi}$ 还只是一个点，要给先验一个分布：

$$\boldsymbol{\mu} \sim N\left(\boldsymbol{\Pi},\ \tau\boldsymbol{\Sigma}\right)\tag{75}$$

$\tau$ 是**先验的不确定度**：$\tau$ 越小，越相信"市场均衡就是对的"。文献里常取 $\tau \in [0.01,\ 0.05]$，也有取 $\tau = 1/T$（$T$ 为样本期数）的做法。

> 注：$\tau$ 只调"收缩的力度"，不改变"往哪个方向收缩"。所以它取 0.02 还是 0.05，通常不会改变组合的排序，只会改变最终权重离市场组合有多远。

### （三）观点：$\mathbf{P}$、$\mathbf{Q}$ 与 $\boldsymbol{\Omega}$

先验说完了，再说"信息"。BL 里的观点只有一种形式：**对（超额）回报的线性判断**。

观点分两类：

- **绝对观点**：直接说某个资产的回报是多少，例如"资产 2 的回报是 3%"；
- **相对观点**：说两个（或几个）资产谁比谁好多少，例如"资产 3 比资产 1 好 10%"。

把它们写成矩阵，就是 BL 的三件套：

- $\mathbf{P}$：$K \times N$ 的**观点矩阵**（$K$ 个观点、$N$ 个资产），每行对应一个观点；元素取 $-1$、$0$、$1$，分别代表卖出、不动、买入；
- $\mathbf{Q}$：$K \times 1$ 的**观点回报向量**，每个观点对应一个数，与资产数无关；
- $\boldsymbol{\Omega}$：$K \times K$ 的**置信矩阵**，对角线上是每个观点的误差方差，同样与资产数无关。

以"资产 3 比资产 1 好 10%、资产 2 的回报是 3%"为例（4 个资产、2 个观点），如表 2 所示。

<p align="center">表2&nbsp;&nbsp;两个观点对应的 $\mathbf{P}$ 与 $\mathbf{Q}$</p>

| 观点 | 内容 | $\mathbf{P}$ 的对应行 | $\mathbf{Q}$ 的对应元素 |
| --- | --- | --- | --- |
| 1（相对） | 资产 3 比资产 1 好 10% | $(-1,\ 0,\ 1,\ 0)$ | $0.10$ |
| 2（绝对） | 资产 2 的回报为 3% | $(0,\ 1,\ 0,\ 0)$ | $0.03$ |

于是

$$\mathbf{P} = \begin{pmatrix} -1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}, \qquad \mathbf{Q} = \begin{pmatrix} 0.10 \\ 0.03 \end{pmatrix}\tag{76}$$

注意第一行：$-r_1 + r_3 = 0.10$，正好就是"资产 3 比资产 1 好 10%"。可以看出规律：**绝对观点只有 1 个非零元，相对观点一般有 2 个**（当然也可以写 3 个以上的组合观点）。

观点不是一个精确的等式，而是"带误差的等式"：

$$\mathbf{P}\boldsymbol{\mu} = \mathbf{Q} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim N\left(\mathbf{0},\ \boldsymbol{\Omega}\right)\tag{77}$$

误差越小，说明对这个观点越有信心。

$\boldsymbol{\Omega}$ 怎么定？这其实是个公认的难题。BL 原文与 He–Litterman 的常用做法是：**假设观点的误差与先验的误差同量级**，

$$\boldsymbol{\Omega} = \operatorname{Diag}\left(\mathbf{P}\,\tau\boldsymbol{\Sigma}\,\mathbf{P}^{T}\right)\tag{78}$$

也就是"我对这套观点的不确定程度，和对市场的不确定程度差不多"。实务中还会再乘一个缩放系数 $c$：$c$ 越小越相信自己的观点，$c$ 越大越听市场的。

> 注：$\boldsymbol{\Omega}$ 不能取太小。若 $\boldsymbol{\Omega} \to \mathbf{0}$，等于宣称观点百分之百准确，后验会被硬拽到观点上，$\mathbf{P}\tau\boldsymbol{\Sigma}\mathbf{P}^{T} + \boldsymbol{\Omega}$ 的数值条件也会变差。

### （四）后验：正态–正态共轭

现在把先验和似然乘起来。按贝叶斯公式(70)，后验密度正比于

$$p\left(\boldsymbol{\mu} \mid \mathbf{Q}\right) \propto \exp\left\{-\frac{1}{2}\left[\left(\boldsymbol{\mu}-\boldsymbol{\Pi}\right)^{T}\left(\tau\boldsymbol{\Sigma}\right)^{-1}\left(\boldsymbol{\mu}-\boldsymbol{\Pi}\right) + \left(\mathbf{P}\boldsymbol{\mu}-\mathbf{Q}\right)^{T}\boldsymbol{\Omega}^{-1}\left(\mathbf{P}\boldsymbol{\mu}-\mathbf{Q}\right)\right]\right\}$$

指数里是一个关于 $\boldsymbol{\mu}$ 的二次型。把它按 $\boldsymbol{\mu}$ 展开、合并同类项，写成

$$-\frac{1}{2}\,\boldsymbol{\mu}^{T}\mathbf{A}\boldsymbol{\mu} + \boldsymbol{\mu}^{T}\mathbf{b} + \text{常数}$$

其中

$$\mathbf{A} = \left(\tau\boldsymbol{\Sigma}\right)^{-1} + \mathbf{P}^{T}\boldsymbol{\Omega}^{-1}\mathbf{P}, \qquad \mathbf{b} = \left(\tau\boldsymbol{\Sigma}\right)^{-1}\boldsymbol{\Pi} + \mathbf{P}^{T}\boldsymbol{\Omega}^{-1}\mathbf{Q}\tag{79}$$

配方就得到后验的均值与协方差：

$$\boldsymbol{\mu}_{BL} = \mathbf{A}^{-1}\mathbf{b}, \qquad \boldsymbol{\Sigma}_{BL} = \mathbf{A}^{-1}\tag{80}$$

写开就是

$$\boldsymbol{\Sigma}_{BL} = \left[\left(\tau\boldsymbol{\Sigma}\right)^{-1} + \mathbf{P}^{T}\boldsymbol{\Omega}^{-1}\mathbf{P}\right]^{-1}, \qquad \boldsymbol{\mu}_{BL} = \boldsymbol{\Sigma}_{BL}\left[\left(\tau\boldsymbol{\Sigma}\right)^{-1}\boldsymbol{\Pi} + \mathbf{P}^{T}\boldsymbol{\Omega}^{-1}\mathbf{Q}\right]\tag{81}$$

式(81)已经是对的，但里面要反演一个 $N \times N$ 的矩阵。用 **Woodbury 恒等式**

$$\left(\mathbf{D} + \mathbf{U}\mathbf{C}\mathbf{V}\right)^{-1} = \mathbf{D}^{-1} - \mathbf{D}^{-1}\mathbf{U}\left(\mathbf{C}^{-1} + \mathbf{V}\mathbf{D}^{-1}\mathbf{U}\right)^{-1}\mathbf{V}\mathbf{D}^{-1}$$

可以把它化成只反演 $K \times K$ 矩阵的形式：

$$\boldsymbol{\mu}_{BL} = \boldsymbol{\Pi} + \tau\boldsymbol{\Sigma}\,\mathbf{P}^{T}\left(\mathbf{P}\,\tau\boldsymbol{\Sigma}\,\mathbf{P}^{T} + \boldsymbol{\Omega}\right)^{-1}\left(\mathbf{Q} - \mathbf{P}\boldsymbol{\Pi}\right)\tag{82}$$

$$\boldsymbol{\Sigma}_{BL} = \tau\boldsymbol{\Sigma} - \tau\boldsymbol{\Sigma}\,\mathbf{P}^{T}\left(\mathbf{P}\,\tau\boldsymbol{\Sigma}\,\mathbf{P}^{T} + \boldsymbol{\Omega}\right)^{-1}\mathbf{P}\,\tau\boldsymbol{\Sigma}\tag{83}$$

> 注：为什么要化成式(82)？因为观点通常只有 5~10 个，资产却可能上百个：$K \times K$ 的求逆比 $N \times N$ 便宜得多，数值上也更稳。K×K 的求逆比 N×N 便宜得多，数值上也更稳。

式(82)的直觉很清楚：$\boldsymbol{\mu}_{BL}$ 等于"先验 $\boldsymbol{\Pi}$"加上一个修正项，修正项的大小取决于**观点与先验的偏离** $\mathbf{Q} - \mathbf{P}\boldsymbol{\Pi}$。看两个极端：

- 若 $\boldsymbol{\Omega} \to \infty$（完全不信观点），修正项趋于 $\mathbf{0}$，$\boldsymbol{\mu}_{BL} \to \boldsymbol{\Pi}$，退化成纯市场均衡；
- 若 $\boldsymbol{\Omega} \to \mathbf{0}$（完全相信观点），后验会被拉到恰好满足 $\mathbf{P}\boldsymbol{\mu}_{BL} = \mathbf{Q}$。

### （五）回到最优权重

拿到后验均值之后，剩下的事和第四部分一模一样——代回式(32)：

$$\mathbf{w}^{**} = \frac{1}{\lambda}\,\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_{BL}\tag{84}$$

（实务上一般还会把权重归一化，使其加总为 1。）这里的 $\lambda$ 是**投资者自己**的风险厌恶系数，与前面估计 $\boldsymbol{\Pi}$ 时用的 $\lambda_{mkt}$ 不是一回事：$\lambda_{mkt}$ 是"市场平均"的口味，$\lambda$ 是你自己的口味。

> 注：协方差用 $\boldsymbol{\Sigma}$ 还是后验的 $\boldsymbol{\Sigma}_{BL}$，两种做法都有人用。前者是"只修正 $\boldsymbol{\mu}$"，后者是"连不确定性一起修正"，理论上后者更一致，实务上差别不大。

### （六）三点使用上的提醒

1. **BL 修的是 $\boldsymbol{\mu}$，不是优化器本身。** $\boldsymbol{\Sigma}$ 的估计误差依然原封不动地留在 $\boldsymbol{\Sigma}^{-1}$ 里。它治好了"预期回报乱猜"，没治好"协方差估计噪声大"。
2. **观点必须是线性可表达的。** $\mathbf{P}\boldsymbol{\mu} = \mathbf{Q}$ 这个形式只能写"谁的回报比谁高多少"，写不了"波动率会上升"或"相关性会改变"这类判断——那些要靠别的模型。
3. **$\tau$ 与 $\boldsymbol{\Omega}$ 的相对大小，决定你听市场的还是听自己的。** $\tau$ 大、$\boldsymbol{\Omega}$ 小，后验靠近观点；反过来，后验靠近市场组合。这两个参数就是 BL 的两个旋钮。

与马科维茨的对比可以总结为表 3。

<p align="center">表3&nbsp;&nbsp;马科维茨与 Black–Litterman 的对比</p>

| | 马科维茨 | Black–Litterman |
| --- | --- | --- |
| $\boldsymbol{\mu}$ 的来源 | 直接给点估计（往往是历史均值） | 由市场权重反推，再用观点更新 |
| 对输入误差 | 极度敏感，$\boldsymbol{\Sigma}^{-1}$ 会把误差放大 | 先验收缩，稳健得多 |
| 没有观点时 | 权重可能集中在少数资产上 | 退化为市场组合 |
| 观点怎么表达 | 无法表达 | $\mathbf{P}$、$\mathbf{Q}$、$\boldsymbol{\Omega}$ 三个矩阵 |
| 计算量 | 一次 $N \times N$ 求逆 | 多一次 $K \times K$ 求逆 |

## 六、风险的衡量：VaR 与 ES

前五章做的都是"把回报做高、把风险压低"：先有 $\mu$–$\sigma$ 图上的点，再有有效前沿和最优权重，最后用 Black–Litterman 修一修输入。可始终没回答一个最直接的问题——**这个组合到底有多危险？** 说"波动率是 15%"固然没错，但老板想问的是"最坏能亏多少"。

这一章换个角度：不再构造组合，只度量组合。工具只有两个：**VaR**（在险价值）回答"大概率不会亏超过多少"，**ES**（预期损失）回答"如果真的超过了，平均亏多少"。这是巴塞尔协议里真正在用的两个数。

### （一）风险是什么：先分类，再度量

"风险"这个词来自 17 世纪的法语 *risque*，本义是"暴露于危险之中"。落到金融上，风险管理的标准定义是：**识别、评估、排序风险，再协调而经济地运用资源，去最小化、监控并控制不幸事件发生的概率与影响**。

"风控是一种艺术"。这句话点破了后面所有公式的用途：度量不是为了消灭风险，而是为了决定"该留多少、该转走多少"。

先看风险有哪些。金融风险可以一直往下拆：

- **市场风险**：价格动了就亏钱。往下分股票、利率、汇率、商品四类价格风险；再往下，可以分成**交易风险**（持仓本身变动）与**缺口风险**（资产负债的期限错配），也可以分成**一般市场风险**（整个市场同涨同跌，即 $\beta$）与**特定风险**（发行人自己的问题，即 $\varepsilon$）。
- **信用风险**：对手方不还钱。往下有**交易风险**与**组合集中度**，再往下是**发行风险**（证券发不出去）、**发行人风险**（发行人不还钱）、**交易对手风险**（衍生品对手违约）。
- **非金融风险**：操作风险、声誉风险、法律风险、监管风险、流动性风险。它们不进 $\mu$、$\sigma$ 的模型，却常常是真正把机构搞垮的那一类——伦敦鲸就是治理与操作问题。

信用风险里最常用的粗算是期望损失：

$$EL = PD \times LGD \times EAD\tag{85}$$

即"违约概率 × 违约损失率 × 违约敞口"。你笔记里简单记成 $PD \times LGD$ 也行，$EAD$ 就是"有多少钱暴露在外面"。

> 注：利率互换（swap）这类工具同时带市场与信用两种风险：一边盯浮动利率，一边盯对手方。归到哪个格子里，取决于你想度量什么。

分类解决了"有哪些风险"，但管理上要的是**一个数**：能不能用单个指标概括"这个组合有多危险"，而且这个指标要好用、可比、经得起追责？下一节先给"好用"定个标准。

### （二）一致风险度量：四条公理

要比较"哪个指标更靠谱"，先得有个标准。Artzner 等人在 1999 年给出了一套公理：满足这四条，才配叫"好的风险度量"。

**定义 4（一致风险度量）**　记风险度量为 $\mathcal{R}(\cdot)$，它作用在一个组合的损益（或价值）上。若对任意组合 $X$、$Y$，任意 $h \ge 0$ 与任意无风险金额 $n$，都满足

① **单调性**：若 $Y \ge X$ 恒成立，则 $\mathcal{R}(Y) \le \mathcal{R}(X)$；

② **次可加性**：$\mathcal{R}(X + Y) \le \mathcal{R}(X) + \mathcal{R}(Y)$；

③ **正齐次性**：$\mathcal{R}(hX) = h\,\mathcal{R}(X)$；

④ **平移不变性**：$\mathcal{R}(X + n) = \mathcal{R}(X) - n$，

则称 $\mathcal{R}$ 为**一致风险度量**（coherent risk measure）。

表 4 把它们摆在一起。

<p align="center">表4&nbsp;&nbsp;一致风险度量的四条公理</p>

| 公理 | 数学表达 | 直觉 |
| --- | --- | --- |
| 单调性 | $Y \ge X \Rightarrow \mathcal{R}(Y) \le \mathcal{R}(X)$ | 收益更高的组合，风险更低 |
| 次可加性 | $\mathcal{R}(X+Y) \le \mathcal{R}(X)+\mathcal{R}(Y)$ | 分散化不会增加风险 |
| 正齐次性 | $\mathcal{R}(hX) = h\,\mathcal{R}(X)$ | 仓位翻倍，风险翻倍 |
| 平移不变性 | $\mathcal{R}(X+n) = \mathcal{R}(X) - n$ | 加一笔无风险现金，风险直接下降 |

判断一个指标好不好，就看它满不满足这四条。**VaR 满足单调性、正齐次性与平移不变性，偏偏不满足次可加性**——两个独立的头寸合在一起，VaR 反而可能变大（反例见第五节）。这在数学上是硬伤：一个"分散化会显得风险更大"的指标，没法拿来做组合管理。

**ES 四条全满足**，所以是一致风险度量。这也是巴塞尔 III 从 VaR 转向 ES 的理论依据。

> 注：这里用 $\mathcal{R}$ 而不是 $R$，是为了和前面表示收益率的 $R$、以及表示无风险利率的 $R$ 区分开。同一个字母在三处表达三个意思，是这类笔记最容易出错的地方。

### （三）VaR：大概率不会超过的那条线

先说个小故事。1990 年代初，JPMorgan 的董事长 Dennis Weatherstone 每天收到一叠风险报告，满是各种希腊字母和分项敞口，却看不出"整个银行明天到底可能亏多少"。他要求下属给一个数：**未来 24 小时、全行交易组合的最大可能损失**。下属借马科维茨的组合理论硬做了出来——这份报告每天下午 4:15 放到董事长桌上，史称"4:15 报告"。这就是 VaR 的诞生。

**定义 5（在险价值）**　设持有期为 $T$、置信水平为 $c$，组合在 $T$ 内的损失为 $L$。则 VaR 是满足下式的损失阈值：

$$\Pr\left(L > \mathrm{VaR}_c\right) = 1 - c\tag{86}$$

等价地，

$$\mathrm{VaR}_c = \inf\left\{\ell : \Pr\left(L \le \ell\right) \ge c\right\}\tag{87}$$

（连续分布下两式等价；若损失有跳变、概率集中在某几个点上，式(86)可能没有解，此时以式(87)为准。其中 $\inf$ 为下确界）

也就是说，VaR 就是**损失分布的 $c$ 分位数**。

你的理解是：在正常市场条件下，给定置信水平与持有期，组合"很可能不会超过"的损失门槛。举个具体例子：

> 95% 置信水平、1 天持有期的 VaR = 100 万元，意思是：明天一天内，损失有 95% 的概率不超过 100 万；反过来说，有 5% 的概率会超过 100 万。一年约 250 个交易日，平均 12.5 天会出现一次超过 100 万的损失。

这里要非常小心一句话：**VaR 不是"最多亏 100 万"**。它只说"超过 100 万是小概率事件"，至于小概率真的发生时会亏多少，VaR 说明不了，得用 ES 补上。

VaR 有两个参数：

- **置信水平 $c$**：监管常用 99%，内部管理常用 95%；
- **持有期 $T$**：交易管理常用 1 天，监管资本要求常用 10 天。

#### 1. 2.33 是怎么来的

如果损失服从正态分布 $L \sim N(\mu, \sigma^2)$，VaR 就能直接写出来。先标准化：

$$\Pr\left(\frac{L - \mu}{\sigma} \le \frac{\mathrm{VaR}_c - \mu}{\sigma}\right) = c \quad\Longrightarrow\quad \frac{\mathrm{VaR}_c - \mu}{\sigma} = z_c = \Phi^{-1}(c)\tag{88}$$

于是

$$\mathrm{VaR}_c = \mu + z_c\,\sigma\tag{89}$$

99% 时 $z_{0.99} = \Phi^{-1}(0.99) = 2.3264$——这就是那个"著名的 2.33"（截到两位小数就是 2.33，笔记里的 2.3264、2.326、2.33 是同一个数的不同精度）。

这里的 $\mu$ 是**损失**分布的均值，是个正数。到第八节换成收益率向量时，损失均值就成了 $-\mathbf{w}^{T}\boldsymbol{\mu}$，所以组合 VaR 里那一项是负的。

> 注：2.33 是**单尾**的 1% 分位点（左侧尾巴占 1%）。常被拿来对照的 2.58 是**双尾**的：$\pm 2.58\sigma$ 覆盖中间 99%，两侧各 0.5%。算 VaR 只关心左尾，所以用 2.33 不用 2.58。

常用的几个分位点如下（双侧覆盖 $\Leftrightarrow$ 单尾概率）：

- $\mu \pm 1\sigma$：约 68%（单尾 16%）
- $\mu \pm 1.645\sigma$：90%（单尾 5%）
- $\mu \pm 1.96\sigma$：95%（单尾 2.5%）
- $\mu \pm 2.576\sigma$：99%（单尾 0.5%）

#### 2. VaR 的三个"不能"

VaR 好用，但必须知道它没说的事：

1. **不能告诉你绝对的最大损失。** 它是一个概率陈述，不是上界。
2. **不能告诉你尾部有多深。** 超过 VaR 之后平均亏多少，它不知道。
3. **不能跨分布直接比较。** 两个 VaR 都是 100 万的组合，尾部可能差异较大。

### （四）时间尺度：√t 法则

监管要 10 天的 VaR，而银行通常只有每天的数据。最省事的做法不是把 1 天 VaR 乘 10，而是乘 $\sqrt{10}$：

$$\mathrm{VaR}_T = \mathrm{VaR}_1 \times \sqrt{T}\tag{90}$$

为什么是根号？因为**方差随时间线性增长**：$T$ 个独立同分布的日收益，方差是 $\sigma^2T$，标准差就是 $\sigma\sqrt{T}$。风险按时间的平方根长大，这就是"平方根法则"——"风险长得慢，时间开根号"。

于是单资产参数法的完整公式为

$$\mathrm{VaR} = \sigma_{\text{日}} \times \sqrt{T} \times z_c \times V\tag{91}$$

其中 $\sigma_{\text{日}}$ 是日波动率，$T$ 是天数，$z_c$ 是分位数，$V$ 是名义金额。

举个课上的例子：10 万美元买 QQQ（纳斯达克 100），日波动率 2.21%，99% 置信：

- 1 日 VaR $= 2.21\% \times \sqrt{1} \times 2.33 \times 100{,}000 = \$5{,}149$
- 10 日 VaR $= 2.21\% \times \sqrt{10} \times 2.33 \times 100{,}000 = \$16{,}284$

> 注：√t 法则成立的前提是**独立同分布**（这里还忽略了均值项）。现实中股价有波动率聚集（ARCH 效应），肥尾也会让 √t 低估长期风险：短周期尚可用，长周期、高置信水平下要谨慎。《Explorations in Asset Returns》提到：对数收益沿时间可加、简单收益沿时间要连乘，二者的分布在长周期会明显分叉（简单收益趋向对数正态），而 √t 只"看见"了方差，看不见偏度。

### （五）ES：尾部风险的真正度量

设想两个组合，日 VaR（99%）都是 100 万。组合 A 超过 VaR 之后平均亏 120 万，组合 B 平均亏 500 万。VaR 看这两个组合一模一样，但显然 B 危险得多。

**定义 6（预期损失）**　在置信水平 $c$ 下，ES 是损失超过 VaR 时的条件期望：

$$\mathrm{ES}_c = \mathbb{E}\left[L \mid L > \mathrm{VaR}_c\right]\tag{92}$$

它还有一个等价写法——把 $c$ 到 1 之间所有的 VaR 平均起来：

$$\mathrm{ES}_c = \frac{1}{1-c}\int_c^1 \mathrm{VaR}_u\,\mathrm{d}u\tag{93}$$

（这个式子把名字解释了：Expected **Shortfall**，就是尾巴上那段亏空的平均值。）别名还有 CVaR（条件 VaR）、CTE（条件尾部期望）、ETL（预期尾部损失），说的都是同一件事。

若损失服从正态分布，这个积分有闭式解：

$$\mathrm{ES}_c = \mu + \sigma\,\frac{\varphi(z_c)}{1-c}\tag{94}$$

其中 $\varphi(\cdot)$ 是标准正态密度、$z_c = \Phi^{-1}(c)$。

（闭式的推导在第八节）

99% 时系数为 $\varphi(2.3264)/0.01 \approx 2.665$，比 VaR 的 2.3264 大，所以

$$\mathrm{ES}_c \ge \mathrm{VaR}_c\tag{95}$$

恒成立：ES 是门口的平均值，VaR 只是门口那道线。

表 5 把两者摆在一起。

<p align="center">表5&nbsp;&nbsp;VaR 与 ES 的对比</p>

| 特性 | VaR | ES |
| --- | --- | --- |
| 回答的问题 | 大概率不会超过多少 | 超过了平均会亏多少 |
| 是否一致风险度量 | 否（不满足次可加性） | 是（四条全满足） |
| 是否看尾部 | 否，只是一个分位数 | 是，尾部的条件期望 |
| 监管地位 | 巴塞尔 II 采用 | 巴塞尔 III 转向采用 |

> 注：式(92)与式(93)在连续分布下等价；遇到离散分布（比如两只债券那种跳变），"条件期望"更好用，因为分位点可能正好落在概率质量的跳变处。严格的一致风险度量定义用的是式(93)的积分形式。

**先看一个反例：VaR 为什么不满足次可加性。**　用两只独立债券就能看出来。每只面值 1000，违约概率 $PD = 3\%$，违约回收率 $RR = 0$（违约就全亏），不违约则不亏不赚。

先看单只债券：损失只有两种可能——0（概率 97%）或 1000（概率 3%）。问它的 5% VaR：

$$\Pr\left(L_X > 0\right) = 3\% < 5\% \quad\Longrightarrow\quad \mathrm{VaR}_{5\%}(X) = 0\tag{96}$$

同理 $\mathrm{VaR}_{5\%}(Y) = 0$。

再看合在一起的组合 $X + Y$。两只独立，联合分布只有三种情形（表 6）。

<p align="center">表6&nbsp;&nbsp;两只独立债券的联合损失分布</p>

| 情形 | 损失 | 概率 | 累计概率 |
| --- | --- | --- | --- |
| 都违约 | 2000 | $0.09\%$ | $0.09\%$ |
| 恰好一只违约 | 1000 | $5.82\%$ | $5.91\%$ |
| 都不违约 | 0 | $94.09\%$ | $100\%$ |

5% 分位落在第二档：累计概率 0.09% 还不到 5%，再加上 5.82% 就到了 5.91%，越过 5%。所以

$$\mathrm{VaR}_{5\%}(X + Y) = 1000 > 0 + 0 = \mathrm{VaR}_{5\%}(X) + \mathrm{VaR}_{5\%}(Y)\tag{97}$$

**合并之后风险反而"变大"了**——明明分散了，指标却报警。是因为单独看时 VaR 那条线落在 0，合并后尾部变厚，线被推到了 1000。

ES 就没有这个问题。单只债券的条件期望是 1000，而组合的尾部期望为

$$\mathrm{ES}_{5\%}(X+Y) = \frac{0.09\% \times 2000 + 5.82\% \times 1000}{5.91\%} \approx 1015\tag{98}$$

于是

$$\mathrm{ES}_{5\%}(X+Y) \approx 1015 \le 1000 + 1000 = \mathrm{ES}_{5\%}(X) + \mathrm{ES}_{5\%}(Y)$$

次可加性成立——分散化真的把风险降下来了。

> 注：这里的 ES 采用式(92)的条件期望口径。若改用式(93)的积分定义，离散分布上会给出不同的数，需要按"超分位数"处理；两种口径在连续分布下一致。

### （六）三种估计方法：参数法、历史模拟、蒙特卡洛

VaR 的定义只说了"损失分布的分位数"，没说这个分布从哪来。三种方法的分歧全在这里。

#### 1. 参数法（方差–协方差 / Delta-Normal）

假设收益率服从正态分布，分布就由 $\mu$、$\sigma$ 两个数决定，分位数直接查表。单资产的完整写法是式(91)。若把 VaR 写成**收益率的阈值**而不是正的损失，就是你笔记里的那个式子：

$$\mathrm{VaR}(X) = \mu + \Phi^{-1}(1-c)\,\sigma\tag{99}$$

注意 $\Phi^{-1}(1-c) = -z_c$，所以它其实是 $\mu - z_c\sigma$——这里的 $\mu$ 是**收益率**均值，所以得到的是 1% 分位点上的收益（通常是负数）。取负号，才得到式(89)那个正的损失。两种写法说的是同一件事，只是符号方向相反；第八节算组合时统一用"损失为正"。

优点：快，一个公式搞定，参数只有两个，还能直接推广到组合（要协方差矩阵）。
缺点：正态假设太强。真实收益有肥尾、有偏度，极端事件比正态预测的更频繁；对期权这类非线性头寸，$\Delta$-Normal 更是直接失真。

#### 2. 历史模拟法（非参数）

思路：假设"历史会重演"。步骤只有三步：

1. 取过去 $T$ 天（比如 500 天）风险因子的每日变化；
2. 把每一天的变化**原样套到今天的组合上**，重估一次，得到 500 个"假如那天重演"的损益；
3. 把这 500 个损益从大到小排好，从最惨的一头数第 $(1-c)\times 500$ 个，就是 VaR。

就是**把过去 500 天出现的事实映射到第 501 天**，于是第 501 天有整整 500 种可能；每种情景下组合价值都能算出来，从最大损失往下排，就得到一个离散分布。

优点：完全非参数，不用估波动率和相关系数（它们已经藏在历史数据里），天然能处理肥尾和相关结构。
缺点：完全依赖那个历史窗口。"万一历史在未来不重演，就没什么用了"——拿疫情期的数据去算经济上行期的 VaR，必然对不上。而且样本量有限：500 天里 5% 分位只有 25 个观测、1% 分位只有 5 个，尾部估计很不稳，这正是监管要求至少 2~3 年数据的原因。

#### 3. 蒙特卡洛模拟

思路：既然历史未必重演，那就"造"出未来。三步：

1. 设定风险因子的随机过程与参数（漂移、波动率、相关性、分布形态）；
2. 生成大量价格路径（比如 10,000 条），每条都推进到目标期限；
3. 每条路径重估一次组合，得到 10,000 个损益，再取分位数。

优点：最灵活，能处理任何分布、任何非线性头寸（期权、路径依赖产品），也能做压力情景。
缺点：设定太多、计算量大；而且它算得准不准，全看设定的模型对不对——设定错了，只是把错误算得更精确。

表 7 是三种方法的横向比较。

<p align="center">表7&nbsp;&nbsp;三种 VaR 估计方法的比较</p>

| 特性 | 参数法 | 历史模拟 | 蒙特卡洛 |
| --- | --- | --- | --- |
| 分布假设 | 需要（正态） | 不需要 | 需要（自定模型） |
| 计算速度 | 最快 | 中等 | 最慢 |
| 肥尾处理 | 差 | 好（历史自带） | 取决于模型 |
| 数据需求 | 均值、协方差 | 2~3 年历史 | 模型参数 |
| 主要风险 | 分布设错 | 窗口选错 | 模型设错 |

### （七）回测与伦敦鲸：模型要验证，治理要跟上

#### 1. 回测：数一数一年里越线几次

VaR 算出来不是就完了，得检验它准不准。做法很土：每天记下当天的 VaR 和第二天实际发生的损益，数一年里**实际损失超过 VaR** 的天数。

99% 的 VaR 意味着 250 个交易日里只该有

$$250 \times (1 - 0.99) = 2.5\tag{100}$$

次越线。实际越线次数太多，就说明模型低估了风险。巴塞尔按次数分三档（表 8）。

<p align="center">表8&nbsp;&nbsp;VaR 回测的交通灯体系（巴塞尔）</p>

| 区域 | 一年越线次数 | 含义 |
| --- | --- | --- |
| 绿灯 | $< 5$ | 模型正常 |
| 黄灯 | $5 \sim 10$ | 需要调查、加强监控 |
| 红灯 | $> 10$ | 模型严重失效，需要增加资本 |

回测的意义不只是"检查数学"：它直接决定这家银行能不能用内部模型算资本，还是被监管按惩罚性系数加钱。

#### 2. 伦敦鲸：模型没错，人错了

2012 年，JPMorgan 伦敦办公室的 CIO 团队在信用衍生品（CDX 指数）上建了一个巨大的合成头寸，交易员 Bruno Iksil 因体量太大被市场称作"伦敦鲸"。头寸最终亏损约 62 亿美元；CEO Jamie Dimon 起初还称其为"茶壶里的风暴"，随后亏损翻倍、再翻倍。

事后调查的重点不在定价，而在治理：**风险限额被例行无视，VaR 模型被重新校准以压低资本要求**（新模型上线后风险数字几乎减半）。讽刺的是，JPMorgan 正是 VaR 与"4:15 报告"的发明者。

所以这件事的教训有两层：一层是**模型风险**——模型可以被调参，而调参本身就是风险；另一层是**文化**——风险管理不是算出来的，是执行出来的。

### （八）组合层面的 VaR 与 ES：协方差矩阵说了算

单个资产的 VaR 会算了，但组合不能把各资产的 VaR 简单相加。两个城市各有一半概率发洪水，若在同一个水系，会一起淹；若在不同水系，同时淹的概率低得多。**相关性决定了分散化到底值多少钱**，而相关性的载体就是协方差矩阵。

现在把前面的记号接上：$\mathbf{w}$ 是权重向量，$\boldsymbol{\mu}$ 是预期回报向量，$\boldsymbol{\Sigma}$ 是协方差矩阵，于是组合的预期回报与标准差为

$$\mu_{\Pi} = \mathbf{w}^{T}\boldsymbol{\mu}, \qquad \sigma_{\Pi} = \sqrt{\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}}\tag{101}$$

#### 1. 组合 VaR 的推导

设资产的（线性）收益服从 $\mathbf{R} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，则组合收益 $R_{\Pi} = \mathbf{w}^{T}\mathbf{R}$ 服从 $N(\mu_{\Pi}, \sigma_{\Pi}^2)$，损失 $L = -R_{\Pi}$ 服从 $N(-\mu_{\Pi}, \sigma_{\Pi}^2)$。按定义 5，

$$\Pr\left(L \le \mathrm{VaR}_c(\mathbf{w})\right) = c\tag{102}$$

把 $L$ 标准化——两边同减 $L$ 的均值 $-\mu_{\Pi}$、再同除标准差 $\sigma_{\Pi}$：

$$\Pr\left(\frac{L + \mu_{\Pi}}{\sigma_{\Pi}} \le \frac{\mathrm{VaR}_c(\mathbf{w}) + \mu_{\Pi}}{\sigma_{\Pi}}\right) = c\tag{103}$$

左边的随机变量已经是标准正态，所以括号里的阈值只能是 $z_c$：

$$\frac{\mathrm{VaR}_c(\mathbf{w}) + \mu_{\Pi}}{\sigma_{\Pi}} = \Phi^{-1}(c) = z_c\tag{104}$$

移项即得

$$\mathrm{VaR}_c(\mathbf{w}) = -\mathbf{w}^{T}\boldsymbol{\mu} + z_c\sqrt{\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}}\tag{105}$$

#### 2. 组合 ES 的推导

ES 的定义是式(93)那个积分。把 VaR 的表达式代进去，注意到 $-\mathbf{w}^{T}\boldsymbol{\mu}$ 与积分变量 $u$ 无关、可以提到积分外面：

$$\mathrm{ES}_c(\mathbf{w}) = \frac{1}{1-c}\int_c^1\left[-\mathbf{w}^{T}\boldsymbol{\mu} + \Phi^{-1}(u)\,\sigma_{\Pi}\right]\mathrm{d}u = -\mathbf{w}^{T}\boldsymbol{\mu} + \frac{\sigma_{\Pi}}{1-c}\int_c^1 \Phi^{-1}(u)\,\mathrm{d}u\tag{106}$$

剩下的就是算这个积分。作换元 $u = \Phi(z)$：由 $\mathrm{d}u = \varphi(z)\,\mathrm{d}z$，下限 $u = c$ 对应 $z = z_c$，上限 $u \to 1$ 对应 $z \to +\infty$，于是

$$\int_c^1 \Phi^{-1}(u)\,\mathrm{d}u = \int_{z_c}^{+\infty} z\,\varphi(z)\,\mathrm{d}z\tag{107}$$

再用标准正态密度的一条性质：$\varphi'(z) = -z\,\varphi(z)$，即 $z\,\varphi(z) = -\varphi'(z)$，所以

$$\int_{z_c}^{+\infty} z\,\varphi(z)\,\mathrm{d}z = \Big[-\varphi(z)\Big]_{z_c}^{+\infty} = \varphi(z_c)\tag{108}$$

（因为 $\varphi(+\infty) = 0$。）代回式(106)：

$$\mathrm{ES}_c(\mathbf{w}) = -\mathbf{w}^{T}\boldsymbol{\mu} + \frac{\varphi(z_c)}{1-c}\sqrt{\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}}\tag{109}$$

99% 时 $z_c = 2.3264$、$\varphi(z_c) = 0.02665$，除以 $1 - c = 0.01$ 得 $2.665$——正是式(94)里那个系数。

> 注：这就是第三节那个"损失为正"的定义重要的原因：组合期望收益为正，$-\mathbf{w}^{T}\boldsymbol{\mu}$ 就是负的，VaR 里把它加上去，得到的才是"净损失"。若用收益率口径，写成 $\mu_{\Pi} - z_c\sigma_{\Pi}$ 就行，符号正好反过来。

#### 3. 算例（Roncalli 三只股票）

三只股票 A、B、C，现价 244、135、315 美元；日预期收益 50bp、30bp、20bp；日波动率 2%、3%、1%；相关系数矩阵为

$$\boldsymbol{\rho} = \begin{pmatrix} 1.00 & & \\ 0.50 & 1.00 & \\ 0.25 & 0.60 & 1.00 \end{pmatrix}\tag{110}$$

组合 #1 持有两只 A、一只 B、一只 C：

$$V = 2 \times 244 + 135 + 315 = 938, \qquad \mathbf{w} = (52.03\%,\ 14.39\%,\ 33.58\%)^{T}\tag{111}$$

算出来 $\mu_{\Pi} = 37.0$ bp、$\sigma_{\Pi} = 1.476\%$。代入式(105)与式(109)：

$$\mathrm{VaR}_{99\%} = -0.370\% + 2.326 \times 1.476\% = 3.06\% \quad (\$28.74)\tag{112}$$

$$\mathrm{ES}_{99\%} = -0.370\% + 2.665 \times 1.476\% = 3.56\% \quad (\$33.44)\tag{113}$$

各置信水平下的结果见表 9。

<p align="center">表9&nbsp;&nbsp;组合 #1 的 VaR 与 ES（置信水平 90%~99.5%）</p>

| 指标 | 90% | 95% | 99% | 99.5% |
| --- | --- | --- | --- | --- |
| VaR（%） | 1.52 | 2.06 | 3.06 | 3.43 |
| VaR（美元） | 14.27 | 19.30 | 28.74 | 32.20 |
| ES（%） | 2.22 | 2.67 | 3.56 | 3.90 |
| ES（美元） | 20.83 | 25.09 | 33.44 | 36.58 |

从这张表能读出三件事：

- 置信水平越高，VaR 与 ES 都单调上升；
- ES 始终压在 VaR 上方，但**两者的差距随置信水平上升而收窄**（90% 差 0.70 个百分点，99.5% 只差 0.47）——尾部越极端，两个指标越靠拢；
- 金额一栏提醒你：百分比风险要乘上组合市值（938 美元）才是钱。风控报告里两者都要给，交易台关心金额，投资经理关心百分比。

> 注：Roncalli 原文的 ES 系数取 2.667，严格算 $\varphi(2.3264)/0.01 \approx 2.665$，所以 3.56% 其实是按 2.665 算出来的；用 2.667 会得到 3.57%。这是四舍五入的差别，不影响结论。

### （九）VaR 的局限与本章收尾

VaR 缺陷还是很多的，就像**一把雨天不好用的雨伞**：平时能遮，真下暴雨就漏。系统地列出来：

- **只是一个分位数，不是尾部。** 超过 VaR 之后亏多少，它不说；ES 才说。
- **对非线性头寸失真。** 期权、可转债这类 delta 会变的头寸，$\Delta$-Normal 一遇大波动就偏。总结为"非线性收益、管理利率、相对价值风险"三类难覆盖的情形。
- **流动性假设不成立。** VaR 的持有期默认你总能按市价平仓；真到危机时市场没深度，10 天也出不完货。
- **假设连续、小幅变动。** 对跳空、熔断、违约这类不连续事件几乎无感。
- **依赖历史。** 参数法依赖"过去能代表未来"，历史模拟更依赖；极端事件往往不在样本里。
- **可被操纵。** 伦敦鲸已经演示过：调一调模型，风险数字就好看一半。

ES 补上了"尾部"和"次可加性"两个洞，但自己也有软肋：对尾部样本极度敏感，估计方差大，需要更长样本或更稳健的方法；而且 ES 也不是"最大损失"，它仍然是期望值，真出事时依然可能亏得更多。

把这些放在一起看，风险度量这件事的边界就清楚了：**指标只能告诉你"有多大的概率亏多少"，不能替你做"亏得起多少"的决定**。限额、资本、对冲比例，都是人在指标之外做的判断。

至此，Module 2 的主线走完了：从 $\mu$–$\sigma$ 图上的一个点，到两资产、$N$ 资产的有效前沿，再用拉格朗日乘子把最优权重解出来；接着用 Black–Litterman 把"预期回报"这个最不可靠的输入修得稳健些；最后用 VaR 与 ES 回答"这个组合到底有多危险"。
