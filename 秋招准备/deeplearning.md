[TOC]

# 1. SGD、Adam、Adagard等优化算法详解
[参考资料](https://arxiv.org/pdf/1609.04747.pdf)
## 1.1. 梯度下降算法的三个变种
根据计算一次梯度更新所使用的数据量大小，梯度下降算法共分为三种:  
- Batch gradient descent;(整个数据集)
- Stochastic gradient descent;（一个数据样本）
- Mini-batch gradient descent;（几个样本）
### 1.1.1. Batch gradient descent
Batch gradient descent的梯度更新公式为:
$$\theta=\theta-\eta \cdot \nabla_{\theta} J(\theta)$$

(打公式我有一个独特的窍门，嘻嘻)
式中$\theta$要更新的参数，$\eta$为学习率,$\nabla_{\theta} J(\theta)$是在整个数据集上损失的梯度。  
**优点**:
- 能保证模型最后收敛于全局最小点（若损失为凸）或局部最小点（损失函数非凸)


**缺点**:
- 每更新一次参数需要计算出整个数据集每一个样本的推断结果并计算损失，计算量大，内存占用大
- 权重更新缓慢
- 无法在线学习
### 1.1.2. Stochastic gradient descent(SGD)
Stochastic gradient descent每次计算损失的梯度时，仅使用一个样本，公式如下：
$$\theta=\theta-\eta \cdot \nabla_{\theta} J\left(\theta ; x^{(i)} ; y^{(i)}\right)$$
**优点**：
- 考虑一种计算情况，假设数据集中每个样本都一样，Batch gradient decent用所有的数据样本来计算损失，存在严重的计算冗余，其实只要计算一个样本即可。尽管这种极端的情况不会出现，但在同一个数据集中，数据必然存在相似性，SGD相比于Batch gradient decent能减少很多计算冗余。
- 研究表明，当学习率较小时，SGD和BGD有相同的收敛点
- 在线学习

**缺点**：
- SGD由于其频繁的权重更新，会导致损失在下降过程中出现较大的波动。但是波动可能使损失函数跳出局部最小值，进入一个小的收敛点。

### 1.1.3. Mini-batch gradient descent
Mini-batch gradient descent则是使用数据集中的几个样本计算损失的梯度。计算公式如下：
$$\theta=\theta-\eta \cdot \nabla_{\theta} J\left(\theta ; x^{(i : i+n)} ; y^{(i : i+n)}\right)$$
**优点**：
- 减小了权重更新的方差，使得损失收敛于一个更稳定的点
- mini-batch的使用使得我们最大化利用GPU的并行计算能力

## 1.2. 梯度下降算法的痛点
- 学习率大小选择困难。太小导致训练缓慢，太大则容易导致难以收敛，在（局部）最小点附近波动
- 算法中所有的参数都使用相同的学习率进行更新，但实际上对于不同的参数，可能有不同的更新需求
- 在高维的损失优化中，很难收敛到每一维数据梯度都接近于0的点，相反，很容易收敛于鞍点或局部最小点

## 1.3. 各种梯度下降算法
### 1.3.1. Momentum
权重更新公式:
$$\begin{aligned} v_{t} &=\gamma v_{t-1}+\eta \nabla_{\theta} J(\theta) \\ \theta &=\theta-v_{t} \end{aligned}$$
记进行第$t$次权重更新时，梯度大小为$g_t$(为了方便表达)，则上式变为：
$$\begin{aligned}v_t &= \gamma v_{t-1} + \eta g_t \\&=\eta g_t + \gamma(\eta g_{t-1}+\gamma v_{t-2})\\&=\eta g_t + \gamma \eta g_{t-1} + \gamma^2v_{t-2}\\
&...\\&=\eta g_t+\gamma \eta g_{t-1}+\gamma ^2 \eta g_{t-2} + ...+\gamma ^ {t-1}\eta g_1\end{aligned}$$
从式中可以知道，权重更新不仅和本次计算出来损失函数的梯度有关，还和之前计算出来每一次的梯度大小有点，距离越近，贡献越大，距离越远，贡献越小（$\gamma<1$）。当$\gamma=0$时则退化成普通的SGD。  
那么，把以往的权重更新考虑进去有什么用呢???
- 若在某个维度多次权重更新方向一致，则会加速该权重的更新
- 若在某个维度权重更新方向一直发生变化，即梯度出现正负交替，多个梯度求和的结果则会减小该方向权重的变化。

但是，Momentum法存在一个缺陷。当某一维度的梯度经过多次同样的方向的权重更新后，达到了最小值。尽管在最小值处的的梯度$g_t=0$,但是！由于累计梯度的原因，$v_{t-1}$并不为0，因此权重会铁憨憨地继续更新，但却是往损失**变大**的方向更新。  
因此，需要给权重的更新提供一点感知下一位置的能力。请看下一个方法！
### 1.3.2. Nesterov accelerated gradient
权重更新公式
$$\begin{array}{l}{v_{t}=\gamma v_{t-1}+\eta \nabla_{\theta} J\left(\theta-\gamma v_{t-1}\right)} \\ {\theta=\theta-v_{t}}\end{array}$$
对比Momentum法，二者在权重更新时，都由两项组成:
- 累计动量项:$\gamma v_{t-1}$
- 损失梯度项:
  $\eta \nabla_{\theta} J\left(\theta-\gamma v_{t-1}\right)$(Nesterov);
  $\eta \nabla_{\theta} J(\theta)$(Momentum)  

二者的区别在于损失梯度项。对于Momentum,是当前位置的梯度损失；对于Nesterov是下一近似位置的梯度损失，正是该项，赋予了该优化方法感知下一刻的能力。
考虑以下这个场景:某一维度的权重经过多次的更新后，累计动量项已相对较大，且接近了最小点，权重再次更新后会导致损失反而变大。看看二者的表现：
- Momentum:计算当前位置损失的梯度（仍与之前的梯度相同），结合累计动量项，更新权重，最终导致损失反而变大
- Nesterov:计算下一位置的近似梯度(过了最小点之后，此时的梯度与之前相反)，结合累计动量项，更新权重。由于损失梯度项变为相反值，一定程度上减少了权重更新的幅度，缓和甚至避免了损失的回升！

![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/2019-05-16_22-32-40.png)
>考虑一个二维权重的更新，水平方向和垂直方向，假设垂直方向的权重已接近最优值，也即权重再增大会导致损失不减反增。  
> Momentum的损失梯度项（红色）垂直方向的分量接近0，但累计动量项仍有很大的垂直分量，两个分量合成后（黄色），垂直方向的权重仍进行了很大的权重更新。  
> Nesterov的损失梯度项是下一位置的近似梯度，由于下一位置梯度和当前相反，因此垂直分量向下。两个分量合成后，垂直方向的分量抵消了一部分，最终垂直方向的权重更新不大。

### 1.3.3. Adagrad
权重更新公式：
$$\theta_{t+1, i}=\theta_{t, i}-\frac{\eta}{\sqrt{G_{t, i i}+\epsilon}} \cdot g_{t, i}$$
式中:
- $\theta_{t+1,i}$是$t+1$时刻$i$方向的权重；
- $G_{t,ii}$是对角阵$G_t$对角上第$i$个元素，该元素大小为$\theta_i$以往每一次梯度的平方和;
- $\epsilon=10^{-8}$,避免除0  

Adagrad方法旨在对不同参数，在不同时刻使用不同的学习率。
对于频繁发生更新(梯度不为0)的权重，其学习率会被调整得较小；而对于更新不频繁的权重，其学习率则会被调整得较大。  
但随着训练的进行，梯度的平方和越来越大，导致最终学习率被调整得很小，导致训练收敛困难。

### 1.3.4. Adadelta
Adadelta旨在处理Adagrad学习速率单调递减的情况。不是使用所有梯度的平方和，而是使用梯度平方的调和均值(与常规均值不一样)，定义如下：
$$E\left[g^{2}\right]_{t}=\gamma E\left[g^{2}\right]_{t-1}+(1-\gamma) g_{t}^{2}$$
相应地，权重变化值变成
$$
\Delta \theta_{t}=-\frac{\eta}{\sqrt{E\left[g^{2}\right]_{t}+\epsilon}} g_{t}
$$
将分母用均方根(RME,先平方、取平均、再求平方根)表示（此处并不是严格的平方根，因为所取的平均是经过调和的）：
$$
\Delta \theta_{t}=-\frac{\eta}{R M S[g]_{t}} g_{t}
$$
与以往不同，此处是对参数的平方进行更新：
$$
E\left[\Delta \theta^{2}\right]_{t}=\gamma E\left[\Delta \theta^{2}\right]_{t-1}+(1-\gamma) \Delta \theta_{t}^{2}
$$
同样可以的到参数更新的均方根：
$$
R M S[\Delta \theta]_{t}=\sqrt{E\left[\Delta \theta^{2}\right]_{t}+\epsilon}
$$
参数更新的均方根反应了以往参数更新的幅度大小，可用来代替学习率，于是，权重更新规则变为：
$$
\begin{array}{c}{\Delta \theta_{t}=-\frac{R M S[\Delta \theta]_{t-1}}{R M S[g]_{t}} g_{t}} \\ {\theta_{t+1}=\theta_{t}+\Delta \theta_{t}}\end{array}
$$
无需设置学习率
### 1.3.5. RMSprop
RMSprop同样是为了解决Adagrad学习率单调递减的情况。和Adadelta初始想法一致，采用权重梯度平方和的均值来调整学习率。更新规则如下：
$$
\begin{array}{c}{E\left[g^{2}\right]_{t}=0.9 E\left[g^{2}\right]_{t-1}+0.1 g_{t}^{2}} \\ {\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{E\left[g^{2}\right]_{t}+\epsilon}} g_{t}}\end{array}
$$

### 1.3.6. Adam
Adam同样是一个调整学习率的优化方法。除了记录梯度的历史平方和，还记录了梯度的指数平均值：
$$
\begin{aligned} m_{t} &=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\ v_{t} &=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2} \end{aligned}
$$
将式子展开：
$$
\begin{aligned}
m_t &=\beta_1m_{t-1}+(1-\beta_1)g_t\\
&=(1-\beta_1)g_t + \beta_1((1-\beta_1)g_{t-1} + \beta_1m_{t-2})\\
&=(1-\beta_1)g_t+(1-\beta_1) \beta_1 g_{t-1}+\beta_1^2m_{t-2}\\
&...\\
&=(1-\beta_1)(g_t+\beta_1g_{t-1}+\beta_1^2g_{t-2}+...+\beta_1^{t-1}g_1)
\end{aligned}
$$
$v_t$同理，都是在指数平均的基础上，再乘以一个缩减的系数。  
通过计算偏差校正的一阶矩和二阶矩估计来抵消偏差：
$$
\begin{aligned} \hat{m}_{t} &=\frac{m_{t}}{1-\beta_{1}^{t}} \\ \hat{v}_{t} &=\frac{v_{t}}{1-\beta_{2}^{t}} \end{aligned}
$$
于是权重的更新公式为：
$$
\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}
$$
一般设置$\beta_1=0.9、\beta_2=0.999、\epsilon=10^{-8}$

## 1.4. 如何挑选优化器
对于调整学习率的方法，如Adagrad、Adadelta、RMSprop、Adam等，实验表明Adam的效果好于其他方法。有趣的是，不带动量SGD搭配衰减的学习率经常能找到更小值，但会耗费更多的时间，并且也容易陷于鞍点。






# 2. 深度学习中的归一化方法汇总
## 2.1. Batch Normalization
对模型的初始输入进行归一化处理，可以提高模型训练收敛的速度;对神经网络内层的数据进行归一化处理，同样也可以达到加速训练的效果。Batch Normalization 就是归一化的一个重要方法。以下将介绍BN是如何归一化数据，能起到什么样的效果以及产生该效果的原因展开介绍。
### 2.1.1. Batch Normalization原理
(一)前向传播 

在训练阶段：   
(1)对于输入的mini-batch数据$\mathcal{B}=\left\{x_{1 \ldots m}\right\}$，假设shape为(m, C, H, W),计算其在Channel维度上的均值和方差：
$$
\begin{aligned}
\mu_{\mathcal{B}} &= \frac{1}{m} \sum_{i=1}^{m} x_{i}\\
\sigma_{\mathcal{B}}^{2} &= \frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{\mathcal{B}}\right)^{2}
\end{aligned}
$$
(2)根据计算出来的均值和方差，归一化mini-Batch中每一个样本：
$$
\widehat{x}_{i} = \frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}
$$
(3)最后，对归一化后的数据进行一次平移+缩放变换：
$$
y_{i} = \gamma \widehat{x}_{i}+\beta \equiv \mathrm{B} \mathrm{N}_{\gamma, \beta}\left(x_{i}\right)
$$
$\gamma、\beta$是需要学习的参数。  

在测试阶段：  
使用训练集中数据的$\mu_{\mathcal{B}}$和$\sigma_{\mathcal{B}}^{2}$无偏估计作为测试数据归一化变换的均值和方差：
$$
\begin{array}{l}{E(x)=E_{\mathcal{B}}\left(\mu_{\mathcal{B}}\right)} \\ {\operatorname{Var}(x)=\frac{m}{m-1} E_{\mathcal{B}}\left(\sigma_{\mathcal{B}}^{2}\right)}\end{array}
$$
通过记录训练时每一个mini-batch的均值和方差最后取均值得到。   
而在实际运用中，常动态均值和动态方差，通过一个动量参数维护:
$$
\begin{aligned} r \mu_{B_t} &=\beta r \mu_{B_{t-1}}+(1-\beta) \mu_{\mathcal{B}} \\ r \sigma_{B_t}^{2} &=\beta r \sigma_{B_{t-1}}^{2}+(1-\beta) \sigma_{\mathcal{B}}^{2} \end{aligned}
$$
$\beta$一般取0.9.  
因此可以得到测试阶段的变换为：
$$
y = \gamma \frac{x-E(x)}{\sqrt{Var(x)+\epsilon}} + \beta
$$
或：
$$
y = \gamma \frac{x-r \mu_{B_t}}{\sqrt{r \sigma_{B_{t-1}}^{2}+\epsilon}} + \beta
$$

（二）反向传播（梯度计算）  
计算梯度最好的方法是根据前向传播的推导公式，构造出计算图，直观反映变量间的依赖关系。  
前向传播公式：
$$
\begin{aligned}
    \mu_{\mathcal{B}} &= \frac{1}{m} \sum_{i=1}^{m} x_{i}\\
\sigma_{\mathcal{B}}^{2} &= \frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{\mathcal{B}}\right)^{2}\\
\widehat{x}_{i} &= \frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}\\
y_{i} &= \gamma \widehat{x}_{i}+\beta
\end{aligned}
$$
计算图：  
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/2019-05-18_10-00-59.png)  
*黑色线表示前向传播的关系，橙色先表示反向传播的关系*
利用计算图计算梯度的套路：
- 先计算离已知梯度近的变量的梯度，这样在计算远一点变量的梯度时，可能可以直接利用已经计算好的梯度；
- 一个变量有几个出边（向外延伸的边），其梯度就由几项相加。

当前已知$\frac{\partial l}{\partial y_{i}}$,要计算loss对图中每一个变量的梯度.
按照由近及远的方法,依次算$\gamma,\beta,\hat{x_i}$.由于$\sigma_{\mathcal{B}}^2$依赖于$\mu_{\mathcal{B}}$,所以先计算$\sigma_{\mathcal{B}}^2$,再计算$\mu_{\mathcal{B}}$,最后计算$x_i
$  
$\gamma$:表面一个出边,实际上有m个出边,因为每一个$y_i$的计算都与$\gamma$有关,因此
$$
\begin{aligned}
\frac{\partial l}{\partial \gamma} &= \sum_{i=1}^m \frac{\partial l}{\partial y_i}\frac{\partial y_i}{\partial \gamma}\\
&= \sum_{i=1}^m \frac{\partial l}{\partial y_i} \hat{x_i}
\end{aligned}
$$
$\beta$:同理
$$
\begin{aligned}
    \frac{\partial l}{\partial \beta} &= \sum_{i=1}^m \frac{\partial l}{\partial y_i} \frac{\partial y_i}{\partial \beta}\\
    &= \sum_{i=1}^m \frac{\partial l}{\partial y_i}
\end{aligned}
$$  
$\hat{x_i}$:一条出边
$$
\begin{aligned}
    \frac{\partial l}{\partial \hat{x_i}} & =\frac{\partial l}{\partial y_i}\frac{\partial y_i}{\partial \hat{x_i}}\\
    & =\frac{\partial l}{\partial y_i}\gamma
\end{aligned}
$$
$\sigma_{\mathcal{B}}^2$:m条出边,每一个$\hat{x_i}$的计算都依赖于$\sigma_{\mathcal{B}}^2$.找到其到loss的路径:$\sigma_{\mathcal{B}}^2\rightarrow\hat{x_i}\rightarrow y_i \rightarrow loss$,由于$\hat{x_i}$关于loss的梯度已经计算好了,所以路径为$\sigma_{\mathcal{B}}^2\rightarrow\hat{x_i} \rightarrow loss$,因此
$$
\begin{aligned}
    \frac{\partial l}{\partial \sigma_{\mathcal{B}}^2} &= \sum_{i=1}^m \frac{\partial l}{\partial \hat{x_i}}\frac{\hat{x_i}}{\sigma_{\mathcal{B}}^2}\\
    &= \sum_{i=1}^m \frac{\partial l}{\partial \hat{x_i}} \frac{-1}{2}(x_i-\mu_{\mathcal{B}})(\sigma_{\mathcal{B}}^2+\epsilon)^{-\frac{3}{2}}
\end{aligned}
$$
$\mu_{\mathcal{B}}$出边有m + 1条.路径:$\mu_{\mathcal{B}} \rightarrow \sigma_{\mathcal{B}}^2 \rightarrow loss$, $\mu_{\mathcal{B}} \rightarrow \hat{x_i} \rightarrow loss$,因此;
$$
\begin{aligned}
    \frac{\partial l}{\partial \mu_{\mathcal{B}}} &=\frac{\partial l}{\partial \sigma_{\mathcal{B}}^2}\frac{\sigma_{\mathcal{B}}^2}{\partial \mu_{\mathcal{B}}} + \sum_{i=1}^m\frac{\partial l}{\partial \hat{x_i}}\frac{\partial \hat{x_i}}{\partial \mu_{\mathcal{B}}}\\
    &=\frac{\partial l}{\partial \sigma_{\mathcal{B}}^2} \frac{-2}{m}\sum_{i=1}^m(x_i-\mu_{\mathcal{B}}) + \sum_{i=1}^m\frac{\partial l}{\partial \hat{x_i}} (\frac{-1}{\sqrt{\sigma_{\mathcal{B}}^2+\epsilon}})\\
\end{aligned}
$$ 
$x_i$:有3条边,路径:$x_i\rightarrow\mu_{\mathcal{B}}\rightarrow loss$, $x_i\rightarrow \sigma_{\mathcal{B}}^2\rightarrow loss$,$x_i\rightarrow \hat{x_i}\rightarrow loss$
$$
\begin{aligned}
    \frac{\partial l}{\partial x_i} &= \frac{\partial l}{\partial \mu_{\mathcal{B}}}\frac{\partial \mu_{\mathcal{B}}}{\partial x_i} + \frac{\partial l}{\sigma_{\mathcal{B}}^2}\frac{\sigma_{\mathcal{B}}^2}{\partial x_i} + \frac{\partial l}{\hat{x_i}}\frac{\hat{x_i}}{\partial x_i}\\
    &=\frac{\partial l}{\partial \mu_{\mathcal{B}}}\frac{1}{m} + \frac{\partial l}{\partial \sigma_{\mathcal{B}}^2}\frac{-2}{m}(x_i-\mu_{\mathcal{B}}) + \frac{\partial l}{\partial \hat{x_i}}\frac{1}{\sqrt{\sigma_{\mathcal{B}}^2+\epsilon}}

\end{aligned}
$$

求导完毕!!!  
附上代码实现:
```python
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    
    if mode == 'train':

        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        out_ = (x - sample_mean) / np.sqrt(sample_var + eps)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        out = gamma * out_ + beta
        cache = (out_, x, sample_var, sample_mean, eps, gamma, beta)

    elif mode == 'test':

        scale = gamma / np.sqrt(running_var + eps)
        out = x * scale + (beta - running_mean * scale)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

```
```python
def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    out_, x, sample_var, sample_mean, eps, gamma, beta = cache

    N = x.shape[0]
    dout_ = gamma * dout
    dvar = np.sum(dout_ * (x - sample_mean) * -0.5 * (sample_var + eps) ** -1.5, axis=0)
    dx_ = 1 / np.sqrt(sample_var + eps)
    dvar_ = 2 * (x - sample_mean) / N

    # intermediate for convenient calculation
    di = dout_ * dx_ + dvar * dvar_
    dmean = -1 * np.sum(di, axis=0)
    dmean_ = np.ones_like(x) / N

    dx = di + dmean * dmean_
    dgamma = np.sum(dout * out_, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta
```
### 2.1.2. Batch Normalization的效果及其证明
(一) 减小Internal Covariate Shift的影响, 权重的更新更加稳健.  
对于不带BN的网络,当权重发生更新后,神经元的输出会发生变化,也即下一层神经元的输入发生了变化.随着网络的加深,该影响越来越大,导致在每一轮迭代训练时,神经元接受的输入有很大的变化,此称为Internal Covariate Shift. 而BatchNormalization通过归一化和仿射变换(平移+缩放),使得每一层神经元的输入有近似的分布.  
假设某一层神经网络为:
$$
\mathbf{H_{i+1}} = \mathbf{W} \mathbf{H_i} + \mathbf{b}
$$
对权重的导数为:
$$
\frac{\partial l}{\partial \mathbf{W}} = \frac{\partial l}{\partial \mathbf{H_{i+1}}} \mathbf{H_i}^T
$$
对权重进行更新:
$$
\mathbf{W} \leftarrow\mathbf{W} - \eta  \frac{\partial l}{\partial \mathbf{H_{i+1}}} \mathbf{H_i}^T
$$
可见,当上一层神经元的输入($\mathbf{H_i}$)变化较大时,权重的更新变化波动大.  
(二)batch Normalization具有权重伸缩不变性,可以有效提高反向传播的效率,同时还具有参数正则化的效果.  
记BN为:
$$
Norm(\mathbf{Wx}) = = \mathbf{g} \cdot \frac{\mathbf{W} \mathbf{x}-\mu}{\sigma}+\mathbf{b}
$$
为什么具有权重不变性?   $\downarrow$  
假设权重按照常量$\lambda$进行伸缩,则其对应的均值和方差也会按比例伸缩,于是有:
$$
\begin{aligned}
Norm\left(\mathbf{W}^{\prime} \mathbf{x}\right) &= \mathbf{g} \cdot \frac{\mathbf{W}^{\prime} \mathbf{x}-\mu^{\prime}}{\sigma^{\prime}}+\mathbf{b}\\&=\mathbf{g} \cdot \frac{\lambda \mathbf{W} \mathbf{x}-\lambda \mu}{\lambda \sigma}+\mathbf{b}\\
&= \mathbf{g} \cdot \frac{\mathbf{W} \mathbf{x}-\mu}{\sigma}+\mathbf{b}\\
&=Norm(\mathbf{W} \mathbf{x}) 
\end{aligned}
$$

为什么能提高反向传播的效率?$\downarrow$  
考虑权重发生伸缩后,梯度的变化:
为方便(公式打累了),记$\mathbf{y}=Norm(\mathbf{Wx})$
$$
\begin{aligned}
    \frac{\partial l}{\partial \mathbf{x}} &= \frac{\partial l}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}}\\
    &=\frac{\partial l}{\partial \mathbf{y}} \frac{\partial (\mathbf{g} \cdot \frac{\mathbf{W} \mathbf{x}-\mu}{\sigma}+\mathbf{b}))}{\partial \mathbf{x}}\\
    &=\frac{\partial l}{\partial \mathbf{y}} \frac{\mathbf{g}\cdot\mathbf{W}}{\sigma}\\
    &=\frac{\partial l}{\partial \mathbf{y}} \frac{\mathbf{g}\cdot \lambda\mathbf{W}}{\lambda\sigma}
\end{aligned}
$$
可以发现,当权重发生伸缩时,相应的$\sigma$也会发生伸缩,最终抵消掉了权重伸缩的影响.  
考虑更一般的情况,当该层权重较大(小)时,相应$\sigma$也较大(小),最终梯度的传递受到权重影响被减弱,提高了梯度反向传播的效率.同时,$g$也是可训练的参数,起到自适应调节梯度大小的作用.

为什么具有参数正则化的作用?$\downarrow$
计算对权重的梯度:
$$
\begin{aligned}
\frac{\partial l}{\partial \mathbf{W}} &= \frac{\partial l}{\partial \mathbf{y}}   \frac{\partial\mathbf{y}}{\partial\mathbf{W}}\\
&=\frac{\partial l}{\partial \mathbf{y}} \frac{\partial(\mathbf{g} \cdot \frac{\mathbf{W} \mathbf{x}-\mu}{\sigma}+\mathbf{b})}{\partial \mathbf{W}}\\
&=\frac{\partial l}{\partial \mathbf{y}} \frac{\mathbf{g}\cdot \mathbf{x}^T}{\sigma}
\end{aligned}
$$
假设该层权重较大,则相应$\sigma$也更大,计算出来梯度更小,相应地,$\mathbf{W}$的变化值也越小,从而权重的变化更为稳定.但当权重较小时,相应$\sigma$较小,梯度相应会更大,权重变化也会变大.  

### 2.1.3. 为什么要加$\gamma$,$\beta$
**为了保证模型的表达能力不会因为规范化而下降**.   
如果激活函数为Sigmoid,则规范化后的数据会被映射到非饱和区(线性区),仅利用到线性变化能力会降低神经网络的表达能力.
如果激活函数使用的是ReLU,则规范化后的数据会固定地有一半被置为0.而可学习参数$\beta$能通过梯度下降调整被激活的比例,提高了神经网络的表达能力.  

**经过归一化后再仿射变换,会不会跟没变一样**  
首先,新参数的引入,将原来输入的分布囊括进去,而且能表达更多的分布;其次,$\mathbf{x}$的均值和方差和浅层的神经网络有复杂的关联,归一化之后变成$\hat{\mathbf{x}}$,再进行仿射变换$\mathbf{y}=\mathbf{g} \cdot \hat{\mathbf{x}}+\mathbf{b}$,能去除与浅层网络的密切耦合;最后新参数可以通过梯度下降来学习,形成利于模型表达的分布.

## 2.2. Normalization 方法对比
### 2.2.1. BN方法的不足之处
BN的不足根源在于测试时使用的两个在训练阶段时维护的参数,均值$\mu_{\mathcal{B}}$和修正的方差$\sigma_{\mathcal{B}}$.
- 当训练集和测试集的数据分布不一致的时候,训练集和测试集的均值和方差存在较大差异,最终影响模型预测精度;
- 即使训练集和测试集的数据分布相对一致,但当batch-size较小时,计算出来的均值和方差所具有的统计意义不强,同样会在预测的时候影响模型准确度.
- 一般BN合适的batch-size是32,但对于图像分割,目标检测这些任务对于显存要求大,机器无法满足大batch-size的要求,往往只能设置为1-2.而随着batch-size不断变小,误差越来越大.如下图.

![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_009.png)

### 2.2.2. 独立于batch 的归一化方法
独立于batch进行归一化的方法有 Layer Normalization, Instance Normalization 和 Group Normalization.对比BN,结合一下几张图说明每一种Normalization的归一化方法.  
考虑一个Batch=3, channels number=6, H, W 的tensor.绿色表示归一化的范围.
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/2019-05-19_11-49-04.png)  
- **BN**:在batch,H,W的维度进行归一化,即同一个Channel的feature maps进行归一化.共操作了6(channel number)次归一化;  
- **LN**:在Channel, H, W的维度进行归一化,即对mini-batch内的每一个样本进行归一化.共操作了3(Batch-size)次归一化;  
- **IN**:在H,W维度进行归一化,即对每一个features map进行归一化,共操作了6\*3(channels number \* batch-size)次归一化  
- **GN**:在一个样本的多个channels内分组,分成多组channels,在各组内进行归一化.图中将channels分成了两组(G=2), 因此归一化操作次数为:2\*3(G\*Batch-size).  

### 2.2.3. Group Normalization
在深度学习没有火起来之前，提取特征通常是使用SIFT，HOG和GIST特征，这些特征有一个共性，都具有按group表示的特性.因此尝试在group内进行归一化.  
和LN和IN的联系:
- 当分组数目为1(G=1),GN变成LN;
- 当分组数目为通道数6(G=channels number),GN变成IN  

实现:
```python
def GroupNorm(x, gamma, beta, G, eps=1e-5):
    # x: 输入特征，shape:[N, C, H, W]
    # gamma, beta: scale 和 offset，shape: [1, C, 1, 1]
    # G: GN 的 groups 数

    N, C, H, W = x.shape
    x = tf.reshape(x, [N, G, C//G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x -mean) / tf.sqrt(var + eps)

    x = tf.reshape(x, [N, C, H, W])

    return x * gamma + beta
```
需要注意的是,不同于BN,GN不再需要维护归一化时的均值和方差.用于仿射变换的$\gamma$和$\beta$不依赖于batch-size,对相同的通道,进行同样的仿射变换.
### 2.2.4. 四种方法效果比较
当Batch-size都为32时,BN的效果最好,GN次之:
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_010.png)

当Batch-zize变小时,GN的稳定性和效果好于BN:
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_011.png)

当GN取不同Group,或Group内进行归一化channels数变化时:
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_013.png)  
可见,当Group取32,或channels数为16时效果较好,都好于LN和IN.
# 3. 分类网络
## 3.1. ResNeXt
ResNeXt是在ResNet之后进一步提出的分类网络,看它的名字就知道了.ResNeXt,可以看成ResNet和Next,果然写文章的都是一群取名鬼才.  
论文:[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)  
[代码](1https://github.com/facebookresearch/ResNeXt)
### 3.1.1. 思路来源
作者认为,尽管增大网络的深度和width(指features maps的个数)可能提高模型的准确率,但与此同时带来的是网络的复杂度不断增大.  
作者发现,增加网络内部transformation的个数,能够在不增大模型复杂度的前提下,提高模型的精度.增加网络内部transformation的个数即通过增加多个分支来实现.
### 3.1.2. 多分支的通用表达
作者给出了一个表达式,用来囊括表达各种多分支的结构.  
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_015.png)  

如图中所示,共有$C(C=32)$个分支,每个分支记为$T_i$,对应一种变换(Transformation),每个变换里可能有不同的结构.输入经过每个分支的变换之后,再加和在一起,因此多个分支的变换表达式为:
$$
F(\bold x) = \sum_{i=1}^CT_i(\bold x)
$$
另外还有一条直连结构(shortcut),因此整个模块的表达式为:  
$$
\bold y = \bold x + \sum_{i=1}^CT_i(\bold x)
$$

### 3.1.3. 多分支模块的设计与等效表达
本着大道至简的原则,作者认为去除"五花八门"的分支(如下图),转而采用同样的分支结构,这样也可以提高模型的鲁棒性,避免在某个数据集上过拟合.
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_016.png)

作者提出的分支模块如下图所示,每个分支具有相同的结构.  
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_015.png)
- 先用$1*1$的卷积降维
- 进行$3*3$的卷积
- 再用$1*1$的卷积升维

下面对该模块进行等效变换.  
记3\*3卷积的输出为:$h^3_{b,i}$,其中3表示3\*3的卷积,$b$表示分支的编号($b=1,2,...,32$),$i$表示特征图channel的编号($i=1,2,3,4$).所有分支3*3卷积之后共计有$32*4=128$个$h$.  
$h^1_{b, j}$ 表示1\*1卷积之后的输出,$j=1,2,...,256$.  
某个分支1\*1卷积后的一个输出可表示为:
$$
h^1_{b, j}=\sum_{i=1}^4w_{b,i,j}h^3_{b,i}$$
将所有分支的结果求和,得到所有分支的汇总输出:

$$
\begin{aligned}
y_j &=\sum_{b=1}^{32}h^1_{b, j}    \\
&=\sum_{b=1}^{32}\sum_{i=1}^4w_{b,i,j}h^3_{b,i}
\end{aligned}
$$
该表达式的本质就是,将每一个分支的每一个通道乘上一个系数,再求和!  
也就是说,在计算的时候,其实和分支关系不大.如果对每个分支的每个通道进行统一的编号,编号方式为$k = (b-1)*4+i$,那么上式可改写成:
$$
y_j = \sum_{k=1}^{128}w_{k,j}h_k^3
$$
根据这个式子,就可以把结构等效成下面这个结构了.  
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_017.png)
- 先将每个分支3\*3卷积的结果堆叠在一起,相当于根据分支对通道进行统一的编号.
- 再将堆叠的结果进行1\*1的卷积计算.  


concat之后的输出为:
$$
h_k^3,\\k=1,2,...,128
$$
1\*1卷积之后的结果为:
$$
y_j = \sum_{k=1}^{128}w_{k,j}h_k^3
$$
和第一种结构的式子完全等价!  



接下来再进行一次变换.
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_018.png)
解释:
- 每个分支的1\*1的卷积可以组合在一起计算,计算之后每个分支分4个channels.由于每一个输出channel是依据输入channel进行全连接计算的,因此每个输出Channel之间是独立的.所以分开计算和一起计算等价.
- 再看3\*3的卷积,每个分支是在各自的4个channel中计算,正好就是Group Convolution(挖个坑,接下来的博客会专门总结各种卷积方式)的计算方式.

### 3.1.4. 对比ResNet
ResNet的BottleNeck结构:  
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_019.png)

ResNeXt的BottleNeck结构:  
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_018.png)

乍一看,两个也长得差不多,主要有两点变化:
- 通道数增加
- 普通卷积换成了分组卷积

二者参数量对比:  

网络|参数量
:-:|:-:
ResNet|$1*1*256*64 + 3*3*64*64+1*1*64*256=69632$
ResNeXt|$1*1*256*128+3*3*(128/32)*(128/32)*32 + 1*1*128*256=70144$
二者参数量相当.  

但是ResNeXt增大了通道数和分支数,相当于增大了网络的宽度(width),提高了网络的表达能力(Model Capacity).所以在保持模型复杂度的前提下,提高了模型的精度.  

再回头看,其实就是把普通卷积换成了分组卷积 = = .但作者能够关联到网络分支,网络结构的表达能力,并给出一系列的论证,并做了大量的实验来验证想法,或许这就是艺术吧!

## 3.2. DenseNet
DenseNet是在ResNet之后的一个分类网络,连接方式的改变,使其在各大数据集上取得比ResNet更好的效果.
### 网络结构
以DenseNet-121为例,介绍网络的结构细节.  

![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/fig_020.png)  
网络结构一开始与ResNet类似,先进行一个大尺度的卷积,再接一个池化层;随后接上连续几个子模块(Dense Block和Transitin Layer);最后接上一个池化和全连接.  
以下重点介绍Dense Block 和 Transition layer.  
**Dense Block**  
从图中可以看到,第一个DenseBlock包含6个[1\*1 conv, 3\*3 conv], 此处的[1\*1 conv, 3\*3 conv]即为Bottleneck结构,具体如下:  
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/2019-05-28_15-36-12.png)

BottleNeck包含两个卷积,和常规的卷积-BN-ReLU模式不一样,此处的BN-ReLU放在卷积前面(有文章实验证明过这样效果更好).1\*1卷积的输出通道数是$4*k$,此处k是一个特征图增长系数,可以理解成BottleNeck贡献的特征图个数;3\*3卷积的输出通道数是$k$;整个模块的输出是将输入和3\*3卷积的输出堆叠在一起(concat),即共输出$in\_channels + k$个通道.    

接下来介绍为什么网络是密集连接的!

DenseNet-121的第一个DenseBlock包含了6个BottleNeck,BottleNeck之间是串联在一起的.
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/2019-05-28_16-18-46.png)

图中横向表示特征图,纵向表示BottleNecks.整个DenseBlck的输入通道个数为$n_0$.相应颜色的BottleNeck产生对应颜色的$k$个特征图.由于BottleNeck的输出将本身的输出($k$个通道)和输入concat在一起了,所以输出为$n_0+k$个通道,以此类推,后续通道数每经过一个BottleNeck,通道数增加$k$个(所以称$k$为通道增加系数).

以粉红色BottleNeck为例,说明整个DenseBlock为密集连接.  
![](https://raw.githubusercontent.com/EEEGUI/ImageBed/master/img/2019-05-28_16-15-58.png)
仔细观察粉红色BotteNeck的输入,其实是来自于前面每一层BottleNeck输出和原始输入的堆叠.而且每一个BottleNeck的输入都是其前面所有层输出的堆叠,这就是DenseNet为什么是密集连接的原因,也是DenseNet取得良好效果的原因:
- 传递到粉红色BottleNeck的梯度,能直接传递到其前面各层BottleNeck中,一方面避免了梯度消失,另一方面加快了参数的迭代速度,提高了训练效率
- 网络在前向传播过程中,每个BottleNeck利用其前面所有网络层的输出结果作为输入,产生$k$个特征图.作者认为这是一个利用当前"集体成果"(前面所有层的输出),产生新特征,同时再将新特征加入"集体成果"中,不断成长壮大的过程.
- ResNet中BottleNeck的输出和原始输入的融合方式采用的是相加,作者认为这种方式会破坏已经学到的特征,因此采用concat的方式.

**Transition layer**  
Transition Layer就比较平平无奇了,是一个卷积加池化,用于整合学到的特征,降低特征图的尺寸.

# 4. 激活函数sigmoid、ReLu、对比
# 5. Focal Loss
# 6. 各类卷积　Group　Convolution、DepthWise Convolution
Depthwise卷积实际速度与理论速度差距较大，解释原因
# 7. 注意力机制
# 8. 模型压缩
# 9. SPP,YOLO
