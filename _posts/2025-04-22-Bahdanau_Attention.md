---
title: Bahdanau Attention
date: 2025-04-22 10:00:00 +0800
categories: [Paper Reading, Replicating]
tags: [Replicate, Sequence, Attention, Paper_with_Code]
math: true
---

Orginal Paper:  [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473)

在此访问[源代码](https://github.com/zhengyang-2002/Paper_reading-Replicating/blob/main/Code/Seq2seq.ipynb)



### 1. 引言

Seq2seq的出现为NMT(Neural Machine Translation)提供了一个完美的，端到端训练的范本，但是同时也存在了很多不成熟的地方有待改进。本文中就提到了传统Seq2seq模型存在的关键问题：

+ 传统的Seq2seq需要compress所有的输入信息到一个固定长度的向量，这大大限制了它在处理长序列任务时的性能。

基于这个问题，本文提出了一个关于新版Seq2seq模型的设想：

+ 这个新版的Seq2seq不仅可以利用一个固定长度的信息向量来解码，它还能进行软路由，即回顾输入的某个子集
+ “软路由”意味着模型需要根据上下文来自动选取合适的子集，并且这个子集可能是不明确的（模型需要想办法表征输入子集）
+ “回顾输入的某个子集”则意味着模型的信息通路需要进行调整（并非一条线到底的信息流）

基于以上设想，作者为这篇工作取名为Neural Machine Translation（点明任务类型）By Jointly Learning To Align And Translation（方法特点）。同时这篇工作中的方法还有一个响亮的名字，即Bahdanau Attention，也被认为是初代Attention机制。

### 2. 方法

简单来说，Bahdanau Attention在原有的Seq2seq基础之上额外设计了一条信息通路，类似：
![image-20250515160953608](assets/image-20250515160953608.png)

即模型在完成传统的Seq2seq模型训练的同时，“jointly”训练“Align”模型。而这个Align模型通过选择性地回顾input，以解决fixed vector表达能力弱的问题。

#### 2.1 双向编码器

原文中采用了双向RNN模型以及他们的隐藏状态来表征所有input，类似：

![image-20250515161429330](assets/image-20250515161429330.png)

具体来说，就是用两个相反的RNN网络来给每一个token都建立一个hidden_state，并且通过concatenate的方式将它们结合起来，得到所谓的annotation（这个annotation后续会通过align model来选择性地回放给decoder）。关于为什么要使用双向RNN而非普通的单向RNN，文中的解释是他们希望annotation不仅表征在它之前的word，还要表征在其之后的word。

我个人的理解是，如果使用单向RNN，那么位于序列开头的word总会得到更多的表征，这引入了不恰当的先验。同时双向RNN也更容易学习到全面的语序关系。

本文中Bi_Encoder的实现如下：

```python
class Bi_Encoder(nn.Module):
    def __init__(self):
        super(Bi_Encoder, self).__init__()
        def __create_xh(embedding_size, hidden_size):
            return nn.Sequential(
                nn.Linear(embedding_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, hidden_size)
            )
        def __create_hh(hidden_size):
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, hidden_size)
            )
    
        self.xh = __create_xh(cfg.embedding_size, cfg.hidden_size)
        self.hh = __create_hh(cfg.hidden_size)
        self.xh_ = __create_xh(cfg.embedding_size, cfg.hidden_size)
        self.hh_ = __create_hh(cfg.hidden_size)
        
        self.tanh = nn.Tanh()

    def forward(self, seq, input_lengths):
        batch_size, seq_len, embedding_size = seq.size()
        mask = torch.arange(seq_len, device=cfg.device).expand(batch_size, -1) < input_lengths.unsqueeze(1)
        
        hidden_state = torch.zeros(batch_size, cfg.hidden_size, device=cfg.device)
        hidden_state_ = torch.zeros(batch_size, cfg.hidden_size, device=cfg.device)

        forward_hidden_states = torch.zeros(batch_size, seq_len, cfg.hidden_size, device=cfg.device)
        backward_hidden_states = torch.zeros(batch_size, seq_len, cfg.hidden_size, device=cfg.device)

        for t in range(seq_len):
            token, token_ = seq[:,t,:], seq[:,seq_len-t-1,:]
            current_mask, current_mask_ = mask[:, t].unsqueeze(1), mask[:, seq_len-t-1].unsqueeze(1)
            
            temp_hidden_state = self.tanh(self.xh(token)+self.hh(hidden_state))
            temp_hidden_state_ = self.tanh(self.xh_(token_)+self.hh_(hidden_state_))
            
            hidden_state = torch.where(current_mask, temp_hidden_state, hidden_state) # batch_size, embedding_size
            hidden_state_ = torch.where(current_mask_, temp_hidden_state_, hidden_state_)

            forward_hidden_states[:, t, :] = hidden_state
            backward_hidden_states[:, seq_len-t-1, :] = hidden_state_

        annotations = torch.concatenate([forward_hidden_states, backward_hidden_states], dim=-1)
        return hidden_state, annotations
```

实现方法上相较于单向RNN Encoder，只需要重新初始化一份相同的RNN网络，并且把token以逆序进行输入即可。同时逆向RNN并不需要重新定义Mask，直接采用Seq2seq文中提到的Mask的构建方法即可。当然在Bi_Encoder中，我们有两倍的参数需要维护，同时我们还需要额外保存所有的Annotations，这也给模型训练额外添加了成本。

总的来说，Bi_Encoder接收输入token，并且得到每个token的annotation，同时还返回正向RNN的最终hidden_state用以放入Decoder。

#### 2.2 注意力解码器

解码器的主要结构如下，大致包含了三个可训练的模块，在图中由蓝绿红三色来表示。

原文中关于这三个可训练模块的描述是：

$$
 p(y_i \mid y_1, \ldots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, c_i)
$$



$$
s_i = f(s_{i-1}, y_{i-1}, c_i) 
$$

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$


The weight \(\alpha_{ij}\) of each annotation \(h_j\) is computed by


$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$
where

$$
e_{ij} = a(s_{i-1}, h_j)
$$



![image-20250515174855555](assets/image-20250515174855555.png)

网络看起来是比较复杂，但是简而言之，现在的解码器的隐藏状态更新和预测都需要用到Context和Yi-1（上一步的预测）。而文章的重点在于如何生成合适的Context。

文章中的实现是用一个Model a，给每个Annotation计算特定隐藏状态（Si-1）下的Energy，这个Energy是一个标量，后续所有Energy会被Softmax并作为每个Annotation的权重，最后进行加和得到Context，具体计算过程如图。

![image-20250515180742771](assets/image-20250515180742771.png)

至此，Model a就会作为整个Seq2seq模型的一部分，共同参与训练。此时梯度下降不仅仅需要RNN网络的编码能力，也同时训练Model a，让Model a能够基于不同的隐藏状态给每个annotation合理分配权重。强相关的annotation会被Model a赋予更多的Energy（因此也意味着更大的权重），这在文中的Figure 3也有所体现。可以看出英法词汇意思相同的部分（即对角线）高亮，证明Model a会准确选择所需的Annotation。但是也注意到非对角线部分也存在高亮的情况，这可能是Model a额外学习到的软对齐。

<img src="assets/image-20250515175659740.png" alt="image-20250515175659740" style="zoom: 25%;" />



整体Decoder的思路不难理解，我们只需要一步步定义g(),f(),generate_context()即可，具体代码实现如下：

```python
class Attention_Decoder(nn.Module):
    def __init__(self):
        super(Attention_Decoder, self).__init__()
        # Output Matrix
        self.Wg = nn.Sequential(
            nn.Linear(cfg.embedding_size+cfg.hidden_size+cfg.hidden_size*2, cfg.embedding_size//2),
            nn.ReLU(),
            nn.Linear(cfg.embedding_size//2, cfg.embedding_size)
        )
        # Hidden Matrix
        self.Wf = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.embedding_size+cfg.hidden_size*2, cfg.hidden_size//2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size//2, cfg.hidden_size)
        )
        # Alignment Matrix
        self.Wa = nn.Sequential(
            nn.Linear(cfg.hidden_size+cfg.hidden_size*2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )

        self.fc = nn.Linear(cfg.embedding_size, cfg.vocab_size)
          
    def g(self, y_prev, s_i, c_i):
        temp_state = torch.concatenate([y_prev, s_i, c_i], dim=-1)
        predication = self.Wg(temp_state)
        return predication
        
    def f(self, s_prev, y_prev, c_i):
        temp_state = torch.concatenate([s_prev, y_prev, c_i], dim=-1)
        new_hidden_state = self.Wf(temp_state)
        return new_hidden_state

    def a(self, s_prev, annotation):
        temp_state = torch.concatenate([s_prev, annotation], dim=-1)
        energy = self.Wa(temp_state).squeeze(-1)  # consecutive dim_size = 1 could squeeze into one dim_size = 1
        return energy

    def generate_context(self, s_prev, annotations):
        # At this moment, s_prev: batch_size, hidden_size; annotations: batch_size, annotation_number, annotation_size
        batch_size, annotation_number, annotation_size = annotations.size()
        s_prev = s_prev.unsqueeze(1).expand(batch_size, annotation_number, -1)

        energys = self.a(s_prev, annotations) 
        attention_weight_expand = torch.nn.functional.softmax(energys, dim=1).unsqueeze(-1)
        context = torch.sum(annotations*attention_weight_expand, dim=1) # batch_size, annotation_size
        return context

    def forward(self, hidden_state, decode_length, annotations):
        batch_size, _ = hidden_state.size()
        outputs = torch.zeros(batch_size, decode_length, cfg.embedding_size, device=cfg.device)
        y = torch.zeros(batch_size, cfg.embedding_size, device=cfg.device)
        for t in range(decode_length):
            y_prev = y
            c_i = self.generate_context(hidden_state, annotations)
            hidden_state = self.f(hidden_state, y_prev, c_i)
            y = self.g(y_prev, hidden_state, c_i)
            outputs[:,t,:] = y
        outputs = self.fc(outputs)
        return outputs
```



### 3. 实验

这一部分部分的

切断通路

```
------ Round 1/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:42<00:00,  9.00it/s]
[Round 1] PSA: 0.7750, LCSR: 0.7821, GeoMean: 0.7783
------ Round 2/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:34<00:00,  9.31it/s]
[Round 2] PSA: 0.8074, LCSR: 0.8182, GeoMean: 0.8122
------ Round 3/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:43<00:00,  8.94it/s]
[Round 3] PSA: 0.8259, LCSR: 0.8415, GeoMean: 0.8315
------ Round 4/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:36<00:00,  9.24it/s]
[Round 4] PSA: 0.8709, LCSR: 0.8760, GeoMean: 0.8731
------ Round 5/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:37<00:00,  9.21it/s]
[Round 5] PSA: 0.7609, LCSR: 0.7939, GeoMean: 0.7754
------ Round 6/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:37<00:00,  9.19it/s]
[Round 6] PSA: 0.8724, LCSR: 0.8788, GeoMean: 0.8753
------ Round 7/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:37<00:00,  9.18it/s]
[Round 7] PSA: 0.8619, LCSR: 0.8684, GeoMean: 0.8649
------ Round 8/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:33<00:00,  9.37it/s]
[Round 8] PSA: 0.8559, LCSR: 0.8603, GeoMean: 0.8579
------ Round 9/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:36<00:00,  9.25it/s]
[Round 9] PSA: 0.8741, LCSR: 0.8845, GeoMean: 0.8788
------ Round 10/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:26<00:00,  9.70it/s]
[Round 10] PSA: 0.4754, LCSR: 0.5052, GeoMean: 0.4851

=== Final Average Metrics ===
PSA: 0.7980
LCSR: 0.8109
GeoMean: 0.8033
```

初始的

```
------ Round 1/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:15<00:00, 10.21it/s]
[Round 1] PSA: 0.8625, LCSR: 0.8661, GeoMean: 0.8642
------ Round 2/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:14<00:00, 10.28it/s]
[Round 2] PSA: 0.6001, LCSR: 0.6758, GeoMean: 0.6316
------ Round 3/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:10<00:00, 10.47it/s]
[Round 3] PSA: 0.6773, LCSR: 0.6973, GeoMean: 0.6859
------ Round 4/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:13<00:00, 10.33it/s]
[Round 4] PSA: 0.8597, LCSR: 0.8657, GeoMean: 0.8625
------ Round 5/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:22<00:00,  9.90it/s]
[Round 5] PSA: 0.7643, LCSR: 0.7787, GeoMean: 0.7709
------ Round 6/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:25<00:00,  9.74it/s]
[Round 6] PSA: 0.7833, LCSR: 0.8193, GeoMean: 0.7995
------ Round 7/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:18<00:00, 10.08it/s]
[Round 7] PSA: 0.5488, LCSR: 0.5688, GeoMean: 0.5573
------ Round 8/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:31<00:00,  9.44it/s]
[Round 8] PSA: 0.8470, LCSR: 0.8541, GeoMean: 0.8502
------ Round 9/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:24<00:00,  9.79it/s]
[Round 9] PSA: 0.6264, LCSR: 0.6553, GeoMean: 0.6385
------ Round 10/10 ------
Training: 100%|█████████████████████████████████████████████████████████████████████| 2000/2000 [03:25<00:00,  9.72it/s]
[Round 10] PSA: 0.7911, LCSR: 0.8214, GeoMean: 0.8045

=== Final Average Metrics ===
PSA: 0.7361
LCSR: 0.7603
GeoMean: 0.7465
```

### 4. 额外改进

#### 4.1 切断编码器和解码器之间的通路

实际上Decoder从Align Model得到的Context中的信息不会比Encoder的最后一个隐藏状态要少，事实上，训练得当的Align Model几乎可以提供任何Decoder所需要的Input Sentence的信息。为了验证我的猜想，我在此处人为切断Encoder和Decoder的直接信息通路，并验证切断前后模型的性能。

![image-20250516161607541](assets/image-20250516161607541.png)

具体的代码改动：只需要修改Decoder的forward函数

```python
def forward(self, hidden_state, decode_length, annotations):
    hidden_state = torch.zeros_like(hidden_state) #添加这一行即可
   	(......)
```

#### 4.2 删去“残差链接”的信息通路

Bahdanau Attention在Residual Connection之前提出，但是Bahdanau Attention也存在类似Residual Connection的思想。比如Align Model得到的Context不仅放入f()用于生成Hidden State，后续还和Hidden State一起放入g()中进行预测。同时这一现象也出现在yi-1中，它不仅参与Si的生产，也同时参与yi的预测。这两个额外的通路有点类似于Residual Connection的思想，即让f()学习yi-1到yi之间的差异，学习Context到yi的差异。本节将去除这两条额外的“捷径”，并测试模型性能的变化。去除“捷径”的Bahdanau Attention结构如图：
![image-20250516173114185](assets/image-20250516173114185.png)

#### 4.3 禁止yi-1参与后续计算

在我看来，让yi-1参与f()的计算意味着需要让f()学习到部分g()的范式，尽管这可能能提升模型的性能，但是我觉得从逻辑上来说不够通顺（尤其是让隐藏状态显示地学习predict），因此此节我将把yi的信息通路删去，结构如图：

![image-20250516173728460](assets/image-20250516173728460.png)
