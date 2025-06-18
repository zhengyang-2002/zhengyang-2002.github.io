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

#### 4.4 让Context和yi-1直接参与预测

这样做的好处是可以让信息流更加纯粹，即影藏状态只为单纯的RNN，而如何处理Context和yi-1则交给g()来处理。同时这种方法还可以降低计算复杂度。

![image-20250523163434795](assets/image-20250523163434795.png)



``

#### 4.5 结果分析

![Bahdanau_Attnetion_Experiment (1)](assets/Bahdanau_Attnetion_Experiment (1).svg)







---




Here is the formatted table based on the provided data:

| Decoder Type      | Max Length | Round | PSA    | LCSR   | GeoMean | Training Time (min) | Training Speed (it/s) |
| ----------------- | ---------- | ----- | ------ | ------ | ------- | ------------------- | --------------------- |
| Attention_Decoder | 10         | 1     | 0.7149 | 0.7151 | **0.7150** | 3.08                | 26.96                 |
| Attention_Decoder | 10         | 2     | 0.7198 | 0.7203 | 0.7200  | 3.13                | 26.52                 |
| Attention_Decoder | 10         | 3     | 0.7133 | 0.7134 | 0.7134  | 3.02                | 27.54                 |
| Attention_Decoder | 10         | Avg   | 0.7160 | 0.7162 | 0.7161  | -                   | -                     |
| Attention_Decoder | 60         | 1     | 0.8692 | 0.8933 | 0.8804  | 13.10               | 6.36                  |
| Attention_Decoder | 60         | 2     | 0.6946 | 0.7481 | 0.7183  | 13.02               | 6.40                  |
| Attention_Decoder | 60         | 3     | 0.7833 | 0.8170 | **0.7988** | 13.05               | 6.38                  |
| Attention_Decoder | 60         | Avg   | 0.7824 | 0.8195 | 0.7992  | -                   | -                     |
| Attention_Decoder | 110        | 1     | 0.7580 | 0.8117 | 0.7833  | 22.18               | 3.75                  |
| Attention_Decoder | 110        | 2     | 0.7027 | 0.8040 | **0.7477** | 22.32               | 3.73                  |
| Attention_Decoder | 110        | 3     | 0.6684 | 0.7290 | 0.6964  | 22.32               | 3.73                  |
| Attention_Decoder | 110        | Avg   | 0.7097 | 0.7816 | 0.7425  | -                   | -                     |
| Attention_Decoder | 160        | 1     | 0.3942 | 0.5281 | **0.4519** | 31.28               | 2.66                  |
| Attention_Decoder | 160        | 2     | 0.6277 | 0.6644 | 0.6448  | 31.53               | 2.64                  |
| Attention_Decoder | 160        | 3     | 0.4175 | 0.4726 | 0.4425  | 31.93               | 2.61                  |
| Attention_Decoder | 160        | Avg   | 0.4798 | 0.5550 | 0.5131  | -                   | -                     |
| Attention_Decoder | 210        | 1     | 0.5057 | 0.5825 | 0.5401  | 41.43               | 2.01                  |
| Attention_Decoder | 210        | 2     | 0.3850 | 0.5309 | 0.4466  | 44.92               | 1.85                  |
| Attention_Decoder | 210        | 3     | 0.4654 | 0.4958 | **0.4797** | 48.60               | 1.71                  |
| Attention_Decoder | 210        | Avg   | 0.4520 | 0.5364 | 0.4888  | -                   | -                     |
| Attention_Decoder | 260        | 1     | 0.4187 | 0.4924 | **0.4518** | 61.63               | 1.35                  |
| Attention_Decoder | 260        | 2     | 0.5051 | 0.5444 | 0.5237  | 61.83               | 1.35                  |
| Attention_Decoder | 260        | 3     | 0.2679 | 0.3827 | 0.3169  | 61.53               | 1.35                  |
| Attention_Decoder | 260        | Avg   | 0.3972 | 0.4732 | 0.4308  | -                   | -                     |
| Attention_Decoder | 310        | 1     | 0.3340 | 0.3951 | 0.3612  | 69.42               | 1.20                  |
| Attention_Decoder | 310        | 2     | 0.3843 | 0.4444 | **0.4120** | 69.38               | 1.20                  |
| Attention_Decoder | 310        | 3     | 0.4130 | 0.4548 | 0.4319  | 69.28               | 1.20                  |
| Attention_Decoder | 310        | Avg   | 0.3771 | 0.4314 | 0.4017  | -                   | -                     |
| Attention_Decoder_41 | 10         | 1     | 0.7160 | 0.7173 | 0.7165  | 3.08                | 27.03                 |
| Attention_Decoder_41 | 10         | 2     | 0.7136 | 0.7137 | **0.7137** | 3.12                | 26.69                 |
| Attention_Decoder_41 | 10         | 3     | 0.7130 | 0.7131 | 0.7130  | 3.07                | 27.17                 |
| Attention_Decoder_41 | 10         | Avg   | 0.7142 | 0.7147 | 0.7144  | -                   | -                     |
| Attention_Decoder_41 | 60         | 1     | 0.8559 | 0.8847 | 0.8691  | 12.93               | 6.44                  |
| Attention_Decoder_41 | 60         | 2     | 0.7028 | 0.7478 | 0.7236  | 12.88               | 6.47                  |
| Attention_Decoder_41 | 60         | 3     | 0.7688 | 0.8056 | **0.7860** | 12.83               | 6.49                  |
| Attention_Decoder_41 | 60         | Avg   | 0.7758 | 0.8127 | 0.7929  | -                   | -                     |
| Attention_Decoder_41 | 110        | 1     | 0.6015 | 0.7245 | **0.6526** | 22.35               | 3.73                  |
| Attention_Decoder_41 | 110        | 2     | 0.4956 | 0.5397 | 0.5154  | 23.68               | 3.52                  |
| Attention_Decoder_41 | 110        | 3     | 0.6830 | 0.7466 | 0.7104  | 23.67               | 3.52                  |
| Attention_Decoder_41 | 110        | Avg   | 0.5933 | 0.6703 | 0.6261  | -                   | -                     |
| Attention_Decoder_41 | 160        | 1     | 0.5788 | 0.6107 | **0.5937** | 33.13               | 2.51                  |
| Attention_Decoder_41 | 160        | 2     | 0.5972 | 0.6275 | 0.6115  | 32.53               | 2.56                  |
| Attention_Decoder_41 | 160        | 3     | 0.5437 | 0.5803 | 0.5608  | 31.80               | 2.62                  |
| Attention_Decoder_41 | 160        | Avg   | 0.5732 | 0.6062 | 0.5887  | -                   | -                     |
| Attention_Decoder_41 | 210        | 1     | 0.5575 | 0.6045 | 0.5797  | 42.13               | 1.98                  |
| Attention_Decoder_41 | 210        | 2     | 0.4037 | 0.4479 | **0.4239** | 40.55               | 2.05                  |
| Attention_Decoder_41 | 210        | 3     | 0.3407 | 0.3810 | 0.3592  | 40.45               | 2.06                  |
| Attention_Decoder_41 | 210        | Avg   | 0.4340 | 0.4778 | 0.4543  | -                   | -                     |
| Attention_Decoder_41 | 260        | 1     | 0.4627 | 0.5039 | 0.4821  | 49.72               | 1.68                  |
| Attention_Decoder_41 | 260        | 2     | 0.3199 | 0.3753 | 0.3444  | 49.63               | 1.68                  |
| Attention_Decoder_41 | 260        | 3     | 0.3899 | 0.4309 | **0.4091** | 49.48               | 1.68                  |
| Attention_Decoder_41 | 260        | Avg   | 0.3908 | 0.4367 | 0.4119  | -                   | -                     |
| Attention_Decoder_41 | 310        | 1     | 0.0722 | 0.0980 | 0.0795  | 60.37               | 1.38                  |
| Attention_Decoder_41 | 310        | 2     | 0.2709 | 0.3350 | **0.2997** | 61.57               | 1.35                  |
| Attention_Decoder_41 | 310        | 3     | 0.3528 | 0.4123 | 0.3799  | 61.00               | 1.37                  |
| Attention_Decoder_41 | 310        | Avg   | 0.2320 | 0.2818 | 0.2530  | -                   | -                     |
| Attention_Decoder_42 | 10         | 1     | 0.7144 | 0.7146 | **0.7145** | 3.07                | 27.16                 |
| Attention_Decoder_42 | 10         | 2     | 0.7165 | 0.7165 | 0.7165  | 3.05                | 27.29                 |
| Attention_Decoder_42 | 10         | 3     | 0.7123 | 0.7123 | 0.7123  | 3.05                | 27.29                 |
| Attention_Decoder_42 | 10         | Avg   | 0.7144 | 0.7145 | 0.7144  | -                   | -                     |
| Attention_Decoder_42 | 60         | 1     | 0.0382 | 0.0592 | 0.0402  | 12.87               | 6.47                  |
| Attention_Decoder_42 | 60         | 2     | 0.6288 | 0.6638 | 0.6445  | 12.73               | 6.54                  |
| Attention_Decoder_42 | 60         | 3     | 0.0371 | 0.0614 | **0.0386** | 13.05               | 6.39                  |
| Attention_Decoder_42 | 60         | Avg   | 0.2347 | 0.2615 | 0.2411  | -                   | -                     |
| Attention_Decoder_42 | 110        | 1     | 0.5075 | 0.5554 | 0.5293  | 22.80               | 3.65                  |
| Attention_Decoder_42 | 110        | 2     | 0.4764 | 0.5300 | **0.5007** | 22.63               | 3.68                  |
| Attention_Decoder_42 | 110        | 3     | 0.3056 | 0.3560 | 0.3275  | 22.08               | 3.77                  |
| Attention_Decoder_42 | 110        | Avg   | 0.4298 | 0.4804 | 0.4525  | -                   | -                     |
| Attention_Decoder_42 | 160        | 1     | 0.0451 | 0.1043 | **0.0557** | 31.65               | 2.63                  |
| Attention_Decoder_42 | 160        | 2     | 0.4313 | 0.4850 | 0.4551  | 31.62               | 2.64                  |
| Attention_Decoder_42 | 160        | 3     | 0.0362 | 0.0759 | 0.0432  | 31.03               | 2.68                  |
| Attention_Decoder_42 | 160        | Avg   | 0.1709 | 0.2218 | 0.1846  | -                   | -                     |
| Attention_Decoder_42 | 210        | 1     | 0.0680 | 0.1214 | **0.0819** | 41.67               | 2.00                  |
| Attention_Decoder_42 | 210        | 2     | 0.0346 | 0.0959 | 0.0468  | 40.92               | 2.04                  |
| Attention_Decoder_42 | 210        | 3     | 0.0885 | 0.1225 | 0.0981  | 40.48               | 2.06                  |
| Attention_Decoder_42 | 210        | Avg   | 0.0637 | 0.1133 | 0.0756  | -                   | -                     |
| Attention_Decoder_42 | 260        | 1     | 0.0383 | 0.1104 | **0.0543** | 50.30               | 1.66                  |
| Attention_Decoder_42 | 260        | 2     | 0.1983 | 0.2362 | 0.2149  | 49.37               | 1.69                  |
| Attention_Decoder_42 | 260        | 3     | 0.0352 | 0.0964 | 0.0482  | 50.70               | 1.64                  |
| Attention_Decoder_42 | 260        | Avg   | 0.0906 | 0.1476 | 0.1058  | -                   | -                     |
| Attention_Decoder_42 | 310        | 1     | 0.4174 | 0.4460 | 0.4307  | 66.27               | 1.26                  |
| Attention_Decoder_42 | 310        | 2     | 0.0789 | 0.1443 | **0.0981** | 71.65               | 1.16                  |
| Attention_Decoder_42 | 310        | 3     | 0.0380 | 0.0900 | 0.0514  | 82.58               | 1.01                  |
| Attention_Decoder_42 | 310        | Avg   | 0.1781 | 0.2268 | 0.1934  | -                   | -                     |
| Attention_Decoder_43   | 10         | 1     | 0.7195 | 0.7196 | 0.7195  | 1.23                | 26.90                 |
| Attention_Decoder_43   | 10         | 2     | 0.7191 | 0.7192 | **0.7192** | 1.18                | 27.88                 |
| Attention_Decoder_43   | 10         | 3     | 0.7107 | 0.7109 | 0.7108  | 1.32                | 25.27                 |
| Attention_Decoder_43   | 10         | Avg   | 0.7164 | 0.7165 | 0.7165  | -                   | -                     |
| Attention_Decoder_43   | 60         | 1     | 0.7482 | 0.8212 | **0.7797** | 4.97                | 6.69                  |
| Attention_Decoder_43   | 60         | 2     | 0.5692 | 0.7314 | 0.6316  | 5.02                | 6.63                  |
| Attention_Decoder_43   | 60         | 3     | 0.8676 | 0.8786 | 0.8729  | 5.10                | 6.52                  |
| Attention_Decoder_43   | 60         | Avg   | 0.7283 | 0.8104 | 0.7614  | -                   | -                     |
| Attention_Decoder_43   | 110        | 1     | 0.5820 | 0.6409 | **0.6091** | 8.57                | 3.89                  |
| Attention_Decoder_43   | 110        | 2     | 0.7741 | 0.8010 | 0.7869  | 8.88                | 3.75                  |
| Attention_Decoder_43   | 110        | 3     | 0.5334 | 0.6112 | 0.5687  | 8.58                | 3.88                  |
| Attention_Decoder_43   | 110        | Avg   | 0.6298 | 0.6844 | 0.6549  | -                   | -                     |
| Attention_Decoder_43   | 160        | 1     | 0.6275 | 0.6723 | 0.6482  | 12.57               | 2.65                  |
| Attention_Decoder_43   | 160        | 2     | 0.6176 | 0.6686 | **0.6416** | 12.52               | 2.66                  |
| Attention_Decoder_43   | 160        | 3     | 0.4741 | 0.5405 | 0.5045  | 12.43               | 2.68                  |
| Attention_Decoder_43   | 160        | Avg   | 0.5731 | 0.6271 | 0.5981  | -                   | -                     |
| Attention_Decoder_43   | 210        | 1     | 0.4458 | 0.5172 | 0.4768  | 16.13               | 2.06                  |
| Attention_Decoder_43   | 210        | 2     | 0.3076 | 0.4286 | 0.3569  | 16.05               | 2.07                  |
| Attention_Decoder_43   | 210        | 3     | 0.3365 | 0.3988 | **0.3645** | 16.62               | 2.01                  |
| Attention_Decoder_43   | 210        | Avg   | 0.3633 | 0.4482 | 0.3994  | -                   | -                     |
| Attention_Decoder_43   | 260        | 1     | 0.3736 | 0.4104 | 0.3907  | 21.63               | 1.54                  |
| Attention_Decoder_43   | 260        | 2     | 0.3653 | 0.3965 | **0.3799** | 21.87               | 1.52                  |
| Attention_Decoder_43   | 260        | 3     | 0.3446 | 0.3861 | 0.3634  | 21.20               | 1.57                  |
| Attention_Decoder_43   | 260        | Avg   | 0.3611 | 0.3977 | 0.3780  | -                   | -                     |
| Attention_Decoder_43   | 310        | 1     | 0.1850 | 0.2057 | 0.1943  | 34.18               | 1.46                  |
| Attention_Decoder_43   | 310        | 2     | 0.2910 | 0.3441 | **0.3153** | 33.72               | 1.48                  |
| Attention_Decoder_43 | 310 | 3 |  |  |  |  |  |
| Attention_Decoder_44 | 10         | 1     | 0.7103 | 0.7108 | 0.7105     | 1.27                | 26.29                 |
| Attention_Decoder_44 | 10         | 2     | 0.7199 | 0.7200 | **0.7199** | 1.23                | 26.81                 |
| Attention_Decoder_44 | 10         | 3     | 0.7213 | 0.7213 | 0.7213     | 1.12                | 29.46                 |
| Attention_Decoder_44 | 10         | Avg   | 0.7172 | 0.7173 | 0.7172     | -                   | -                     |
| Attention_Decoder_44 | 60         | 1     | 0.8837 | 0.8906 | **0.8870** | 4.83                | 6.90                  |
| Attention_Decoder_44 | 60         | 2     | 0.6284 | 0.6534 | 0.6391     | 4.72                | 7.07                  |
| Attention_Decoder_44 | 60         | 3     | 0.6944 | 0.7451 | 0.7172     | 4.65                | 7.16                  |
| Attention_Decoder_44 | 60         | Avg   | 0.7355 | 0.7630 | 0.7478     | -                   | -                     |
| Attention_Decoder_44 | 110        | 1     | 0.6725 | 0.7162 | 0.6927     | 8.48                | 3.93                  |
| Attention_Decoder_44 | 110        | 2     | 0.6998 | 0.7692 | **0.7319** | 8.40                | 3.96                  |
| Attention_Decoder_44 | 110        | 3     | 0.7707 | 0.7943 | 0.7819     | 8.55                | 3.90                  |
| Attention_Decoder_44 | 110        | Avg   | 0.7143 | 0.7599 | 0.7355     | -                   | -                     |
| Attention_Decoder_44 | 160        | 1     | 0.0687 | 0.1491 | 0.0868     | 12.33               | 2.70                  |
| Attention_Decoder_44 | 160        | 2     | 0.5028 | 0.5374 | **0.5190** | 12.38               | 2.69                  |
| Attention_Decoder_44 | 160        | 3     | 0.4045 | 0.4878 | 0.4419     | 12.40               | 2.69                  |
| Attention_Decoder_44 | 160        | Avg   | 0.3253 | 0.3914 | 0.3492     | -                   | -                     |
| Attention_Decoder_44 | 210        | 1     | 0.1897 | 0.2526 | 0.2155     | 16.62               | 2.01                  |
| Attention_Decoder_44 | 210        | 2     | 0.4192 | 0.4864 | **0.4499** | 16.62               | 2.00                  |
| Attention_Decoder_44 | 210        | 3     | 0.0738 | 0.1269 | 0.0878     | 16.37               | 2.04                  |
| Attention_Decoder_44 | 210        | Avg   | 0.2276 | 0.2887 | 0.2511     | -                   | -                     |
| Attention_Decoder_44 | 260        | 1     | 0.2872 | 0.3812 | **0.3285** | 26.15               | 1.27                  |
| Attention_Decoder_44 | 260        | 2     | 0.0761 | 0.1281 | 0.0904     | 25.25               | 1.32                  |
| Attention_Decoder_44 | 260        | 3     | 0.2628 | 0.3766 | 0.3110     | 25.07               | 1.33                  |
| Attention_Decoder_44 | 260        | Avg   | 0.2087 | 0.2953 | 0.2433     | -                   | -                     |
| Attention_Decoder_44 | 310        | 1     | 0.2713 | 0.3510 | **0.3066** | 26.62               | 1.25                  |
| Attention_Decoder_44 | 310        | 2     | 0.3563 | 0.4207 | 0.3858     | 27.10               | 1.23                  |
| Attention_Decoder_44 | 310        | 3     | 0.0482 | 0.0917 | 0.0587     | 26.60               | 1.25                  |
| Attention_Decoder_44 | 310        | Avg   | 0.2253 | 0.2878 | 0.2504     | -                   | -                     |

[0.7199, 0.8870, 0.7319, 0.5190 , 0.4499, 0.3285, 0.3066]

 