import numpy as np
import torch.nn as nn
from datasets import *

# transformer 的公式计算步骤参考如下 https://wmathor.com/index.php/archives/1438/
# https://zhuanlan.zhihu.com/p/338817680

d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度 512->2048->512 线性曾是用来作特征提取的
d_k = d_v = 64  # Q、K、V 向量的维度，其中 Q 与 K 的维度必须相等，V 的维度没有限制，我们都设为 64
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8

# similar code from following post
# https://blog.csdn.net/BXD1314/article/details/126187598

# Mask 输入时掩盖数据 >>> masked multi-head attention
def get_attn_pad_mask(seq_q, seq_k):                                
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """
        这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
        encoder和decoder都可能调用这个函数，所以seq_len视情况而定
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()    # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()

    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]] -> (2,1,5) -> [[[F, F, F, F, T]],[[F, F, F, F, T]]]
    # [batch_size, 1, len_k], True is masked
    # seq_k.data.eq(0) 返回一个大小和 seq_k 一样的 tensor，值只有 True 和 False
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)                   # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k] seq中有占位符 对于占位符 用1标记  然后维度进行扩展
    # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)
    return pad_attn_mask.expand(batch_size, len_q, len_k)           # 扩展成多维度--dim=1 进行扩展

# Decoder 输入的mask 屏蔽子序列的mask 只decoder用到用来屏蔽未来时刻单词的信息
def get_attn_subsequence_mask(seq):                                 # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)            # 先生成全1矩阵，然后生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()    # [batch_size, tgt_len, tgt_len]
    return subsequence_mask

# ==================================================================================
# ---------------------------------------------------------
# position-encoding + world-embedding
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    # d_model: embedding 的维度
    # max_len: 最大序列长度
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])           # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])           # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).cuda()        # enc_inputs: [seq_len, d_model]
    
    # positional output = embedding + position
    def forward(self, enc_inputs):                                  # enc_inputs: [batch_size, seq_len, d_model/embedding_size]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.cuda())

# ---------------------------------------------------------------------
# https://zhuanlan.zhihu.com/p/82312421
# attention 有好多种 additive attention、local-based、general、dot-product、scaled dot-product
# Transformer 中用到的是scaled dot-product attention 
# ---------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):                              
        """d_k 输入的维度 64=512/8
            Q: [batch_size, n_heads, len_q, d_k]    
            K: [batch_size, n_heads, len_k, d_k]
            V: [batch_size, n_heads, len_v(=len_k), d_v]
            在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同 decoder 中 Q用的是encoder的，K用的是decoder自己的，因此有可能不同
            attn_mask: [batch_size, n_heads, seq_len, seq_len] attention 输出的维度
        """

        # https://zhuanlan.zhihu.com/p/48508221 
        # 注意力矩阵 a 的计算方式 dimension -1, -2 are swapped
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)    
        # scores : [batch_size, n_heads, len_q, len_k]
        
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        scores.masked_fill_(attn_mask, -1e9)                            # 如果时停用词P就等于 0
        
        attn = nn.Softmax(dim=-1)(scores)   # 对最后一个维度(v)做softmax
        # softmax(Q*K/sqrt(d_k)) * V -- z 矩阵        
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)     # [batch_size, n_heads, len_q, d_v]
        return context, attn

# Encoder 中的 Multi-head attention: residual connection + layer normalization
# Multi-head attention = h * self-attention
class MultiHeadAttention(nn.Module):
    """
        # enc_input              dec_input                    dec_output
        ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E']
        完整代码中一定会有三处地方调用 MultiHeadAttention()，
        Encoder Layer 调用一次
            传入的 input_Q、input_K、input_V 全部都是 enc_inputs
        Decoder Layer 中两次调用
            第一次传入的全是 dec_inputs
            第二次传入的分别是 dec_outputs，enc_outputs，enc_outputs
        输入: seq_len * d_model
        输出: seq_len * d_model
        Add & Norm: Add & Norm 两部分组成 LayerNorm(X+Attention) LayerNorm(X+FeedForward())
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # d_model: embedding length
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)    # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        
        # 全链接曾 保证多头attention的输出的维度仍然是 seq_len * d_model
        # 要做一次线性变换 在不同的实现中都存在这一步？？https://zhuanlan.zhihu.com/p/47812375
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        

    def forward(self, input_Q, input_K, input_V, attn_mask):    
        """
            input_Q: [batch_size, len_q, d_model]   len_q sequence 的长度(input sequence)
            input_K: [batch_size, len_k, d_model]   len_k sequence 的长度(output sequence)
            input_V: [batch_size, len_v(=len_k), d_model]
            attn_mask: [batch_size, seq_len, seq_len]
        """
        
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim  D_new: Q, K coln 的 dimension
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # pytorch.tensor.view() 将数据进行dimension的转换 转换成单个head的dimension 然后再输入到attention中
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)    # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)    # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)    # V: [batch_size, n_heads, len_v(=len_k), d_v]
        
        # 因为是多头，所以mask矩阵要相对应的扩充成4维的
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)                 # attn_mask : [batch_size, n_heads, seq_len, seq_len]        
        # context b 矩阵 -- 用来进行计算 没有参数
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)             
        # context: [batch_size, n_heads, len_q, d_v]    # attn: [batch_size, n_heads, len_q, len_k]
        # 下面将不同头的输出向量拼接在一起   
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]                                            
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)    # context: [batch_size, len_q, n_heads * d_v]
        
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        output = self.fc(context)                                                   # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn

# Encoder 中的feed-forward net 需要有残差 + normalization.
# Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
class PoswiseFeedForwardNet(nn.Module):
    # https://zhuanlan.zhihu.com/p/47812375
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):      # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]

# encoder = multi-head attention + position-wise feed-forward network
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                   # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                      # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):              
        """E
            enc_inputs: [batch_size, src_len, d_model]
            enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V

        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V            
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  
        # enc_outputs: [batch_size, src_len, d_model]   # attn: [batch_size, n_heads, src_len, src_len]
        
        enc_outputs = self.pos_ffn(enc_outputs)     # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)                     # 把字转换字向量
        self.pos_emb = PositionalEncoding(d_model)                               # 加入位置信息
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):                                               
        # enc_inputs: [batch_size, src_len]
        #输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V                          # enc_self_attn_mask: [batch_size, src_len, src_len]
        # embedding
        enc_outputs = self.src_emb(enc_inputs)                                   # enc_outputs: [batch_size, src_len, d_model]
        # embedding + position
        enc_outputs = self.pos_emb(enc_outputs)                                  # enc_outputs: [batch_size, src_len, d_model]
        # 第三步，Mask掉句子中的占位符号
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)           # enc_self_attn_mask: [batch_size, src_len, src_len]
        
        enc_self_attns = []     # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            # 这个只是为了可视化
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# ==================================================================================
# ==================================================================================
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask,
                dec_enc_attn_mask):                                             
        """
            dec_inputs: [batch_size, tgt_len, d_model]
            enc_outputs: [batch_size, src_len, d_model]     
            dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
            dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """                                             
        # 这里的Q,K,V全是Decoder自己的输入 第一层attention                                   
        # dec_outputs: [batch_size, tgt_len, d_model]   # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs,
                                                        dec_self_attn_mask)     
        
        # 第二层attention Attention层的Q(来自decoder) 和 K,V(来自encoder)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs,
                                                      dec_enc_attn_mask)        
        # dec_outputs: [batch_size, tgt_len, d_model]   
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        
        dec_outputs = self.pos_ffn(dec_outputs)     # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):                         
        """
            dec_inputs: [batch_size, tgt_len]
            enc_inputs: [batch_size, src_len]
            enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        # target sequence embedding
        dec_outputs = self.tgt_emb(dec_inputs)              # [batch_size, tgt_len, d_model]
        # target position embedding
        dec_outputs = self.pos_emb(dec_outputs).cuda()      # [batch_size, tgt_len, d_model]

        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()   # [batch_size, tgt_len, tgt_len]
        
        # Masked Self_Attention：当前时刻是看不到未来的信息的 -- upper-triangular mask
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda()   # [batch_size, tgt_len, tgt_len]

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵
        # (因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #  dec_inputs只是提供expand的size的
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)   # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:                                                   
            # dec_outputs: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder().cuda()
        self.Decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs):                          # enc_inputs: [batch_size, src_len]
                                                                        # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)          # enc_outputs: [batch_size, src_len, d_model],
                                                                        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)                        # dec_outpus    : [batch_size, tgt_len, d_model],
                                                                        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
                                                                        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)                       # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
