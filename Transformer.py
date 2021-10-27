import torch.nn as nn
import copy
import math
import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential
from torch.nn import Embedding, Conv1d, MaxPool1d, Linear, ReLU, Tanh, Dropout, LayerNorm
from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch

'''
Transformer
输入：（batch, len, dim)
输出：（batch, len, classnum)     对每一帧进行预测

这里相当于加几层self-attention层,与LSTM一样,填充零的部分不计算损失,只取实际长度计算损失

如果不对每帧进行预测，而是整体进行预测，可参考BERT里的CLS(ViT里的class token)和美团CPVT的GAP，这里不展开了
'''
class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        if params.task == "classification":
            classnum = 2
        else:
            classnum = 1

        params.d_in = 10
        params.d_model = 64
        params.n_layers = 6
        params.n_heads = 8
        params.d_fc = 128
        params.dr = 0.2
        params.scale_embedding = False
        params.d_fc_out = 256
        params.out_dr = 0.2
        params.no_pe = False
        self.params = params

        self.proj = nn.Linear(params.d_in, params.d_model, bias=False)
        self.encoder = TransformerEncoder(params.n_layers, params.d_model, params.n_heads, params.d_fc,
                                                      dropout=params.dr, scale_embedding=params.scale_embedding)
        self.fc = nn.Sequential(nn.Linear(params.d_model, params.d_fc_out), nn.ReLU(True),
                                nn.Dropout(params.dr), nn.Linear(params.d_fc_out, classnum))

    def forward(self, x, x_len):

        x = self.proj(x) if self.proj is not None else x
        x = x.transpose(0,1)
        x_padding_mask = get_padding_mask(x, x_len)
        x = self.encoder(x, x_padding_mask, no_pe=self.params.no_pe)
        x = x.transpose(0, 1)
        y = self.fc(x)
        return y

def get_padding_mask( x, x_lens):
    """
    :param x: (seq_len, batch_size, feature_dim)
    :param x_lens: sequence lengths within a batch with size (batch_size,)
    :return: padding_mask with size (batch_size, seq_len)
    """

    mask = torch.ones(x.size(1), x.size(0), device=x.device)
    for seq, seq_len in enumerate(x_lens):
        mask[seq, :seq_len] = 0
    mask = mask.bool()
    return mask

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.1, max_len=800):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1) # Note: pe with size (seq_len, 1, feature_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: with size (seq_len, batch_size, feature_dim)
        :return:
        """
        x = x + self.pe[:x.size(0), :] # Note: ":" means for all rest axis
        return self.dropout(x)

class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self/cross-attn and feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        n_heads: the number of heads in the multiheadattention models (required).
        d_fc: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, n_heads, d_fc=64, dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        # Note: start using dropout for multi-head attention in 04/23
        self.multihead_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, d_fc)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_fc, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)


    def forward(self, tgt, src, src_mask=None, src_key_padding_mask=None):
        """
            tgt: (tgt_seq_len, batch_size, feature_dim)
            src: (src_seq_len, batch_size, feature_dim)
            src_mask: (tgt_seq_len, src_seq_len)
            src_key_padding_mask: (batch_size, src_seq_len)
            output: (tgt_seq_len, batch_size, feature_dim)
        """
        tgt2 = self.multihead_attn(tgt, src, src, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

class TransformerEncoder(Module):
    def __init__(self, n_layers, d_model, n_heads, d_fc=64, dropout=0.1, activation="gelu", scale_embedding=False):
        super(TransformerEncoder, self).__init__()
        self.n_layers = n_layers
        self.scale = math.sqrt(d_model) if scale_embedding == True else 1.0
        self.pe = PositionalEncoding(d_model, dropout)
        self.layers = _get_clones(TransformerEncoderLayer(d_model, n_heads, d_fc, dropout, activation), n_layers)
        self.norm = LayerNorm(d_model)

    def forward(self, x, x_padding_mask, no_pe=False):
        x = x * self.scale
        if no_pe == False:
            x = self.pe(x)
        for layer in self.layers:
            x = layer(x, x, src_key_padding_mask=x_padding_mask)
        x = self.norm(x)
        return x

if __name__ == '__main__':
    from torchinfo import summary
    # 使用summary可查看参数量
    import argparse
    args = argparse.Namespace()
    args.task="classification"
    model = Transformer(args)
    x1 = torch.randn(3, 8, 10)
    x2 = torch.zeros(3, 2, 10)
    x = torch.cat((x1,x2),1)
    x_len = torch.IntTensor([8]*3)
    # y= model(x,x_len)
    # print(y.shape)
    summary(model=model,input_data=(x,x_len))

