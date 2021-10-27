import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
from torch.autograd import Variable

'''
LSTM
输入：（batch, len, dim)
输出：（batch, classnum)     回归任务classnum=1，分类任务classnum=类别数量，这里我们是二分类，所以classnum=2
'''
class LSTM(nn.Module):
    def __init__(self, params):
        super(LSTM, self).__init__()
        if params.task == "classification":
            classnum = 2
        else:
            classnum = 1

        #这里固定了网络参数，若要写成参数形式则取消在main.py里面的把网络结构参数注释，再把这里全部注释掉
        params.d_rnn = 128        #LSTM维度
        params.num_layers = 4     #LSTM层数
        params.rnn_bi = True      #是否双向
        params.dropout = 0.2
        self.params = params

        self.linear = nn.Linear(in_features=params.feature_dim, out_features=params.d_rnn, bias=False)    #线性映射 这里bias是False，为了填充的零映射后还是零
        self.rnn = nn.LSTM(input_size=params.d_rnn, hidden_size=params.d_rnn, bidirectional=params.rnn_bi,
                           num_layers=params.num_layers, dropout=params.dropout)
        self.fc = nn.Linear(params.d_rnn*2, classnum)     #双向LSTM，输入维度要乘2

    #线性映射，LSTM，再两层全连接层分类
    def forward(self, x, x_len):

        x = self.linear(x)

        # LSTM,用pack_padded_sequence和pad_packed_sequence打包和解压序列
        x_packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        x_out = self.rnn(x_packed)[0]
        x_padded = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        # 只取LSTM最后一帧的输出
        I = torch.LongTensor(x_len.long()).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.params.d_rnn*2) - 1).cuda()
        x = torch.gather(x_padded, 1, I).squeeze(1)

        #分类
        y = self.fc(x)
        return y

'''
LSTM2
输入：（batch, len, dim)
输出：（batch, len, classnum)     对每一帧都做预测，针对每一帧都有标签的时序任务

注意填充零的部分不计算损失,只取实际长度计算损失
'''
class LSTM2(nn.Module):
    def __init__(self, params):
        super(LSTM2, self).__init__()
        if params.task == "classification":
            classnum = 2
        else:
            classnum = 1

        params.d_rnn = 128       #LSTM维度
        params.num_layers = 4    #LSTM层数
        params.rnn_bi = True     #双向
        params.hidden = 128      #回归隐藏层
        params.dropout = 0.2
        self.params = params

        self.linear = nn.Linear(in_features=params.feature_dim, out_features=params.d_rnn, bias=False)    #线性映射 这里bias是False，为了填充的零映射后还是零
        self.rnn = nn.LSTM(input_size=params.d_rnn, hidden_size=params.d_rnn, bidirectional=params.rnn_bi,
                           num_layers=params.num_layers, dropout=params.dropout)
        self.fc = nn.Sequential(nn.Linear(params.d_rnn*2, params.hidden),
                                  nn.ReLU(True),
                                  nn.Dropout(params.dropout),
                                  nn.Linear(params.hidden, classnum))

    def forward(self, x, x_len):
        x = self.linear(x)

        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        x_out = self.rnn(x_packed)[0]
        x_padded = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        # 这里不取最后一帧，每一帧都要取。
        # 输出长度是最大长度，填充零的部分最后也有输出，在train.py里计算loss反向传播时，只取实际长度计算。
        y = self.fc(x_padded)
        return y


'''
LSTM2 和 transformer 针对每一帧都有标签的任务，这里没有提供这种标签
可使用summary查看网络结构和参数量
'''
if __name__ == '__main__':

    from torchinfo import summary
    import argparse
    args = argparse.Namespace()
    args.task="classification"
    args.feature_dim = 128

    model = LSTM2(args).cuda()
    x = torch.randn(50, 20, 128).cuda()
    x_len = torch.IntTensor([20]*50)
    # y= model(x,x_len)
    # print(y.shape)
    # 使用summary可查看网络结构和参数量
    summary(model=model,input_data=(x,x_len))

