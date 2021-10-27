# -*- coding: utf-8 -*-
import os
import librosa
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import scipy.io as scio

'''
加载不同音频图像数据 (.wav .mat .jpg .csv .npy)

Dataset: 数据预处理，基本是把原始数据转化为tensor
DataLoader：数据加载，读取数据的方式，常见是设置batch_size和是否打乱shuffle，进阶有过采样，收集函数等
'''

#加载.wav (batch,80000)
class Dataset_wav(Dataset):
    def __init__(self, root):  # __init__是初始化该类的一些基础参数，     ./data/audio/wav/train
        self.max_len = 80000   #指定最大长度，如果音频是都是等长的则不需要
        self.data_list = []
        self.label_list = []
        label_data = pd.read_csv("./data/label.csv")  # 读取label文件
        label_data.set_index("filename", inplace=True)  # 设置filename为索引
        for filename in os.listdir(root):                # 遍历root下所有文件夹
            self.data_list.append(os.path.join(root,filename))  # 将文件名装入data_list   ./data/audio/wav/train/1.wav
            basename = os.path.splitext(filename)[0]            # 获取基础文件名  1
            self.label_list.append(label_data.loc[basename].label)  # 根据文件名 1 在label.csv查找对应label并装入label_list

    def __len__(self):                             # 返回整个数据集的大小
        return len(self.data_list)

    #也可以在__init__函数中处理好，直接在这里调用
    def __getitem__(self, index):                  # 根据索引index返回dataset[index]
        x, sr = librosa.load(self.data_list[index])  # 用librosa读取音频，读取的音频x是numpy格式
        waveData = torch.from_numpy(x)              # numpy -> trensor (其实后面Dataloader会自动转，删掉这步也行）

        fea_len = waveData.size(0)                   # 记录特征的真实长度，网络计算中需要
        if waveData.size(0) < self.max_len:           # 如果数据不够最大长度，填充零到最大长度,
                                                      # 一般在collate_fn函数中处理，为了简单便于理解我们直接在这里处理
            tem = torch.zeros([self.max_len - waveData.size(0)])
            waveData = torch.cat((waveData, tem), 0)

        label = int(self.label_list[index])
        #waveData = waveData.unsqueeze(0)  # (Batch,1,80000)
        sample = (waveData,fea_len,label)  # 根据特征，特征长度和标签创建元组
        return sample

#加载.mat (batch,20,768)
class Dataset_wav2vec(Dataset):
    def __init__(self, root):
        #self.max_len = 80000
        self.data_list = []
        self.label_list = []
        label_data = pd.read_csv("./data/label.csv")
        label_data.set_index("filename", inplace=True)
        for filename in os.listdir(root):
            self.data_list.append(os.path.join(root,filename))
            basename = os.path.splitext(filename)[0]
            self.label_list.append(label_data.loc[basename].label)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        mat = scio.loadmat(self.data_list[index]) #读取.mat
        wav2vec = mat["feature"]                  #.mat是字典的形式，将feature提取出来（保存的时候写的feature）
        wav2vec = torch.from_numpy(wav2vec)
        fea_len = 20                              #这里是为了与其他特征格式上保持一致。特征是定长20，可以不要
        label = int(self.label_list[index])
        sample = (wav2vec, fea_len, label)
        return sample

#读取图片，并 resize 成224*224
def readImage(path, size=224):
    mode = Image.open(path)
    transform1 = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor()
    ])
    mode = transform1(mode)
    return mode
#加载.jpg (batch,3,224,224)
class Dataset_image(Dataset):
    def __init__(self, root):
        #self.max_len = 80000
        self.data_list = []
        self.label_list = []
        label_data = pd.read_csv("./data/label.csv")
        label_data.set_index("filename", inplace=True)
        for filename in os.listdir(root):
            self.data_list.append(os.path.join(root,filename))
            basename = os.path.splitext(filename)[0]
            self.label_list.append(label_data.loc[basename].label)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image = readImage(self.data_list[index])   #读取.jpg
        #image = torch.from_numpy(image)           #readImage里已经转成tensor了
        fea_len = 224                              #为了与其他特征格式上保持一致，可以不要
        label = int(self.label_list[index])
        sample = (image, fea_len, label)
        return sample

#加载.csv (batch,30,34)
class Dataset_AU(Dataset):
    def __init__(self, root):
        self.max_len = 30
        self.data_list = []
        self.label_list = []
        label_data = pd.read_csv("./data/label.csv")
        label_data.set_index("filename", inplace=True)
        for filename in os.listdir(root):
            self.data_list.append(os.path.join(root,filename))
            basename = os.path.splitext(filename)[0]
            self.label_list.append(label_data.loc[basename].label)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        df = pd.read_csv(self.data_list[index], header=0)
        df = df.iloc[:, 5:-1]
        feature = np.array(df)
        feature = torch.from_numpy(feature)

        fea_len = feature.size(0)
        if feature.size(0) < self.max_len:
            tem = torch.zeros([self.max_len - feature.size(0), 34])
            feature = torch.cat((feature.float(), tem), 0)

        label = int(self.label_list[index])
        sample = (feature, fea_len, label)
        return sample

#加载.npy (batch,20,128)
class Dataset_vggface(Dataset):
    def __init__(self, root):
        #self.max_len = 20
        self.data_list = []
        self.label_list = []
        label_data = pd.read_csv("./data/label.csv")
        label_data.set_index("filename", inplace=True)
        for filename in os.listdir(root):
            self.data_list.append(os.path.join(root,filename))
            basename = os.path.splitext(filename)[0]
            self.label_list.append(label_data.loc[basename].label)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        feature = np.load(self.data_list[index])
        feature = torch.from_numpy(feature)
        fea_len = 20                            #为了与其他特征格式上保持一致，可以不要
        label = int(self.label_list[index])
        sample = (feature, fea_len, label)
        return sample

def Dataloader(root,feature,batch_size,partition):
    if feature=="wav":
        dataloader = Dataset_wav(root+"/"+partition)
    if feature=="wav2vec":
        dataloader = Dataset_wav2vec(root+"/"+partition)
    if feature=="image":
        dataloader = Dataset_image(root+"/"+partition)
    if feature == "AU":
        dataloader = Dataset_AU(root+"/"+partition)
    if feature=="vggface":
        dataloader = Dataset_vggface(root+"/"+partition)

    if partition=="train":
        shuffle = True
    else:
        shuffle  =False

    dataloader = DataLoader(dataloader, batch_size=batch_size, shuffle=shuffle, num_workers=0)  # 使用DataLoader加载数据
    return dataloader


if __name__ == '__main__':

    trainloader = Dataloader("./data/audio/wav/","wav",3,"train")
    # trainloader = Dataloader("./data/audio/wav2vec/", "wav2vec", 3, "train")
    # trainloader = Dataloader("./data/vision/image/", "image", 3, "train")
    # trainloader = Dataloader("./data/vision/AU/", "AU", 3, "train")
    # trainloader = Dataloader("./data/vision/vggface/", "vggface", 3, "train")
    for i,data in enumerate(trainloader,1):
        fea, fea_len, label = data
        print(fea.shape,fea_len)


    # 保存.mat
    # import scipy.io as scio
    # data = np.random.rand(20,768)e
    # scio.savemat("1.mat", {'feature': data})

    # 保存.npy
    # data = np.random.rand(20, 128)
    # np.save("1.npy", data)






