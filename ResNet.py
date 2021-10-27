import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.resnet = models.resnet34(pretrained=False)  # 直接调用torchvision的resnet50

    #main里直接用model = models.resnet50(pretrained=False)也可,这里是为了添加x_len与其他模型保持一致
    def forward(self, x, x_len):
        y = self.resnet(x)
        return y

