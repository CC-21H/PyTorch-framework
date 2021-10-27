# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.optim as optim
from dataload import *
from math import sqrt

#训练模型
def train_model(model, trainloader,devloader, epochs, lr, model_path, task):

    optimizer = optim.Adam(model.parameters(), lr=lr)    #Adam优化器
    best_val_loss = float('inf')
    print('Start training')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, trainloader, epoch, optimizer, task)
        val_loss = evaluate(model, devloader, task)
        print('-' * 50)
        print(f'Epoch:{epoch:>3} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} |   [Val] | Loss: {val_loss:>.4f} ')     #输出训练和验证loss
        print('-' * 50)

        if val_loss < best_val_loss:                   #保存模型
            best_val_loss = val_loss
            best_model = os.path.join("./model/",f'{model_path}_{task}_{epoch}.pth')
            torch.save(model, best_model)

    print(f'Best Val Loss: {best_val_loss:>.4f}')

#每轮训练模型
def train(model, trainloader, epoch, optimizer, task):

    report_loss, report_size = 0, 0
    total_loss, total_size = 0, 0

    if task == "regression":
        criterion = nn.MSELoss()           #回归任务loss是MSE
    else:
        criterion = nn.CrossEntropyLoss()  #分类任务loss是交叉熵

    model.train()                          #切换模型为训练状态
    for batch, batch_data in enumerate(trainloader, 1):
        features, fea_len, labels= batch_data

        if task == "regression":       #计算损失时需转换类型
            labels = labels.float()
        else:
            labels = labels.long()

        batch_size = features.size(0)
        model, features, labels =model.cuda(), features.cuda(),labels.cuda()
        optimizer.zero_grad()
        preds = model(features.float(), fea_len)
        loss = criterion(preds, labels)         #计算loss
        loss.backward()                         #梯度回传
        optimizer.step()

        report_loss += loss.item() * batch_size
        report_size += batch_size

        avg_loss = report_loss / report_size
        print(f"Epoch:{epoch:>3} | Batch: {batch:>3} | Training loss: {avg_loss:>.4f}")

        total_loss += report_loss
        total_size += report_size
        report_loss, report_size = 0, 0

    train_loss = total_loss / total_size
    #train_loss = sqrt(total_loss / total_size)    #回归任务若要打印 RMSE还需加上sqrt
    return train_loss

def evaluate(model, val_loader, task):
    model.eval()     #切换模型为预测模型
    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_size = 0
    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader, 1):
            features, fea_len, labels = batch_data
            batch_size = features.size(0)
            model, features, labels = model.cuda(), features.cuda(), labels.cuda()
            preds = model(features.float(),fea_len)
            loss = criterion(preds, labels)              #验证时只计算，不反向传播
            total_loss += loss.item() * batch_size
            total_size += batch_size
    val_loss = total_loss / total_size
    return val_loss