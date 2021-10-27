# -*- coding: utf-8 -*-
import torch
import argparse
from dataload import Dataloader
import os
import torch.nn as nn

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', default='./data/audio/wav2vec/', help='path to dataset')
    parser.add_argument('--model_path', default='./model/wav2vec_4.pth', type=str, help='path to dataset')
    parser.add_argument('--feature',required=True,choices=['wav', 'wav2vec', 'image', "AU", "vggface"],
                        help='specify the feature')
    parser.add_argument('--feature_dim',type=int,help='the dimension of the feature')
    parser.add_argument('--task',default='classification',choices=['classification', 'regression'],
                        help='regression or classification')
    parser.add_argument('--batch_size', default=3, type=int, help='size of a mini-batch')
    parser.add_argument('--model',required=True, choices=['lstm', 'transformer', 'resnet'],
                        help='choose the model')

    args = parser.parse_args()

    testloader = Dataloader(root=args.dataset_path, feature=args.feature, batch_size=args.batch_size, partition="train")

    if os.path.exists(args.model_path):
        print("load model: ",args.model_path)
        model = torch.load(args.model_path)
    else:
        print("model is not exist")

    test(model, testloader, args.task)

def test(model, val_loader, task):
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
    print(f"test loss : {val_loss} ")
    return val_loss

if __name__ == "__main__":
    main()

