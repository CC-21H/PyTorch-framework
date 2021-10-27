# -*- coding: utf-8 -*-
import argparse
import torch.nn as nn
from dataload import Dataloader
from Transformer import Transformer
from LSTM import LSTM
from ResNet import ResNet
from train import train_model

def main():

    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument('--dataset_path', default='./data/audio/wav2vec/', help='path to dataset')
    parser.add_argument('--parallel', action='store_true',help='whether use DataParallel')
    parser.add_argument('--model_path', default='wav2vec', type=str, help='path to dataset')
    parser.add_argument('--feature',required=True,choices=['wav', 'wav2vec', 'image', "AU", "vggface"],
                        help='specify the feature')
    parser.add_argument('--feature_dim',type=int,help='the dimension of the feature')
    parser.add_argument('--task',default='classification',choices=['classification', 'regression'],
                        help='regression or classification')

    # Training and optimization
    parser.add_argument('--epochs', default=5, help='number of training epochs')
    parser.add_argument('--batch_size', default=3, type=int, help='size of a mini-batch')
    parser.add_argument('--lr', default=0.002, help='learning rate')
    parser.add_argument('--model',required=True, choices=['lstm', 'transformer', 'resnet'],
                        help='choose the model')

    # network parameter
    # parser.add_argument('d_rnn', default=128)
    # parser.add_argument('num_layers', default=4)
    # parser.add_argument('rnn_bi', default=True)
    # parser.add_argument('dropout', default=0.2)

    args = parser.parse_args()

    trainloader = Dataloader(root=args.dataset_path,feature=args.feature,batch_size=args.batch_size,partition="train")
    devloader = Dataloader(root=args.dataset_path,feature=args.feature,batch_size=args.batch_size,partition="dev")

    if args.model=="lstm":
        model = LSTM(args)
    if args.model == "transformer":
        model = Transformer(args)
    if args.model == "resnet":
        model = ResNet()

    if args.parallel:
        model = nn.DataParallel(model)

    print('=' * 50)

    train_model(model, trainloader, devloader, args.epochs, args.lr, args.model_path, args.task)

    print('=' * 50)

if __name__ == "__main__":
    main()
