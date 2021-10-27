# PyTorch framework
一个简单且完整PyTorch的框架，提供了各种数据的加载以及简单任务的解决方案，易于扩展。

1.该框架提供了各种数据类型的加载(.wav .mat .jpg .csv .npy)方案。

2.该框架提供了简单分类任务和回归任务的解决方案，以及几个基础模型：CNN、RNN、Attention （ResNet,LSTM,Transformer-encoder)

3.该框架是一个**简单且完整**的框架，只保留了必要的部分并有详细的注释，方便阅读和理解。

并且解耦了各个模块，易于扩展和迁移。迁移到其他任务上只需要更改dataloader和model部分 (还有损失函数)。

用法：

训练和验证

    python main.py --dataset_path ./data/audio/wav2vec/ --model_path  wav2vec --feature wav2vec --feature_dim 768 --task regression --model lstm
    python main.py --dataset_path ./data/vision/AU/ --model_path  AU --feature AU --feature_dim 34 --task regression --model lstm
    python main.py --dataset_path ./data/vision/vggface/ --model_path  vggface --feature vggface --feature_dim 128 --task regression --model lstm
    python main.py --dataset_path ./data/vision/image/ --model_path  image --feature image  --task classification --model resnet
    
测试

    python test.py --dataset_path ./data/audio/wav2vec/ --model_path  ./model/wav2vec_regression_1.pth --feature wav2vec --feature_dim 768 --task regression --model lstm

多卡训练

    CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset_path ./data/vision/image/ --model_path  image --feature image  --task classification --model resnet --parallel

CUDA\_VISIBLE\_DEVICE 和 parallel 搭配使用，单用 parallel 会默认使用所有卡。

<br/><br/><br/><br/><br/>


如果有任何问题，欢迎联系我（cong.cai@nlpr.ia.ac.cn)
