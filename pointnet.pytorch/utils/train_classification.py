from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm    # 快速,可扩展的Python进度条,可以在Python长循环中添加一个进度提示信息,用户只需要封装任意的迭代器tqdm(iterator)


# --dataset /home/zlc/PointCloud/3Dpoint/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0 --nepoch=5 --dataset_type shapenet
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
'''
使用 argparse 的第一步是创建一个 ArgumentParser 对象。
ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
'''

opt = parser.parse_args()
print(opt)
'''
Namespace(batchSize=32, dataset='/home/zlc/PointCloud/3Dpoint/pointnet.pytorch/shapenetcore_partanno_segmentation_benchmark_v0', 
dataset_type='shapenet', feature_transform=False, model='', nepoch=5, num_points=2500, outf='cls', workers=4)
'''

blue = lambda x: '\033[94m' + x + '\033[0m'     # 改变后面输出中test的输出文本颜色

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)     # Random Seed:  8598

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 1. 判断传入的数据集类型是 shapenet数据集 还是 modulenet40数据集
if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,           # 设置路径
        classification=True,
        npoints=opt.num_points)     # 设置数据集，一个三D模型里面有 2500 个点

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)    # 设置测试集
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')

# 2. 上面确定好dataset数据集类型, 下面正式开始加载数据集, 得到的dataloader是迭代器
dataloader = torch.utils.data.DataLoader(
    dataset,                        # 训练数据集由上面第1步得到
    batch_size=opt.batchSize,       # batch_size = 32
    shuffle=True,                   # 所有元素随机排序
    num_workers=int(opt.workers))   # workers=4, 设置4个进程读取数据

testdataloader = torch.utils.data.DataLoader(
    test_dataset,                   # 测试数据集由上面第1步得到
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))


feature, label = iter(dataloader).next()
print(feature.shape, label)# label.shape 和 label.shape[0]一样, 都是
# 通过iter()函数获取这些可迭代对象的迭代器。 对获取到的迭代器不断使⽤next()函数来获取下⼀条数据。
# X是测试集的新图片，label是批量(每次32个)读取的3维模型 真实标签    # 详细看3.5.1节


print(len(dataset), len(test_dataset))  # 12137的训练数据集的迭代器, 每个迭代器32个3维模型,  2874个测试数据集的迭代器, 每个迭代器也是32个三维模型
num_classes = len(dataset.classes)      # 训练数据集一共有16类物体
print('classes', num_classes)           # classes 16


try:
    os.makedirs(opt.outf)
except OSError:
    pass


# 3. 定义网络, 分类任务的网络, 具体在model.py中的class PointNetCls(nn.Module)
classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
print(classifier)   # 打印出来可以观察到详细的网络结构

if opt.model != '':     # 如果模型存在就导入
    classifier.load_state_dict(torch.load(opt.model))


# 4. 定义优化算法, optimizer 是 optim类 创建的实例, scheduler 为了调整学习率
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))       # 使用Adam算法, betas(β)为超参数
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)           # scheduler.step()调用step_size次, 学习率才会调整一次
classifier.cuda()   # 让模型在cuda上训练

print(len(dataset)) # 12137的训练数据集的迭代器, 每个迭代器32个3维模型   相当于dataset中一共读入了12137*32=388384个三维模型
num_batch = len(dataset) / opt.batchSize    # 所以每次参与训练的迭代器数量为 12137 / 32= 379.28个迭代器, 三维模型为 379*32=12128
print('num_batch: %d' % (num_batch))


# 5. 正式开始训练
for epoch in range(opt.nepoch):                 # opt.nepoch = 5, 训练5次
    scheduler.step()                            # 更新一下学习率 每step_size=20 调整一次
    # n = 0
    for i, data in enumerate(dataloader, 0):    # enumerate在字典上是枚举、列举的意思, enumerate参数为可遍历/可迭代的对象(如列表、字符串), 后面的0是索引从0开始
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()   # 复制到cuda上进行计算
        optimizer.zero_grad()                           # 梯度清零，等价于net.zero_grad()
        classifier = classifier.train()                 # 模型设置为训练模式
        pred, trans, trans_feat = classifier(points)
        # print(pred.shape) torch.Size([32, 16])
        loss = F.nll_loss(pred, target)                 # 使用nll_loss损失函数, 传入预测的分数矩阵和真实标签值target, nll损失函数的计算方法见：https://blog.csdn.net/qq_22210253/article/details/85229988
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()                                 # 小批量的损失对模型参数求梯度
        optimizer.step()                                # 梯度下降优化 以batch为单位, 通过调用optim实例的step函数来迭代模型参数, w, b
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum().item()
        # n += target.shape[0]
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct / float(opt.batchSize)))

        if i % 10 == 0:         # 每10个batch执行一次,即为每10*batchSize个点云执行一次验证
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()   # 复制到cuda上进行计算
            classifier = classifier.eval()                  # 模型设置为评估模式
            pred, _, _ = classifier(points)
            # print(pred.shape)                             # torch.Size([32, 16])          输入有32个点云模型, 输出就有32*16大小的矩阵, 32行代表32个点云模型, 每行16个值代表可能属于16个类别的分数
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]               # 按行寻找最大值的位置
            correct = pred_choice.eq(target.data).cpu().sum().item()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct/float(opt.batchSize)))
    # print('n = ', n)
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))      # 每个epoch都保存一个模型


total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):           # enumerate在字典上是枚举、列举的意思, enumerate参数为可遍历/可迭代的对象(如列表、字符串), 后面的0是索引从0开始
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()           # 转移到 GPU 进行训练
    classifier = classifier.eval()                          # 评估模式, 这会关闭dropout
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()                         # 另一个常用函数就是item(), 它可以将一个标量Tensor转换成一个Python number
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
