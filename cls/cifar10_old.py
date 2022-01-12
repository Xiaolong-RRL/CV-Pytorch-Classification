"""
参考网易云课堂《深度学习与PyTorch入门实战》
https://github.com/zhuozhudd/PyTorch-Course-Note/tree/master/ch08_CIFAR10_ResNet
https://blog.csdn.net/sunqiande88/article/details/80100891
"""
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from models.backbones import *
import matplotlib.pyplot as plt
from utils.utils import adjust_lr


def main():
    # 网络模型设置
    # 定义是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('cuda_count: ', torch.cuda.device_count())
    # model = ResNet18()
    # model = ResNet50()
    model = ResNet18().to(device)
    if torch.cuda.device_count() > 1:  # device_count() = 4
        device_ids = [0, 1, 2, 3]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        cudnn.benchmark = True
    # reproducible
    # 设置全局随机种子，使得参数随机化初始值一致
    torch.manual_seed(42)

    # Hyper parameters
    EPOCH = 240
    BATCH_SIZE = 128
    LR = 0.1
    STEP = 80
    DOWNLOAD_CIFAR10 = False  # 若已经下载好了，就写上False
    ROOT = '../data/cifar10'

    # download CIFAR10
    # ../data/不存在或者为空，则需要重新下载CIFAR10数据集
    if not (os.path.exists('../data/cifar10')) or not os.listdir('../data/cifar10'):
        DOWNLOAD_CIFAR10 = True

    # 加载 CIFAR10 训练集
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 四周填充零，再把图像随机裁剪成32 x 32大小
        # transforms.Resize((32, 32)),  # 调整图像大小，以匹配CNN的输入
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    # 加载 CIFAR10 训练集
    train_data = datasets.CIFAR10(
        root=ROOT, train=True, transform=train_transforms, download=DOWNLOAD_CIFAR10)
    # 将训练集进行批处理，batch_size=BATCH_SIZE
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 加载 CIFAR10 测试集
    test_data = datasets.CIFAR10(
        root=ROOT, train=False, transform=test_transforms, download=DOWNLOAD_CIFAR10)
    test_loader = DataLoader(test_data, batch_size=100,
                             shuffle=False, num_workers=2)

    # Cifar-10的标签
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 查看所用数据集格式
    print('Training data shape: ', train_data.data.shape)
    print('Testing data shape: ', test_data.data.shape)

    # 损失函数设置
    criterion = nn.CrossEntropyLoss().to(device)  # 包含了softmax
    # 优化器设置
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=0.9, weight_decay=5e-4)

    # Training
    # 加载上次保存的参数
    PATH = '../weights/ResNet18_CIFAR10_Pytorch.pth'
    if os.path.exists(PATH):
        print('loading weight ... ')
        model.load_state_dict(torch.load(PATH))
    else:
        f = open(PATH, 'w')
        f.close()

    total_test_acc = []
    total_train_acc = []
    # 开始训练
    for epoch in range(EPOCH):
        # 动态设置学习率
        adjust_lr(optimizer, epoch, LR, STEP)

        model.train()
        sum_loss = 0
        correct = 0
        total = 0
        train_acc = 0
        for batch, (x, label) in enumerate(train_loader):
            # x:[b,3,32,32], label:[b]
            # train_outputs:[b,10]
            x, label = x.to(device), label.to(device)
            train_outputs = model(x)
            loss = criterion(train_outputs, label)
            # 反向传播更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            pred = train_outputs.argmax(dim=1)
            total += label.size(0)
            correct += torch.eq(pred, label).float().sum().item()
            train_acc = correct / total
            # 每训练200个batch打印一次训练集的loss和准确率
            if (batch + 1) % 100 == 0:
                print('[INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(epoch + 1,
                                                                                             batch + 1,
                                                                                             loss.item(),
                                                                                             train_acc))
        # 计算每个epoch内训练集的acc
        total_train_acc.append(train_acc)

        # 每训练完一个epoch输出测试集的loss和acc
        with torch.no_grad():
            correct = 0
            total = 0
            test_acc = 0
            for test_x, test_labels in test_loader:
                model.eval()
                test_x, test_labels = test_x.to(device), test_labels.to(device)
                test_outputs = model(test_x)
                pred = test_outputs.argmax(dim=1)
                total += test_labels.size(0)
                correct += torch.eq(pred, test_labels).float().sum().item()
            test_acc = correct / total
            print(
                '[INFO] Epoch-{}-Test Accurancy: {:.3f}'.format(epoch + 1, test_acc), '\n')
        total_test_acc.append(test_acc)

    # 保存该次训练结果
    torch.save(model.state_dict(), PATH)

    # 数据可视化
    plt.figure()
    plt.plot(range(EPOCH), total_train_acc, label='Train Accurancy')
    plt.plot(range(EPOCH), total_test_acc, label='Test Accurancy')
    plt.xlabel('Epoch')
    plt.ylabel('Accurancy %')
    plt.title('ResNet18-CIFAR10-Accurancy')
    plt.legend()
    plt.savefig('output/cls/ResNet18-CIFAR10-Accurancy.jpg')  # 自动保存plot出来的图片
    plt.show()


if __name__ == '__main__':
    tic = time.time()
    main()
    toc = time.time()
    # 计算本次运行时间
    print('Running time: ' + str(toc - tic) + 's')
