'''
全连接神经网络实现MNIST分类
1. 导入所需模块库
2. 导入数据
3. FC网络搭建
4. 定义损失函数及优化器
5. train
6. 结果可视化
'''

# 1. 导入所需模块库
import os
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision import transforms
import matplotlib.pyplot as plt

# 2. 导入数据
# pytorch内置集成了MNIST数据集，只需要几行代码就可加载

# ./data/mnist不存在或者为空，则需要重新下载MNIST数据集
DOWNLOAD_MNIST = False
if not (os.path.exists('./data/mnist')) or not os.listdir('./data/mnist'):
    DOWNLOAD_MNIST = True
train_data = mnist.MNIST(
    root='./data/mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST)

test_data = mnist.MNIST(
    root='./data/mnist',
    train=False,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)
print(train_data.data.size())  # torch.Size([60000, 28, 28])
print(test_data.data.size())  # torch.Size([10000, 28, 28])
print(len(train_data))  # 60000
# train_set[i][0]，表示第i张图像的像素信息
# train_set[i][1]，表示第i张图像的标签信息
print(train_data[10][0].reshape(28, 28))
print(train_data[10][1])

# 可视化数据
'''
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    # 随机产生0-59999之间一个随机整数
    idx = random.randint(0, len(train_set))
    digit_0 = train_set[idx][0].numpy()
    digit_image = digit_0.reshape(28, 28)
    ax.imshow(digit_image, interpolation="nearest")
    ax.set_title('label: {}'.format(train_set[idx][1]), fontsize=10, color='black')
plt.show()
'''

# 获取GPU设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)


# 3. FC网络搭建
# 构建输入层、四层全连接层和输出层
# 输入层的节点个数为784,FC1的节点个数为512,FC2的节点个数为256,FC3的节点个数为128,输出层的节点个数是10（分类10个数）
# 每个全连接层后都接一个 激活函数，这里激活函数选用Relu
class Net(nn.Module):
    def __init__(self, in_planes=28 * 28, planes=10):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_planes, 512)
        # in-place operation在pytorch中是指改变一个tensor的值的时候，不经过复制操作，而是直接在原来的内存上改变它的值，
        # 可以把它成为原地操作符
        self.ac1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, 256)
        self.ac2 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(256, 128)
        self.ac3 = nn.ReLU(inplace=True)

        self.fc4 = nn.Linear(128, planes)

    # 神经网络前向传播
    def forward(self, x):
        x = self.ac1(self.fc1(x))
        x = self.ac2(self.fc2(x))
        x = self.ac3(self.fc3(x))
        x = self.fc4(x)
        return x


# 定义网络对象
net = Net()
net.to(device)

# 加载上次保存的参数
PATH = './weights/MNIST_FC_NET.pth'
print(os.path.exists(PATH))

if os.path.exists(PATH):
    print('loading weight ... ')
    net.load_state_dict(torch.load(PATH))
else:
    f = open(PATH, 'w')
    f.close()

# 4. 定义损失函数及优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=5e-4)

# 5. train：前向传播+反向传播梯度更新
losses = []  # 记录训练损失
acces = []  # 记录训练精度
eval_losses = []  # 记录测试损失
eval_acces = []  # 记录测试精度

print()
EPOCH = 20
# 开始训练
for epoch in range(EPOCH):
    train_loss = 0
    train_acc = 0
    net = net.train()
    for batch, (img, label) in enumerate(train_loader):
        # print(img.size()): torch.Size([64, 1, 28, 28])
        # 因为这里的网络是全连接网络，输入图像的尺寸必须要固定，所以需要reshape
        img = img.reshape(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)
        img, label = img.to(device), label.to(device)
        # 前向传播
        out = net(img)
        loss = criterion(out, label)
        # 反向传播更新梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录训练误差
        train_loss += (loss.item())
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        if (batch + 1) % 200 == 0:
            print('[INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(epoch + 1,
                                                                                         batch + 1,
                                                                                         loss.item(),
                                                                                         acc))
        train_acc += acc

    # 统计平均损失函数值和准确率
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))

    # 测试模式
    net = net.eval()
    eval_loss = 0
    eval_acc = 0
    # 测试集不训练
    for img, label in test_loader:
        img = img.reshape(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)
        img, label = img.to(device), label.to(device)
        out = net(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        eval_acc += acc
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))

    print('[INFO] Epoch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f} | Test: Loss-{:.4f}, Accuracy-{:.4f}'.format(
        epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader), eval_loss / len(test_loader),
        eval_acc / len(test_loader)))

# Save my module
torch.save(net.state_dict(), PATH)

# 6. 结果可视化
plt.figure()
plt.suptitle('Test', fontsize=12)
ax1 = plt.subplot(1, 2, 1)
ax1.plot(eval_losses)
ax1.set_title('Loss', fontsize=10, color='r')
ax2 = plt.subplot(1, 2, 2)
ax2.plot(eval_acces)
ax2.set_title('Acc', fontsize=10, color='b')
plt.show()
