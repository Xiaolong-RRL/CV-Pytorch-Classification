import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # CV数据库模块
import matplotlib.pyplot as plt
from utils.utils import plot_curve, plot_image

# reproducible
# 设置全局随机种子，使得参数随机化初始值一致
torch.manual_seed(1)

# Hyper parameters
EPOCH = 2
BATCH_SIZE = 64
LR = .001
DOWNLOAD_MNIST = False  # 若已经下载好了，就写上False

# download MNIST
# ./data/mnist不存在或者为空，则需要重新下载MNIST数据集
if not (os.path.exists('./data/mnist')) or not os.listdir('./data/mnist'):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./data/mnist',
    train=True,
    # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W)
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./data/mnist',
    train=False,
    # 转换 PIL.Image or numpy.ndarray 成 torch.FloatTensor (C x H x W)
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# plot one example
print(train_data.data.size())
print(test_data.data.size())
# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# 批训练50 samples, 1 channel, 28 x 28 (50, 1, 28, 28)
# DataLoader主要是对数据进行batch划分，其中shuffle规定要不要打乱数据
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
# pick 2000 samples
# shape from (2000, 28, 28) to (2000, 1, 28, 28),
# value in range(0,1)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor) / 255.
test_y = test_data.test_labels


# define my network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input shape (1 x 28 x 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,  # same填充，保证卷积后的图片大小不变
            ),  # output shape (16 x 28 x 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (16 x 14 x 14)
        )
        # input shape (16 x 14 x 14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32 x 14 x 14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # output shape (32 x 7 x 7)
        )
        # 全连接层，输入为二维数据，输出为10个预测值
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)
"""
CNN(
  (conv1): Sequential(
    (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (out): Linear(in_features=1568, out_features=10, bias=True)
)
"""

# Training

# 加载上次保存的参数
PATH = 'weights/MNIST_mofan_net.pth'
if os.path.exists(PATH):
    print('loading weight ... ')
    cnn.load_state_dict(torch.load(PATH))
else:
    f = open('PATH', 'w')
    f.close()

# 设置优化器和损失函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
train_loss = []

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            train_loss.append(loss.item())
            print(epoch, step, loss.item())
# plot_curve(train_loss)
# Save my module
torch.save(cnn.state_dict(), PATH)

# Testing_v1 只测试前十个数据
# test_output = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')

# Testing_v2 测试整个数据集
total_correct = 0.0
for x, y in test_loader:
    test_out = cnn(x)
    pred_out = torch.max(test_out, 1)[1]
    correct = pred_out.eq(y).sum().float().item()
    total_correct += correct
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('acc: ', acc)
