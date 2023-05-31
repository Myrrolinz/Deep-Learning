import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random


seed = 3407
random.seed(seed)
torch.manual_seed(seed)  # 根据论文seed:3407 is all your need选择3407种子　　

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 搭建基于SENet的Conv Block和Identity Block的网络结构
class Block(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False):
        super(Block, self).__init__()

        # 各个Stage中的每一大块中每一小块的输出维度，即channel（filter1 = filter2 = filter3 / 4）
        filter1, filter2, filter3 = filters

        self.is_1x1conv = is_1x1conv  # 判断是否是Conv Block
        self.relu = nn.ReLU(inplace=True)  # RELU操作

        # 第一小块， stride = 1(stage = 1) or stride = 2(stage = 2, 3, 4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU()
        )

        # 中间小块
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU()
        )

        # 最后小块，不需要进行ReLu操作
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3),
        )

        # Conv Block的输入需要额外进行卷积和归一化操作（结合Conv Block网络图理解）
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filter3)
            )

        # SENet(结合SENet的网络图理解)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Conv2d(filter3, filter3 // 16, kernel_size=1),  # 16表示r，filter3//16表示C/r，这里用卷积层代替全连接层
            nn.ReLU(),
            nn.Conv2d(filter3 // 16, filter3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_shortcut = x
        x1 = self.conv1(x)  # 执行第一Block操作
        x1 = self.conv2(x1)  # 执行中间Block操作
        x1 = self.conv3(x1)  # 执行最后Block操作

        x2 = self.se(x1)  # 利用SENet计算出每个通道的权重大小
        x1 = x1 * x2  # 对原通道进行加权操作

        if self.is_1x1conv:  # Conv Block进行额外的卷积归一化操作
            x_shortcut = self.shortcut(x_shortcut)

        x1 = x1 + x_shortcut  # Add操作
        x1 = self.relu(x1)  # ReLU操作

        return x1


# 搭建SEResNet50
class SEResnet(nn.Module):
    def __init__(self, cfg):
        super(SEResnet, self).__init__()
        classes = cfg['classes']  # 分类的类别
        num = cfg['num']  # ResNet50[3, 4, 6, 3]；Conv Block和 Identity Block的个数

        # Stem Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage1
        filters = (64, 64, 256)  # channel
        self.Stage1 = self._make_layer(in_channels=64, filters=filters, num=num[0], stride=1)

        # Stage2
        filters = (128, 128, 512)  # channel
        self.Stage2 = self._make_layer(in_channels=256, filters=filters, num=num[1], stride=2)

        # Stage3
        filters = (256, 256, 1024)  # channel
        self.Stage3 = self._make_layer(in_channels=512, filters=filters, num=num[2], stride=2)

        # Stage4
        filters = (512, 512, 2048)  # channel
        self.Stage4 = self._make_layer(in_channels=1024, filters=filters, num=num[3], stride=2)

        # 自适应平均池化，(1, 1)表示输出的大小(H x W)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层 这里可理解为网络中四个Stage后的Subsequent Processing 环节
        self.fc = nn.Sequential(
            nn.Linear(2048, classes)
        )

    # 形成单个Stage的网络结构
    def _make_layer(self, in_channels, filters, num, stride=1):
        layers = []

        # Conv Block
        block_1 = Block(in_channels, filters, stride=stride, is_1x1conv=True)
        layers.append(block_1)

        # Identity Block结构叠加; 基于[3, 4, 6, 3]
        for i in range(1, num):
            layers.append(Block(filters[2], filters, stride=1, is_1x1conv=False))

        # 返回Conv Block和Identity Block的集合，形成一个Stage的网络结构
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem Block环节
        x = self.conv1(x)

        # 执行四个Stage环节
        x = self.Stage1(x)
        x = self.Stage2(x)
        x = self.Stage3(x)
        x = self.Stage4(x)

        # 执行Subsequent Processing环节
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# SeResNet50的参数  （注意调用这个函数将间接调用SEResnet，这里单独编写一个函数是为了方便修改成其它ResNet网络的结构）
def SeResNet50():
    cfg = {
        'num': (3, 4, 6, 3),  # ResNet50，四个Stage中Block的个数（其中Conv Block为1个，剩下均为增加Identity Block）
        'classes': (10)  # 数据集分类的个数
    }

    return SEResnet(cfg)  # 调用SEResnet网络


# 直接设置好默认的分类类别有10类
net = SeResNet50().to(device)
with open('Log/SE-ResNet.txt', 'w') as f:
    f.write(str(net))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)


def train(trainloader, epoch, lr_scheduler=None, log_interval=100):
    # Set model to training mode
    net.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(trainloader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)
        # Zero gradient buffers
        optimizer.zero_grad()
        # Pass data through the network
        output = net(data)
        # Calculate loss
        loss = criterion(output, target)
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()  # w - alpha * dL / dw

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.data.item()))

    # 调整梯度下降算法的学习率
    if lr_scheduler is not None:
        lr_scheduler.step()


def validate(testloader, loss_vector, accuracy_vector):
    net.eval()
    val_loss, correct = 0, 0
    for data, target in testloader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        val_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(testloader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(testloader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(testloader.dataset), accuracy))


if __name__ == '__main__':
    # 设置好对应读取数据集和测试集的loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 128
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    print(net)

    epochs = 10
    lossv, accv = [], []
    # print("hello")

    # 调整学习率 (step_size:每训练step_size个epoch，更新一次参数; gamma:更新lr的乘法因子)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, epochs + 1):
        train(trainloader, epoch, lr_scheduler)
        validate(testloader, lossv, accv)

    # print the figure
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), lossv)
    plt.title('validation loss')
    plt.show()

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), accv)
    plt.title('validation accuracy')
    plt.show()

    # save the module
    PATH = './Log/SE-ResNet.pth'
    torch.save(net.state_dict(), PATH)