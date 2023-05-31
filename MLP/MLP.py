import torch
import torch.nn as nn
from torchvision import datasets, transforms  # 针对图像进行操作（包括变换、采点、旋转、加噪音等操作）
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    """
    定义网络结构
    """
    def __init__(self):
        """
        初始化网络结构
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)  # 全连接层，图片像素是28 * 28, 所以输入也是28 * 28
        self.fc1_drop = nn.Dropout(0.2)  # 防止过拟合，在训练时随机以20%概率丢掉一些数据
        self.fc2 = nn.Linear(50, 80)  # 全连接层
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)  # 最后有十个类型（1-10），所以将最终的结果映射到10个维度上

    def forward(self, x):
        """
        前馈神经网络，前向传播，默认调用每一层时都会调用forward函数向前传递，每一层接收的是上一层的输出
        :param x: 输入的张量
        :return: 训练出来的每张图片对应的每个类别的概率
        """
        x = x.view(-1, 28 * 28)  # 改变tensor的形状，删除多余的维度 [32, 28 * 28]
        x = F.relu(self.fc1(x))  # 利用Relu激活函数实现激活层
        x = self.fc1_drop(x)  # 一般是将dropout放在激活函数后
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)  # [32, 10] 32张图片，每张图片对应的每个类别的概率
        return F.log_softmax(self.fc3(x), dim = 1)  # log_softmax_log(p), dim = 1是每张图片对第一个维度做softmax

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(400, 80)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)   
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc1_drop = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2_drop = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3_drop = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)   
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.fc1_drop(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)  
        x = self.bn3(self.fc3(x))
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)
        return F.log_softmax(self.fc4(x), dim=1)
        # return F.log_softmax(self.fc3(x), dim=1)

def train(epoch, log_interval=200):
    """
    定义训练函数
    :param epoch: 训练的周期
    :param log_interval: 每训练多少次打印一次loss
    :return:
    """
    model.train()  # 打开训练模式，训练模式和测试模式有些不同（测试模式dropout不会使用）

    # batch_index, (data, target)就是X_train和Y_train（数据和标签）
    for batch_index, (data, target) in enumerate(train_loader):
        data = data.to('cpu')
        target = target.to('cpu')
        optimizer.zero_grad()  # 存的是梯度的信息，将梯度全部置为0
        output = model(data)  # 把数据传入模型
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 将损失自动反向回传
        optimizer.step()  # 更新网络权重，w - learning_rate * dL / dw  （梯度下降）
        if batch_index % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(epoch, batch_index * len(data), len(train_loader.dataset), 100. * batch_index / len(train_loader), loss.data.item()))


def validate(loss_vector, accuracy_vector):
    """
    计算准确率
    :param loss_vector:
    :param accuracy_vector:
    :return:
    """
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to('cpu')
        target = target.to('cpu')
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))



if __name__ == "__main__":
    # 加载数据集
    batch_size = 32  # 每次训练加载的数量
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transforms.ToTensor())  # 加载训练集
    validation_dataset = datasets.MNIST('./data', train=False, download=False, transform=transforms.ToTensor())  # 加载验证集
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # 相当于一个迭代器，把数据放在类似于list中顺序读取
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    # 画图
    pltsize = 2
    plt.figure(figsize=(10*pltsize, pltsize))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.axis("off")
        for (X_train, Y_train) in train_loader:
            plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap="gray_r")
            plt.title('Class: ' + str(Y_train[i].item()))
            break

    # 调用网络           
    model = Net().to('cpu')  # 创建网络，可选Net, Net1和Net3
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 优化器, lr是学习率，momentum动量
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.75, weight_decay=3e-4) # 这一组更好
    criterion = nn.CrossEntropyLoss()  # 定义评价标准，交叉商函数

    epochs = 10
    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch)
        validate(lossv, accv)

    # 画出loss和准确率的图片
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs+1), lossv)
    plt.title('validation loss')
    plt.show()

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1, epochs+1), accv)
    plt.title('validation accuracy')
    plt.show()
    