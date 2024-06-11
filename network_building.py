import torch
import torch.optim as optim

from torch.utils.data import dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

channel = 3  # the channel of the images


## this class is used to prepare data for training, make each input correctly  correspond to the groundtruth
class NumbersDataset(dataset.Dataset):
    def __init__(self, trainset, labels):
        self.samples = trainset.squeeze() / 1.0
        self.lables = labels / 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.Tensor(self.samples[idx])
        label = torch.Tensor(self.lables[idx])
        return data, label


## build the network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=12, stride=4, padding=0)
        self.bacthnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(6, 3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=15, stride=10, padding=0)
        self.bacthnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(4,2)
        self.linear = nn.Linear(896, 3)
        self.dropout1d = nn.Dropout(p=0.1)
    def forward(self, x):  # x:3*800*2000
        x1 = self.bacthnorm1(self.conv1(x))  # 32x498*198
        x2 = self.pool1(F.relu(x1))  # x2:32*165*65

        x3 = self.bacthnorm2(self.conv2(x2))  # x3:64*16*6
        x4 = self.pool2(F.relu(x3))  # x4:64*7*2
        x5 = x4.view(x4.size(0), -1)#896*1
        x6 = self.linear(x5)#1X3
        x7 = self.dropout1d(x6)
        return x7


## This is to train the network  the trained structures and parameters will saved in a file, and returned
def build_network(train_loader,val_loader,epochs,pre_trained_net=None):

    if pre_trained_net is None:
        net = Net()
        net.train()
    else:
        net = torch.load(pre_trained_net)  # dataset_trained(day1and2)2.pth   dataset_train_on_batch(day1_2)(without s_value).pth
        net.train()
    # net=torch.load() you can use our well trained network which names:

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 使用带有动量的随机梯度下降
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)#0.01
    #Lr = 0.001
    #optimizer = optim.Adam(net.parameters(), lr=Lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    loss_list = []
    not_800 = 0
    loss_record = 0
    val_loss_list = []
    val_acc_list = []
    for epoch in range(epochs):
        net.train()
        for batch, (X, y) in enumerate(train_loader):
            # 正向传播

            if X.shape != torch.Size([1, 3, 800, 2000]):
                #print('训练集有1张不是800*2000\n')
                not_800 +=1
                continue
            # 梯度归零
            optimizer.zero_grad()
            y_pred = net(X)
            # print(torch.argmax(y_pred))
            # print('----\n')
            # print(y)
            # 计算损失

            y_one_hot = F.one_hot(y,num_classes=3)
            y_one_hot = y_one_hot.float()

            loss = criterion(y_pred, y_one_hot)
            loss_record += loss.data.item()
            # 梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
        ave_loss_record = loss_record / len(train_loader)
        loss_list.append(ave_loss_record)
        print(f"----epoch{epoch} loss:{ave_loss_record}-----------\n")
        loss_record = 0
        val_loss = 0
        predict_true = 0
        predict_false = 0
        #net.eval()
        for val_batch, (val_X, val_y) in enumerate(val_loader):
            if val_X.shape != torch.Size([1, 3, 800, 2000]):
                print('validation 里有一张不是800X2000\n')
                continue
            val_y_pred = net(val_X)
            val_y_one_hot = F.one_hot(val_y, num_classes=3)
            val_y_one_hot = val_y_one_hot.float()

            loss = criterion(val_y_pred, val_y_one_hot)
            val_loss += loss.data.item()

            outputs = torch.argmax(val_y_pred)
            if outputs == val_y:
                predict_true += 1
            else:
                predict_false += 1
        val_acc = predict_true / (predict_false + predict_true)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss / len(val_loader))
        print(f"----epoch{epoch} val_loss:{val_loss / len(val_loader)}-----------\n")
        print(f"----epoch{epoch} val_acc:{val_acc}-----------\n")
        if val_acc >= 0.95:
            torch.save(net, f'network_in_training_{val_acc}.pth')

    torch.save(net, "trained_network_2.pth")
    trained_network_name = "trained_network_2.pth"
    with open('train_loss_value_2.txt', 'w') as file1:
        file1.write(str(loss_list))
    with open('val_loss_value_2.txt', 'w') as file2:
        file2.write(str(val_loss_list))
    with open('val_acc_value_2.txt', 'w') as file3:
        file3.write(str(val_acc_list))
    print('Finished Training\n')
    print(f"不是800X2000的有{not_800}张")
    return trained_network_name

