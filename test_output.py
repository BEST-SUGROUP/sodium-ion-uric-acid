import numpy as np
import torch

from torch.utils.data import dataset

from torch.autograd import Variable


def trained_network_test(test_loader, trained_network_name):


    net = torch.load(
        trained_network_name)  # dataset_trained(day1and2)2.pth   dataset_train_on_batch(day1_2)(without s_value).pth


    #net.eval()
    pred_res = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    not_800 =0
    for inputs in test_loader:
        imgs, label = inputs
        if imgs.shape != torch.Size([1, 3, 800, 2000]):
            not_800 += 1
            continue
        input = Variable(imgs)
        output = net(input)

        outputs = torch.argmax(output)

        pred_res[outputs.data.item()][label.data.item()] += 1
    print(f'测试集有{not_800}张不是800X2000\n')
    return pred_res
