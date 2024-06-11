
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
def merge_datasets(dataset, sub_dataset):
    '''
        需要合并的Attributes:
            classes (list): List of the class names sorted alphabetically.
            class_to_idx (dict): Dict with items (class_name, class_index).
            samples (list): List of (sample path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
    '''
    # 合并 classes
    dataset.classes.extend(sub_dataset.classes)
    dataset.classes = sorted(list(set(dataset.classes)))
    # 合并 class_to_idx
    dataset.class_to_idx.update(sub_dataset.class_to_idx)
    # 合并 samples
    dataset.samples.extend(sub_dataset.samples)
    # 合并 targets
    dataset.targets.extend(sub_dataset.targets)
    return dataset

def data_prcoess(img_path,batch_size):

    data_path = img_path  # 'D:\Apps\CNN_project\data'

    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomRotation([180, 180]),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    ])

    data_aug = torchvision.datasets.ImageFolder(root=data_path,
                                                transform=transform_aug)
    transform_orig = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    ])

    data_ori = torchvision.datasets.ImageFolder(root=data_path,
                                                transform=transform_orig)

    full_data = merge_datasets(data_ori, data_aug)

    train_size = int(len(full_data) * 0.8)  # 这里train_size是一个长度矢量，并非是比例，我们将训练和测试进行8/2划分
    test_size = int(len(full_data) * 0.1)
    val_size = len(full_data) - train_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(full_data, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    return train_loader, test_loader, val_loader





# for epoch in range(3):  # 查看前3个epoch图像
#     step = 0
#     for data in train_loader:
#         print(data)
#         # imgs, targets = data  # 读取图像和标签
#         # print(imgs.shape)
#         # print(targets)
#
#         step = step + 1


