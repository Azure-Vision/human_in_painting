# https://www.pythonf.cn/read/110040

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import IPython


def get_images_and_labels(path1, path2, ds_list):
    '''
    从图像数据集的根目录dir_path下获取所有类别的图像名列表和对应的标签名列表
    :param dir_path: 图像数据集的根目录
    :return: images_list, labels_list
    '''

    labels_list = []  # 标签列表
    images_list = []  # 文件名列表

    for ds in ds_list:
        ds_path1 = path1 + ds + '/'
        for png_label_file in os.listdir(ds_path1):

            png_img_file_path = path2 + ds + '/' + png_label_file

            if os.path.exists(png_img_file_path):
                images_list.append(png_img_file_path)
                labels_list.append(ds_path1 + png_label_file)

    print('get_images_and_labels', len(images_list), len(labels_list))
    return images_list, labels_list


class DsDataset(Dataset):
    def __init__(self, pathlabel, pathimg, ds_list, transform=None):
        self.pathlabel = pathlabel    # 数据集根目录
        self.pathimg = pathimg
        self.ds_list = ds_list
        self.transform = transform
        self.images, self.labels = get_images_and_labels(self.pathlabel, self.pathimg, self.ds_list)

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        #img = Image.open(img_path).convert('RGB')
        #img = torch.tensor(img)
        #label = torch.tensor(label)
        sample = {'image': img, 'label': label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        else:
            _transform = transforms.Compose([transforms.ToTensor(),])
            sample['image'] = _transform(sample['image'])
        return sample


if __name__ == '__main__':
    path1 = '/mnt/sda1/songzimeng/cv_project/label/'
    path2 = '/mnt/sda1/songzimeng/cv_project/img/'
    ds_list = ['ds1', 'ds2']

    img, label = get_images_and_labels(path1, path2, ds_list)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    valid_dataset = DsDataset(path1, path2, ds_list)
    #train_dataset = SEVDataset('/mnt/sda1/songzimeng/officialSEV/train/', transform=data_transform)
    #train_dataset = SEVDataset('./officialSEV/test/', transform=data_transform)
    dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
    for index, batch_data in enumerate(dataloader):
        print(index, batch_data['image'].shape, batch_data['label'].shape)

        IPython.embed()
        os._exit(0)