import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import PIL.Image as Image
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, root_path, trans=None):
        assert os.path.isdir(root_path), f'{root_path} is not a dir'
        self.root_path = root_path

        if trans == None:
            self.trans_I = transforms.Compose([
                transforms.Resize([480, 480]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.trans_M = transforms.Compose([
                transforms.Resize([30, 30]),
                transforms.ToTensor(),
            ])
        else:
            self.trans_I = trans[0]
            self.trans_M = trans[1]

        self.sub_dir = os.listdir(root_path)
        self.sub_num = []
        self.sub_gt_list = []
        self.sub_train_num = []
        for sd in self.sub_dir:
            now_gt_list = os.listdir(os.path.join(root_path, sd, 'GroundTruth'))
            self.sub_gt_list.append(now_gt_list)
            self.sub_num.append(len(now_gt_list))
            self.sub_train_num.append(len(now_gt_list) * (len(now_gt_list) - 1))

    def __len__(self):
        return sum(self.sub_train_num)

    def __getitem__(self, index):
        dir_idx = 0
        for i in range(len(self.sub_dir)):
            if index + 1 > self.sub_train_num[i]:
                dir_idx = dir_idx + 1
                index = index - self.sub_train_num[i]
            else:
                break
        dir = self.sub_dir[dir_idx]
        gt_list = self.sub_gt_list[dir_idx]
        ori_num = self.sub_num[dir_idx]

        pair_idx = index % (ori_num - 1)
        ori_idx = (index - pair_idx) // (ori_num - 1)
        pair_list = gt_list.copy()
        del pair_list[ori_idx]

        M1_path = os.path.join(self.root_path, dir, 'GroundTruth', gt_list[ori_idx])
        M2_path = os.path.join(self.root_path, dir, 'GroundTruth', pair_list[pair_idx])
        I1_path = os.path.join(self.root_path, dir, gt_list[ori_idx].replace('.png', '.jpg'))
        I2_path = os.path.join(self.root_path, dir, pair_list[pair_idx].replace('.png', '.jpg'))

        I1 = Image.open(I1_path).convert('RGB')
        I2 = Image.open(I2_path).convert('RGB')
        M1 = Image.open(M1_path).convert('1')
        M2 = Image.open(M2_path).convert('1')

        I1 = self.trans_I(I1)
        I2 = self.trans_I(I2)
        M1 = self.trans_M(M1)
        M2 = self.trans_M(M2)

        zero = torch.zeros_like(M1)
        if zero.equal(M1) or zero.equal(M2):
            M1 = zero
            M2 = zero

        return I1, I2, M1, M2


class VocDataset(Dataset):
    def __init__(self, root_path, is_train=True, label_size=30, trans=None):
        self.root_path = root_path

        file = open(os.path.join(root_path, 'CosegTrain.txt' if is_train else 'CosegVal.txt'), 'r')
        self.data_list = [line.strip() for line in file]
        file.close()

        if trans == None:
            self.trans_I = transforms.Compose([
                transforms.Resize([480, 480]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.trans_M = transforms.Compose([
                transforms.Resize([label_size, label_size]),
                transforms.ToTensor(),
            ])
        else:
            self.trans_I = trans[0]
            self.trans_M = trans[1]

        if not is_train:
            self.trans_I = transforms.Compose([
                transforms.Resize([480, 480]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.trans_M = transforms.Compose([
                transforms.Resize([label_size, label_size]),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_str = self.data_list[index]
        data_str = data_str.split('#')
        image_name = data_str[0].split(' ')
        class_co_str = data_str[1].split(' ')
        class_co = [int(e) for e in class_co_str]

        I1 = Image.open(os.path.join(self.root_path, 'UsedImage', image_name[0] + '.jpg'))
        I2 = Image.open(os.path.join(self.root_path, 'UsedImage', image_name[1] + '.jpg'))
        I1 = self.trans_I(I1)
        I2 = self.trans_I(I2)

        M1 = Image.open(os.path.join(self.root_path, 'GroundTruth', image_name[0] + '.png'))
        M2 = Image.open(os.path.join(self.root_path, 'GroundTruth', image_name[1] + '.png'))
        M1 = np.array(M1)
        M2 = np.array(M2)

        M1[M1 == 255] = 0
        M2[M2 == 255] = 0
        for c in class_co:
            M1[M1 == c] = 255
            M2[M2 == c] = 255
        M1[M1 != 255] = 0
        M2[M2 != 255] = 0
        M1 = Image.fromarray(M1)
        M2 = Image.fromarray(M2)
        M1 = self.trans_M(M1)
        M2 = self.trans_M(M2)

        return I1, I2, M1, M2


def get_train_loader_byint(dir, batch_size=32):  # Internet数据集上的trainloader
    dataset = MyDataset(dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    return loader


def get_train_loader_byvoc(dir, batch_size=16, istrain=True, label_size=30,trans=None):
    dataset = VocDataset(dir, is_train=istrain, label_size=label_size, trans=trans)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=istrain, num_workers=4)

    return loader


if __name__ == '__main__':
    da = MyDataset('../Data/Internet/train')
    trainload = DataLoader(da, batch_size=1, shuffle=True, num_workers=8)

    for data in tqdm(trainload):
        I1, I2, M1, M2 = data

    print(da)

    # da = VocDataset('../Data/VOC2012_cos')
    # trainload = DataLoader(da, batch_size=100, shuffle=True, num_workers=8)
    # for data in trainload:
    #     print(123)
