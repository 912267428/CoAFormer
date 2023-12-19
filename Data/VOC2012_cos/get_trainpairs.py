import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def get_class(path):
    mask = Image.open(path)
    arr = np.array(mask)
    arr = np.unique(arr)
    arr = arr[(arr != 0) & (arr != 255)]

    return arr

def get_nextfile(oldfile,new_idx):
    oldfile.close()
    newfile = open(f'./CosegTrain/CosegTrain{new_idx}.txt', 'w')
    return newfile



def get_trainandval():
    num = 0
    file_idx = 1
    file = open('./CosegTrain/CosegTrain1.txt', 'w')

    Ground = './GroundTruth'
    mask_list = os.listdir('./GroundTruth')
    mask_list.sort()
    for z_idx in tqdm(range(len(mask_list))):
        z_class = get_class(os.path.join(Ground, mask_list[z_idx]))
        for y_idx in range(z_idx+1, len(mask_list)):
            y_class = get_class(os.path.join(Ground, mask_list[y_idx]))
            common = np.intersect1d(z_class, y_class)
            if common.size == 0:
                continue
            else:
                string_class = ' '.join(map(str, common))
                file.write(f'{mask_list[z_idx].replace(".png","")} {mask_list[y_idx].replace(".png","")}#{string_class}\n')
                num+=1
                if num==10000:
                    file_idx +=1
                    num=0
                    file = get_nextfile(file, file_idx)
    print(file_idx*10000+num)
    file.close()

def get_train():
    num = 0
    file = open('./CosegTrain.txt', 'w')
    rfile = open('./train.txt', 'r')
    train_list = [line.strip() for line in rfile]
    rfile.close()
    train_list.sort()

    Ground = './GroundTruth'
    for z_idx in tqdm(range(len(train_list))):
        z_class = get_class(os.path.join(Ground, train_list[z_idx]+'.png'))
        for y_idx in range(z_idx + 1, len(train_list)):
            y_class = get_class(os.path.join(Ground, train_list[y_idx]+'.png'))
            common = np.intersect1d(z_class, y_class)
            if common.size == 0:
                continue
            else:
                string_class = ' '.join(map(str, common))
                file.write(
                    f'{train_list[z_idx]} {train_list[y_idx]}#{string_class}\n')
                num += 1
    print(num)
    file.close()

def get_val():
    num = 0
    file = open('./CosegVal.txt', 'w')
    rfile = open('./val.txt', 'r')
    train_list = [line.strip() for line in rfile]
    rfile.close()
    train_list.sort()

    Ground = './GroundTruth'
    for z_idx in tqdm(range(len(train_list))):
        z_class = get_class(os.path.join(Ground, train_list[z_idx]+'.png'))
        for y_idx in range(z_idx + 1, len(train_list)):
            y_class = get_class(os.path.join(Ground, train_list[y_idx]+'.png'))
            common = np.intersect1d(z_class, y_class)
            if common.size == 0:
                continue
            else:
                string_class = ' '.join(map(str, common))
                file.write(
                    f'{train_list[z_idx]} {train_list[y_idx]}#{string_class}\n')
                num += 1
    print(num)
    file.close()

if __name__ == '__main__':
    #get_train()
    get_val()