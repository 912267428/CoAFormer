import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from tqdm import tqdm
import PIL.Image as Image
import Modules


trans = transforms.Compose([
    transforms.Resize([480, 480]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 将图像转换为Tensor

gt_trans = transforms.Compose([
    transforms.Resize([30, 30]),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.bool)])

gt_trans_up = transforms.Compose([
    transforms.Resize([480, 480]),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.bool)])


def get_input(image_path):
    im = Image.open(image_path).convert('RGB')
    im = trans(im)
    im = im.unsqueeze(0)
    return im


def get_gt(gt_path):
    gt = Image.open(gt_path).convert('1')
    # gt.show()
    gt = gt_trans(gt)
    return gt

def get_gt_Up(gt_path):
    gt = Image.open(gt_path).convert('1')
    # gt.show()
    gt = gt_trans_up(gt)
    return gt

def is_neg(gt):
    zero = torch.zeros_like(gt)
    if zero.equal(gt):
        return True
    return False

def cal_acc_iou(pre, gt):
    pre = pre.numpy()
    gt = gt.numpy()
    h, w = gt.shape
    inter = (pre & gt).astype(np.float32)
    union = (pre | gt).astype(np.float32)

    acc = np.sum((pre == gt).astype(np.float32)) / h / w
    iou = inter.sum() / union.sum() if np.sum(gt.astype(np.float32)) > 0 else None
    return acc, iou



def Validation_Internet(backbone, netHead, data_dir, upsample, device, logger=None, old=True):
    backbone.eval()
    netHead.eval()

    all_mean_acc = []
    all_mean_iou = []
    for i_dir, dir in enumerate(os.listdir(data_dir)):
        if not os.path.isdir(os.path.join(data_dir, dir)):
            continue
        data_path = os.path.join(data_dir, dir)
        gt_root = os.path.join(data_path, 'GroundTruth')

        image_list = os.listdir(data_path)
        image_list.remove('GroundTruth')
        if old:
            image_list.remove('GroundTruth_old')

        single_iou = []
        single_acc = []
        loop = tqdm(range(len(image_list)))
        loop.set_description(f'eval dir [{i_dir + 1}/{len(os.listdir(data_dir))}], dir:{dir}')
        for orign_idx in loop:
            orign_path = os.path.join(data_path, image_list[orign_idx])
            orign_image = get_input(orign_path).to(device)
            gt_name = image_list[orign_idx].replace('.jpg', '.png')
            gt_path = os.path.join(gt_root, gt_name)
            if upsample:
                gt = get_gt_Up(gt_path).squeeze()
            else:
                gt = get_gt(gt_path).squeeze()
            gt.to(device)
            if is_neg(gt):
                continue

            orign_feat = backbone(orign_image)

            pair_list = image_list.copy()
            del pair_list[orign_idx]
            o_all_iou = []
            o_all_acc = []
            for pair_idx in range(len(image_list) - 1):
                pair_path = os.path.join(data_path, pair_list[pair_idx])
                pair_image = get_input(pair_path).to(device)
                pair_feat = backbone(pair_image)
                pair_gt_path = os.path.join(gt_root, pair_list[pair_idx].replace('.jpg', '.png'))
                if upsample:
                    pair_gt = get_gt_Up(pair_gt_path).squeeze()
                else:
                    pair_gt = get_gt(pair_gt_path).squeeze()
                if is_neg(pair_gt):
                    continue

                out1, _ = netHead(orign_feat, pair_feat)
                out1.squeeze()
                mask1 = (out1 > 0.5)
                o_acc, o_iou = cal_acc_iou(mask1.cpu(), gt.cpu())

                o_all_acc.append(o_acc)
                o_all_iou.append(o_iou)

            single_iou.append(np.mean(o_all_iou))
            single_acc.append(np.mean(o_all_acc))
            loop.set_postfix(iou=np.mean(o_all_iou), acc=np.mean(o_all_acc))
        all_mean_iou.append(np.mean(single_iou))
        all_mean_acc.append(np.mean(single_acc))
        loop.set_postfix(iou=all_mean_acc[-1], acc=all_mean_acc[-1])
        if logger is not None:
            logger.info(f'@val {dir}:mean IoU:{all_mean_iou[-1]}, mean Acc:{all_mean_acc[-1]}')
    return all_mean_iou, all_mean_acc


def Test_Internet(backbone, netHead, data_dir, upsample, device, logger=None, old=True):
    backbone.eval()
    netHead.eval()

    all_mean_acc = []
    all_mean_iou = []
    for i_dir, dir in enumerate(os.listdir(data_dir)):
        if not os.path.isdir(os.path.join(data_dir, dir)):
            continue
        data_path = os.path.join(data_dir, dir)
        gt_root = os.path.join(data_path, 'GroundTruth')

        image_list = os.listdir(data_path)
        image_list.remove('GroundTruth')
        if old:
            image_list.remove('GroundTruth_old')

        single_iou = []
        single_acc = []
        loop = tqdm(range(len(image_list)))
        loop.set_description(f'eval dir [{i_dir + 1}/{len(os.listdir(data_dir))}], dir:{dir}')
        for orign_idx in loop:
            orign_path = os.path.join(data_path, image_list[orign_idx])
            orign_image = get_input(orign_path).to(device)
            gt_name = image_list[orign_idx].replace('.jpg', '.png')
            gt_path = os.path.join(gt_root, gt_name)
            if upsample:
                gt = get_gt_Up(gt_path).squeeze()
            else:
                gt = get_gt(gt_path).squeeze()
            gt.to(device)
            if is_neg(gt):
                continue

            orign_feat = backbone(orign_image)

            pair_list = image_list.copy()
            del pair_list[orign_idx]
            o_all_iou = []
            o_all_acc = []
            o_best_iou = 0
            best_iou_pacc = 0
            for pair_idx in range(len(image_list) - 1):
                pair_path = os.path.join(data_path, pair_list[pair_idx])
                pair_image = get_input(pair_path).to(device)
                pair_feat = backbone(pair_image)
                pair_gt_path = os.path.join(gt_root, pair_list[pair_idx].replace('.jpg', '.png'))
                if upsample:
                    pair_gt = get_gt_Up(pair_gt_path).squeeze()
                else:
                    pair_gt = get_gt(pair_gt_path).squeeze()
                if is_neg(pair_gt):
                    continue

                out1, _ = netHead(orign_feat, pair_feat)
                out1.squeeze()
                mask1 = (out1 > 0.5)
                o_acc, o_iou = cal_acc_iou(mask1.cpu(), gt.cpu())
                if o_iou > o_best_iou:
                    o_best_iou = o_iou
                    best_iou_pacc = o_acc
                    best_pair_idx = pair_idx

                o_all_acc.append(o_acc)
                o_all_iou.append(o_iou)

            single_iou.append(o_best_iou)
            single_acc.append(best_iou_pacc)
            loop.set_postfix(iou=o_best_iou, acc=best_iou_pacc, p_idx=best_pair_idx)
        all_mean_iou.append(np.mean(single_iou))
        all_mean_acc.append(np.mean(single_acc))
        if logger is not None:
            logger.info(f'@val {dir}:mean IoU:{all_mean_iou[-1]}, mean Acc:{all_mean_acc[-1]}')
    return all_mean_iou, all_mean_acc

def validation_train_byvoc_DCsample_fusion_no_freeze():
    eval_dir_root = '../Data/Internet/eval'

    # 准备model
    pth_path = '../Checkpoints/CoAFormer/train_byvoc_DCsample_fusion/BestIou.pth'
    param = torch.load(pth_path)
    backbone, _ = Modules.get_backbone_resnet50_output5feature()
    netHead = Modules.get_CoAFormerHead_UpSample()
    netHead.load_state_dict(param['encoder'])
    backbone = backbone.cuda()
    netHead = netHead.cuda()
    # iou, acc = Test_Internet(backbone, netHead, eval_dir_root, True, 'cuda', old=True)
    iou, acc = Validation_Internet(backbone, netHead, eval_dir_root, True, 'cuda', old=True)
    print(iou, np.mean(iou))
    print(acc, np.mean(acc))

if __name__ == '__main__':
    eval_dir_root = '../Data/Internet/eval'

    #准备model
    pth_path = '../Checkpoints/CoAFormer/train_byvoc_DCsample_fusion/BestIou.pth'
    param = torch.load(pth_path)
    # backbone, _ = Modules.get_backbone_resnet50()
    # netHead = Modules.get_CoAFormerhead()
    backbone, _ = Modules.get_backbone_resnet50_output5feature()
    netHead = Modules.get_CoAFormerHead_UpSample()
    netHead.load_state_dict(param['encoder'])
    backbone = backbone.cuda()
    netHead = netHead.cuda()
    iou, acc = Test_Internet(backbone, netHead, eval_dir_root, True, 'cuda', old=True)
    print(iou, np.mean(iou))
    print(acc, np.mean(acc))