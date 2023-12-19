import argparse
import os

import PIL.Image
import numpy as np
from torchvision import transforms
import PIL.Image as Image
import Modules
import torch
import cv2 as cv


def get_trans(size1, size2):
    tot = transforms.Compose([
        transforms.Resize([480, 480]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    toPIL1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([size1[1], size1[0]])
    ])
    toPIL2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([size2[1], size2[0]])
    ])
    return tot, toPIL1, toPIL2


def get_mask(backbone, netHead, image1, image2):
    f1, f2 = backbone(image1), backbone(image2)
    out1, out2 = netHead(f1, f2)
    out1 = out1.squeeze()
    out2 = out2.squeeze()
    mask1 = (out1 > 0.5).float()
    mask2 = (out2 > 0.5).float()

    return mask1, mask2


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--show-mask', type=bool, default=True, help='show mask')
    parser.add_argument('--show-image', type=bool, default=True, help='show masked image')
    parser.add_argument('--save', type=bool, default=False, help='save')
    # model
    parser.add_argument('--model-name', type=str, default='CoAFormer', help='model name')
    parser.add_argument('--sub-name', type=str, default='train_by_voc2012_noUpSample', help='model sub name')
    parser.add_argument('--use-sample', type=str, default=None, choices=[None, 'Double_Crossing'],
                        help='use up sample or not')  # 使用上采样时给出上采样的类名
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')

    # image
    parser.add_argument('--root', type=str, default='../Data/Internet/eval_old')
    parser.add_argument('--cla', type=str, default='Car100')
    parser.add_argument('--image1', type=str, default='0002')
    parser.add_argument('--image2', type=str, default='0018')

    args = parser.parse_args()
    return args


def get_masked_image(image_path, mask, y=50):
    image = cv.imread(image_path)
    mask = np.array(mask)
    assert image.shape[0:-1] == mask.shape, 'The dimensions of image and mask should be the same'
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i][j] < y:
                image[i,j,:] = 0
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image = PIL.Image.fromarray(image)
    return image


if __name__ == '__main__':
    args = get_args()
    modle_path = args.model_name + '/' + args.sub_name
    Out_path = os.path.join('../Checkpoints', modle_path)

    I1_path = os.path.join(args.root, args.cla, args.image1 + '.jpg')
    I2_path = os.path.join(args.root, args.cla, args.image2 + '.jpg')
    # I1_path = os.path.join(args.root, args.cla, args.image1 + '.bmp')
    # I2_path = os.path.join(args.root, args.cla, args.image2 + '.bmp')
    o_image1 = Image.open(I1_path).convert('RGB')
    o_image2 = Image.open(I2_path).convert('RGB')
    size1 = o_image1.size
    size2 = o_image2.size

    trans = get_trans(size1, size2)

    image1 = trans[0](o_image1).view(1, 3, 480, 480)
    image2 = trans[0](o_image2).view(1, 3, 480, 480)
    image1 = image1.to(args.device)
    image2 = image2.to(args.device)

    if args.use_sample is None:
        backbone, feat_dim = Modules.get_backbone_resnet50()
        netHead = Modules.get_CoAFormerhead()
    else:
        backbone, feat_dim = Modules.get_backbone_resnet50_output5feature()
        netHead = Modules.get_CoAFormerHead_UpSample(args.use_sample)
    backbone = backbone.to(args.device)
    netHead = netHead.to(args.device)
    param = torch.load(os.path.join(Out_path, 'BestIou.pth'))
    netHead.load_state_dict(param['encoder'])
    backbone.eval()
    netHead.eval()

    mask1, mask2 = get_mask(backbone, netHead, image1, image2)

    mask1 = trans[1](mask1)
    mask2 = trans[2](mask2)

    masked_image1 = get_masked_image(I1_path, mask1)
    masked_image2 = get_masked_image(I2_path, mask2)

    if args.show_mask:
        mask1.show()
        mask2.show()
    if args.show_image:
        masked_image1.show()
        masked_image2.show()

    if args.save:
        mask1.save('./mask/' + args.cla + '/' + args.image1 + '_' + args.image2 + '_1.png')
        mask2.save('./mask/' + args.cla + '/' + args.image1 + '_' + args.image2 + '_2.png')

        masked_image1.save('./mask/' + args.cla + '/' + 'masked' + args.image1 + '_' + args.image2 + '_1.png')
        masked_image2.save('./mask/' + args.cla + '/' + 'masked' + args.image1 + '_' + args.image2 + '_2.png')