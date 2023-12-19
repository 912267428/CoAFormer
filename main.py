import argparse
import itertools
from datetime import datetime
import logging
import os
import random

import numpy as np
import torch

import Modules
import Utils
import Train


def set_random_seed(seed_value):
    """
    设置种子点，使得实验可以复现

    :param seed_value: 种子点（int）
    :returns: 无
    """
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True


def get_logger(logdir):
    """
    获取日志logger，并设置日志输入格式、日志等级为INFO()

    :param logdir: 日志保存地址
    :returns: logger
    """

    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s  #%(message)s")

    ts = str(datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "log_{}.log".format(ts))
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)

    return logger


def get_args():
    """
    获取命令行参数args

    :param  无
    :returns: args
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ##mangement
    parser.add_argument('--model-name', type=str, default='CoAFormer', help='model name')
    parser.add_argument('--sub-name', type=str, default='train_by_voc2012_noUpSample', help='model sub name')
    parser.add_argument('--use-sample', type=str, default=None, choices=[None, 'Double_Crossing','Easy'],
                        help='use up sample or not')  # 使用上采样时给出上采样的类名
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
    parser.add_argument('--seed', type=int, default=2023, help='seed')

    ##train param
    parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--n-epoch', type=int, default=100, help='nb of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--iters-ep', type=int, default=10600, help='iters of each epoch')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'],
                        help='choice optimizer')
    parser.add_argument('--loss', type=str, default='fusion', choices=['fusion', 'BCE', 'Dice'], help='choice loss')
    parser.add_argument('--Dice-rate', type=float, default=2.1, help='Dice rate')
    parser.add_argument('--BCE-rate', type=float, default=1.2, help='BCE rate')

    ##dir
    parser.add_argument('--train-dir', type=str, default='./Data/VOC2012_cos', help='data of train')
    parser.add_argument('--vali-dir', type=str, default='./Data/Internet/eval', help='data of vali')
    parser.add_argument('--head-pretrain-dir', type=str,
                        default='./Checkpoints/CoAFormer/train_by_voc2012_noUpSample/BestIou.pth', help='data of vali')
    parser.add_argument('--freeze-head', action='store_true', help='freeze or not')

    ##resume
    parser.add_argument('--resume', action='store_true', help='resume or not')

    ## paramter transformer
    parser.add_argument('--mode', type=str, choices=['tiny', 'small', 'base', 'large'], default='small',
                        help='different size of transformer encoder')
    parser.add_argument('--pos-weight', type=float, default=0.1, help='weight for positional encoding')
    parser.add_argument('--feat-weight', type=float, default=1, help='weight of feature')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout in the transformer layer')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'],
                        help='activation in the transformer layer')
    parser.add_argument('--layer-type', type=str, nargs='+', default=['C', 'I', 'C', 'I', 'C', 'I'],
                        help='which type of layers: I is for inner image attention, C is for Co-attention')
    parser.add_argument('--drop-feat', type=float, default=0.1, help='drop feature to make the task difficult')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    modle_path = args.model_name + '/' + args.sub_name
    Log_path = os.path.join('./Log', modle_path)
    Out_path = os.path.join('./Checkpoints', modle_path)
    if not os.path.exists(Log_path):
        os.mkdir(Log_path)
    if not os.path.exists(Out_path):
        os.mkdir(Out_path)

    set_random_seed(args.seed)
    logger = get_logger(Log_path)
    logger.info('@' + str(args))

    if args.use_sample is None:
        train_loader = Utils.get_train_loader_byvoc(args.train_dir, args.batch_size, True, 30, None)
        backbone, feat_dim = Modules.get_backbone_resnet50()
        netHead = Modules.get_CoAFormerhead(
            mode=args.mode,
            feat_dim=feat_dim,
            pos_weight=args.pos_weight,
            feat_weight=args.feat_weight,
            dropout=args.dropout,
            activation=args.activation,
            layer_type=args.layer_type,
            drop_feat=args.drop_feat,
        )
    else:
        train_loader = Utils.get_train_loader_byvoc(args.train_dir, args.batch_size, True, 480, None)
        backbone, feat_dim = Modules.get_backbone_resnet50_output5feature()
        netHead = Modules.get_CoAFormerHead_UpSample(args.use_sample, args.head_pretrain_dir, args.freeze_head,
                                                     mode=args.mode,
                                                     feat_dim=feat_dim,
                                                     pos_weight=args.pos_weight,
                                                     feat_weight=args.feat_weight,
                                                     dropout=args.dropout,
                                                     activation=args.activation,
                                                     layer_type=args.layer_type,
                                                     drop_feat=args.drop_feat,
                                                     )
    backbone = backbone.to(args.device)
    netHead = netHead.to(args.device)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(itertools.chain(*[filter(lambda p : p.requires_grad, netHead.parameters())]), lr=args.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    else:
        optimizer = eval('torch.optim.' + args.optimizer)(itertools.chain(*[filter(lambda p : p.requires_grad, netHead.parameters())]), args.lr,
                                                          weight_decay=args.weight_decay)

    if args.loss == 'fusion':
        bce = torch.nn.BCELoss()
        dice = Modules.get_DiceLoss(True)
        criterion = [bce, dice]
    elif args.loss == 'BCE':
        criterion = torch.nn.BCELoss()
    elif args.loss == 'Dice':
        criterion = Modules.get_DiceLoss(True)

    history = {'valIoU': [], 'valAcc': [], 'trainAcc': [], 'trainLoss': [], 'end_epoch': -1, 'bestIoU': 0.,
               'bestAcc': 0.}

    if args.resume:
        param = torch.load(os.path.join(Out_path, 'netLast.pth'))
        netHead.load_state_dict(param['encoder'])
        optimizer.load_state_dict(param['optimiser'])
        history = np.load(os.path.join(Out_path, 'history.npy'), allow_pickle=True).item()
        if args.optimizer == 'SGD':
            scheduler.load_state_dict(param['schedule'])

    start_epoch = history['end_epoch'] + 1
    best_iou = history['bestIoU']
    best_acc = history['bestAcc']

    #train loop
    for i in range(start_epoch, args.n_epoch):
        #train
        backbone, netHead, optimizer, history = Train.trainOneEpoch(train_loader, backbone, netHead,
                                                                    optimizer, criterion, history,
                                                                    logger, i, args)
        #validation
        iou, acc = Train.Validation_Internet(backbone, netHead, args.vali_dir,
                                             False if args.use_sample is None else True,
                                             args.device, logger)

        meaniou = np.mean(iou)
        meanacc = np.mean(acc)
        history['valIoU'].append(iou)
        history['valAcc'].append(acc)
        print(f'\nvalIoU:{iou}({meaniou}), valAcc:{acc}({meanacc})\n')
        logger.info(f'@epoch:{i + 1},valIoU:{iou}({meaniou}),valAcc{acc}({meanacc})')

        if args.optimizer == 'SGD':
            scheduler.step()

        param = {'backbone': backbone.state_dict(),
                 'encoder': netHead.state_dict(),
                 'optimiser': optimizer.state_dict()
                 # 'schedule': lr_schedule.state_dict()
                 }
        if args.optimizer == 'SGD':
            param['schedule'] = scheduler.state_dict()

        if meaniou > best_iou:
            best_iou = meaniou
            history['bestIoU'] = meaniou
            torch.save(param, os.path.join(Out_path, 'BestIou.pth'))
            logger.info(f'@epoch:{i + 1},get best IoU:{meaniou}')
        if meanacc > best_acc:
            best_acc = meanacc
            history['bestAcc'] = meanacc
            torch.save(param, os.path.join(Out_path, 'BestAcc.pth'))
            logger.info(f'@epoch:{i + 1},get best Acc:{meanacc}')

        torch.save(param, os.path.join(Out_path, 'netLast.pth'))

        history['end_epoch'] = i
        np.save(os.path.join(Out_path, 'history.npy'), history)
