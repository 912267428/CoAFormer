import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

def trainOneEpoch(train_loader, backbone, netHead, optimizer, criterion, history, logger, epoch, args):
    backbone.eval()
    netHead.train()

    loss_log = []
    acc_log = []

    loop = tqdm(train_loader, total=args.iters_ep)
    loop.set_description(f'Epoch:[{epoch + 1}/{args.n_epoch}]')

    for i, data in enumerate(loop):
        I1, I2, M1, M2 = data
        I1 = I1.to(args.device)
        I2 = I2.to(args.device)
        M1 = M1.to(args.device)
        M2 = M2.to(args.device)

        optimizer.zero_grad()
        with torch.no_grad():
            I1 = backbone(I1)
            I2 = backbone(I2)

        out1, out2 = netHead(I1, I2)
        pred = torch.cat((out1, out2), dim=0)
        target = torch.cat((M1, M2), dim=0)

        if isinstance(criterion, list):
            loss_bce = criterion[0](pred, target)
            loss_dice = criterion[1](pred, target)
            loss = args.Dice_rate*loss_dice + args.BCE_rate*loss_bce
        else:
            loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())
        target_size = target.size()
        with torch.no_grad():
            right_num = ((pred > 0.5).type(torch.FloatTensor).cuda() == target).type(torch.IntTensor).sum().item()
            acc = right_num / (target_size[0]*target_size[1]*target_size[2]*target_size[3])
            acc_log.append(acc)

        logger.info('epoch{} iters{}!loss:{:.3f} acc:{:.3f}'.format(epoch+1, i, loss.item(),acc))
        loop.set_postfix(loss=loss.item(), acc=acc)
        if i >= args.iters_ep:
            break

    history['trainAcc'].append(np.mean(acc_log))
    history['trainLoss'].append(np.mean(loss_log))
    loop.set_postfix(epoch_loss=np.mean(loss_log), epoch_acc=np.mean(acc_log))
    logger.info('@this epoch mean loss:{}, mean acc:{}'.format(np.mean(loss_log), np.mean(acc_log)))

    return backbone, netHead, optimizer, history
