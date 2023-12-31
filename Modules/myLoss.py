import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=True):
        super(DiceLoss, self).__init__()
        if smooth:
            self.epsilon = 1
        else:
            self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = predict.view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

def get_DiceLoss(smooth=True):
    return DiceLoss(smooth=smooth)