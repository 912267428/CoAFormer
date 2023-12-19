import torch
import torchvision.models as models
import os
import torch.nn as nn

class double_resnet50(torch.nn.Module):
    def __init__(self):
        super(double_resnet50, self).__init__()
        resume_path = os.path.join(os.path.dirname(__file__), 'moco_v2_800ep_pretrain_torchvision.pth.tar')
        param = torch.load(resume_path)['model']
        new_param = {}
        for key in param.keys():
            if 'fc' in key:
                continue
            new_param[key] = param[key]

        backbone = models.resnet50(weights=None)
        backbone.load_state_dict(new_param, strict=False)
        resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']
        resnet_module_list = [getattr(backbone, l) for l in resnet_feature_layers]

        self.layer1 = nn.Sequential(*resnet_module_list[:3])
        self.layer2 = nn.Sequential(*resnet_module_list[3:5])
        self.layer3 = nn.Sequential(*resnet_module_list[5])
        self.layer4 = nn.Sequential(*resnet_module_list[6])

    def forward(self, I1):
        O1 = self.layer1(I1)
        O2 = self.layer2(O1)
        O3 = self.layer3(O2)
        O4 = self.layer4(O3)

        return O4, O3, O2, O1


def get_backbone_resnet50():
    resume_path = os.path.join(os.path.dirname(__file__), 'moco_v2_800ep_pretrain_torchvision.pth.tar')
    param = torch.load(resume_path)['model']
    new_param = {}
    for key in param.keys():
        if 'fc' in key:
            continue
        new_param[key] = param[key]

    backbone = models.resnet50(weights = None)
    backbone.load_state_dict(new_param, strict=False)
    resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']
    resnet_module_list = [getattr(backbone, l) for l in resnet_feature_layers]
    last_layer_idx = resnet_feature_layers.index('layer3')
    backbone = torch.nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
    feat_dim = 1024
    # output: (B, 1024, 30, 30)
    return backbone, feat_dim


def get_backbone_resnet50_output5feature():
    feat_dim = [1024, 512, 256, 64]
    return double_resnet50(), feat_dim


if __name__ == '__main__':
    # x = torch.FloatTensor(5, 3, 480, 480)
    # backbone, _ = get_backbone_resnet50()
    # out1 = backbone(x)
    # print(out1.shape)

    model = get_backbone_resnet50_output5feature()
    x = torch.cuda.FloatTensor(5, 3, 480, 480)
    model = model.cuda()

    out = model(x)
    print(out)