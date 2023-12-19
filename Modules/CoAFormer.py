import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn.modules.activation import MultiheadAttention
from typing import Union, Callable
from torch import Tensor

import Modules


class PositionEncodingSine2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super(PositionEncodingSine2D, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, isTarget=False):
        '''
        input x: B, C, H, W
        return pos: B, C, H, W

        '''
        not_mask = torch.ones(x.size()[0], x.size()[2], x.size()[3]).to(x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            ## no diff between source and target

            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class EncoderLayerInnerAttention(nn.Module):
    """
    Transformer encoder with all paramters
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, pos_weight, feat_weight):
        super(EncoderLayerInnerAttention, self).__init__()

        self.pos_weight = pos_weight
        self.feat_weight = feat_weight
        self.inner_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                              dim_feedforward=dim_feedforward, dropout=dropout,
                                                              activation=activation)
        self.posEncoder = PositionEncodingSine2D(d_model // 2)

    def forward(self, x, y):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        '''

        bx, cx, hx, wx = x.size()

        by, cy, hy, wy = y.size()

        posx = self.posEncoder(x)
        posy = self.posEncoder(y)

        featx = self.feat_weight * x + self.pos_weight * posx
        featy = self.feat_weight * y + self.pos_weight * posy

        ## input of transformer should be : seq_len * batch_size * feat_dim
        featx = featx.flatten(2).permute(2, 0, 1)
        featy = featy.flatten(2).permute(2, 0, 1)

        ## input of transformer: (seq_len*2) * batch_size * feat_dim
        len_seq_x, len_seq_y = featx.size()[0], featy.size()[0]

        output = torch.cat([featx, featy], dim=0)

        with torch.no_grad():
            src_mask = torch.BoolTensor(hx * wx + hy * wy, hx * wx + hy * wy).fill_(True).to(output.device)
            src_mask[:hx * wx, :hx * wx] = False
            src_mask[hx * wx:, hx * wx:] = False

        output = self.inner_encoder_layer(output, src_mask=src_mask)

        outx, outy = output.narrow(0, 0, len_seq_x), output.narrow(0, len_seq_x, len_seq_y)
        outx, outy = outx.permute(1, 2, 0).view(bx, cx, hx, wx), outy.permute(1, 2, 0).view(by, cy, hy, wy)

        return outx, outy


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class CoAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None):
        super(CoAttentionEncoderLayer, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.co_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                          **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, feat1, feat2):
        q, k, v = feat1, feat2, feat2

        out = self.norm1(feat1 + self._ca_block(q, k, v))
        out = self.norm2(out + self.__ff_block(out))

        return out

    def _ca_block(self, q, k, v):
        out = self.co_attn(q, k, v, need_weights=False)[0]

        return self.dropout1(out)

    def __ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class EncoderLayerCOAttention(nn.Module):
    """
    Transformer encoder with all paramters
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(EncoderLayerCOAttention, self).__init__()

        self.co_encoder_layer_12 = CoAttentionEncoderLayer(d_model=d_model, nhead=nhead,
                                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                                           activation=activation)

        self.co_encoder_layer_21 = CoAttentionEncoderLayer(d_model=d_model, nhead=nhead,
                                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                                           activation=activation)

    def forward(self, featx, featy):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        '''
        bx, cx, hx, wx = featx.size()
        by, cy, hy, wy = featy.size()

        ## input of transformer should be : seq_len * batch_size * feat_dim
        featx = featx.flatten(2).permute(2, 0, 1)
        featy = featy.flatten(2).permute(2, 0, 1)

        ## input of transformer: (seq_len*2) * batch_size * feat_dim
        len_seq_x, len_seq_y = featx.size()[0], featy.size()[0]

        # 对比crossattention的改变
        # 单独输入，不需要cat

        outx = self.co_encoder_layer_12(featx, featy)
        outy = self.co_encoder_layer_21(featy, featx)

        outx, outy = outx.permute(1, 2, 0).view(bx, cx, hx, wx), outy.permute(1, 2, 0).view(by, cy, hy, wy)

        return outx, outy


class EncoderLayerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, pos_weight, feat_weight, layer_type):
        super(EncoderLayerBlock, self).__init__()

        co_encoder_layer = EncoderLayerCOAttention(d_model, nhead, dim_feedforward, dropout, activation)
        att_encoder_layer = EncoderLayerInnerAttention(d_model, nhead, dim_feedforward, dropout, activation, pos_weight,
                                                       feat_weight)

        if layer_type[0] == 'C':
            self.layer1 = co_encoder_layer
        elif layer_type[0] == 'I':
            self.layer1 = att_encoder_layer

        if layer_type[1] == 'C':
            self.layer2 = co_encoder_layer
        elif layer_type[1] == 'I':
            self.layer2 = att_encoder_layer

    def forward(self, featx, featy):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        '''
        featx, featy = self.layer1(featx, featy)
        featx, featy = self.layer2(featx, featy)

        return featx, featy


### --- Transformer Encoder --- ###

class Encoder(nn.Module):
    def __init__(self, feat_dim, pos_weight=0.1, feat_weight=1, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation='relu', layer_type=None,
                 drop_feat=0.1, have_final=True):
        super(Encoder, self).__init__()
        if layer_type is None:
            layer_type = ['C', 'I', 'C', 'I', 'C', 'I']
        self.have_final = have_final
        self.num_layers = num_layers
        self.feat_proj = nn.Conv2d(feat_dim, d_model, kernel_size=1)
        self.drop_feat = nn.Dropout2d(p=drop_feat)
        self.encoder_blocks = nn.ModuleList([EncoderLayerBlock(d_model, nhead, dim_feedforward, dropout, activation,
                                                               pos_weight, feat_weight, layer_type[i * 2: i * 2 + 2])
                                             for i in range(num_layers)])
        if self.have_final:
            self.final_linear = nn.Conv2d(d_model, 1, kernel_size=1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        '''
        featx = self.feat_proj(x)
        featx = self.drop_feat(featx)

        featy = self.feat_proj(y)
        featy = self.drop_feat(featy)

        for i in range(self.num_layers):
            featx, featy = self.encoder_blocks[i](featx, featy)

        if self.have_final:
            outx = self.sigmoid(self.final_linear(featx))
            outy = self.sigmoid(self.final_linear(featy))
        else:
            outx = featx
            outy = featy

        return outx, outy


class CoAFormerHead(nn.Module):
    def __init__(self, feat_dim=1024, pos_weight=0.1, feat_weight=1, dropout=0.1, activation='relu', mode='small',
                 layer_type=None, drop_feat=0.1, have_final=True):
        super(CoAFormerHead, self).__init__()

        if layer_type is None:
            layer_type = ['C', 'I', 'C', 'I', 'C', 'I']
        if mode == 'tiny':
            d_model = 128
            nhead = 2
            num_layers = 3
            dim_feedforward = 256

        elif mode == 'small':
            d_model = 256
            nhead = 2
            num_layers = 3
            dim_feedforward = 256

        elif mode == 'base':
            d_model = 512
            nhead = 8
            num_layers = 3
            dim_feedforward = 2048

        elif mode == 'large':
            d_model = 512
            nhead = 8
            num_layers = 6
            dim_feedforward = 2048

        self.net = Encoder(feat_dim, pos_weight, feat_weight, d_model, nhead, num_layers, dim_feedforward, dropout,
                           activation, layer_type, drop_feat, have_final=have_final)

    def forward(self, x, y):
        '''
        input x: B, C, H, W
        input y: B, C, H, W
        '''
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        outx, outy = self.net(x, y)
        return outx, outy

    def initialize(self):
        for l in self.modules():
            if isinstance(l, nn.Linear):
                nn.init.kaiming_normal_(l.weight.data)


class Double_Crossing_UpSample(nn.Module):
    def __init__(self, feat_dim, d_model=256):
        super(Double_Crossing_UpSample, self).__init__()
        self.Relu = nn.ReLU()
        self.O1_upconv = nn.Conv2d(feat_dim[0], feat_dim[1], 1)
        self.O1_bn_1 = nn.BatchNorm2d(feat_dim[1])
        self.O1_aux = nn.Conv2d(feat_dim[1], feat_dim[1], 3, 2, 1)
        self.O1_bn_2 = nn.BatchNorm2d(feat_dim[1])
        self.O1_end = nn.Conv2d(feat_dim[1] * 2, d_model, 1)
        self.O1_bn_3 = nn.BatchNorm2d(d_model)
        self.O1_DeConv = nn.ConvTranspose2d(d_model * 2, d_model // 2, 4, 2, 1)
        self.O1_bn_4 = nn.BatchNorm2d(d_model // 2)

        self.O2_upconv = nn.Conv2d(feat_dim[1], feat_dim[2], 1)
        self.O2_bn_1 = nn.BatchNorm2d(feat_dim[2])
        self.O2_aux = nn.Conv2d(feat_dim[2], feat_dim[2], 3, 2, 1)
        self.O2_bn_2 = nn.BatchNorm2d(feat_dim[2])
        self.O2_end = nn.Conv2d(feat_dim[2] * 2, d_model // 2, 1)
        self.O2_bn_3 = nn.BatchNorm2d(d_model // 2)
        self.O2_DeConv = nn.ConvTranspose2d((d_model // 2) * 2, d_model // 4, 4, 2, 1)
        self.O2_bn_4 = nn.BatchNorm2d(d_model // 4)

        self.O3_upconv = nn.Conv2d(feat_dim[2], feat_dim[3], 1)
        self.O3_bn_1 = nn.BatchNorm2d(feat_dim[3])
        self.O3_aux = nn.Conv2d(feat_dim[3], feat_dim[3], 3, 2, 1)
        self.O3_bn_2 = nn.BatchNorm2d(feat_dim[3])
        self.O3_end = nn.Conv2d(feat_dim[3] * 2, d_model // 4, 1)
        self.O3_bn_3 = nn.BatchNorm2d(d_model // 4)
        self.O3_DeConv = nn.ConvTranspose2d((d_model // 4) * 2, d_model // 8, 4, 2, 1)
        self.O3_bn_4 = nn.BatchNorm2d(d_model // 8)

        self.O4_upconv = nn.Conv2d(feat_dim[3], d_model // 8, 1)
        self.O4_bn_1 = nn.BatchNorm2d(d_model // 8)
        self.O4_DeConv = nn.ConvTranspose2d((d_model // 8) * 2, d_model // 16, 4, 2, 1)
        self.O4_bn_2 = nn.BatchNorm2d(d_model // 16)

        self.end_up = nn.Conv2d(d_model // 16, 1, 1)

        self.initialize()

    def forward(self, net, X):
        X1, X2, X3, X4 = X

        c1 = self.Relu(torch.cat((self.O1_bn_1(self.O1_upconv(X1)), self.O1_bn_2(self.O1_aux(X2))), dim=1))
        c1 = self.Relu(self.O1_bn_3(self.O1_end(c1)))
        feat = torch.cat((net, c1), dim=1)
        feat = self.Relu(self.O1_bn_4(self.O1_DeConv(feat)))

        c2 = self.Relu(torch.cat((self.O2_bn_1(self.O2_upconv(X2)), self.O2_bn_2(self.O2_aux(X3))), dim=1))
        c2 = self.Relu(self.O2_bn_3(self.O2_end(c2)))
        feat = torch.cat((feat, c2), dim=1)
        feat = self.Relu(self.O2_bn_4(self.O2_DeConv(feat)))

        c3 = self.Relu(torch.cat((self.O3_bn_1(self.O3_upconv(X3)), self.O3_bn_2(self.O3_aux(X4))), dim=1))
        c3 = self.Relu(self.O3_bn_3(self.O3_end(c3)))
        feat = torch.cat((feat, c3), dim=1)
        feat = self.Relu(self.O3_bn_4(self.O3_DeConv(feat)))

        c4 = self.Relu(self.O4_bn_1(self.O4_upconv(X4)))
        feat = torch.cat((feat, c4), dim=1)
        feat = self.Relu(self.O4_bn_2(self.O4_DeConv(feat)))

        feat = self.end_up(feat)
        return feat

    def initialize(self):
        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_normal_(l.weight.data)
            if isinstance(l, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(l.weight.data)


class Easy_UpSample(nn.Module):
    def __init__(self, feat_dim, d_model=256):
        super(Easy_UpSample, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(feat_dim[0], d_model, 1)
        self.bn1_1 = nn.BatchNorm2d(d_model)
        self.deconv1 = nn.ConvTranspose2d(d_model * 2, d_model // 2, 4, 2, 1)
        self.bn1_2 = nn.BatchNorm2d(d_model // 2)

        self.conv2_1 = nn.Conv2d(feat_dim[1], d_model // 2, 1)
        self.bn2_1 = nn.BatchNorm2d(d_model // 2)
        self.deconv2 = nn.ConvTranspose2d((d_model // 2) * 2, d_model // 4, 4, 2, 1)
        self.bn2_2 = nn.BatchNorm2d(d_model // 4)

        self.conv3_1 = nn.Conv2d(feat_dim[2], d_model // 4, 1)
        self.bn3_1 = nn.BatchNorm2d(d_model // 4)
        self.deconv3 = nn.ConvTranspose2d((d_model // 4) * 2, d_model // 8, 4, 2, 1)
        self.bn3_2 = nn.BatchNorm2d(d_model // 8)

        self.conv4_1 = nn.Conv2d(feat_dim[3], d_model // 8, 1)
        self.bn4_1 = nn.BatchNorm2d(d_model // 8)
        self.deconv4 = nn.ConvTranspose2d((d_model // 8) * 2, d_model // 16, 4, 2, 1)
        self.bn4_2 = nn.BatchNorm2d(d_model // 16)

        self.end_up = nn.Conv2d(d_model // 16, 1, 1)

        self.initialize()

    def forward(self, net, X):
        X1, X2, X3, X4 = X
        feat = self.Relu(self.bn1_2(self.deconv1(torch.cat((self.bn1_1(self.conv1_1(X1)), net), dim=1))))
        feat = self.Relu(self.bn2_2(self.deconv2(torch.cat((self.bn2_1(self.conv2_1(X2)), feat), dim=1))))
        feat = self.Relu(self.bn3_2(self.deconv3(torch.cat((self.bn3_1(self.conv3_1(X3)), feat), dim=1))))
        feat = self.Relu(self.bn4_2(self.deconv4(torch.cat((self.bn4_1(self.conv4_1(X4)), feat), dim=1))))

        feat = self.end_up(feat)
        return feat

    def initialize(self):
        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_normal_(l.weight.data)
            if isinstance(l, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(l.weight.data)


class CoAFormerHead_UpSample(nn.Module):
    def __init__(self, UpSamplename='Double_Crossing', Head_pretrain=None, feat_dim=None, pos_weight=0.1, feat_weight=1,
                 dropout=0.1, activation='relu',
                 mode='small', layer_type=None, drop_feat=0.1, have_final=False):
        super(CoAFormerHead_UpSample, self).__init__()
        if layer_type is None:
            layer_type = ['C', 'I', 'C', 'I', 'C', 'I']
        if feat_dim is None:
            feat_dim = [1024, 512, 256, 64]
        if mode == 'small':
            d_model = 256
        elif mode == 'tiny':
            d_model = 128
        else:
            d_model = 512

        self.CoAFormerHead = CoAFormerHead(feat_dim=feat_dim[0], pos_weight=pos_weight, feat_weight=feat_weight,
                                           dropout=dropout, activation=activation, mode=mode,
                                           layer_type=layer_type, drop_feat=drop_feat, have_final=have_final)
        if Head_pretrain is not None:
            param = torch.load(Head_pretrain)
            self.CoAFormerHead.load_state_dict(param['encoder'], strict=False)
        upsample_order = UpSamplename + f'_UpSample({feat_dim}, {d_model})'
        self.UpSample = eval(upsample_order)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X1, X2):
        feat1 = X1[0]
        feat2 = X2[0]
        outX, outY = self.CoAFormerHead(F.normalize(feat1, dim=1), F.normalize(feat2, dim=1))

        feat1 = self.UpSample(outX, X1)
        feat2 = self.UpSample(outY, X2)
        mask1 = self.sigmoid(feat1)
        mask2 = self.sigmoid(feat2)
        return mask1, mask2

    def freeze_Head(self):
        for name, param in self.named_parameters():
            if 'CoAFormerHead' in name:
                param.requires_grad = False


def get_CoAFormerhead(mode='small', feat_dim=1024, pos_weight=0.1, feat_weight=1, dropout=0.1, activation='relu',
                      layer_type=None, drop_feat=0.1):
    if layer_type is None:
        layer_type = ['C', 'I', 'C', 'I', 'C', 'I']
    return CoAFormerHead(mode=mode, feat_dim=feat_dim, pos_weight=pos_weight, feat_weight=feat_weight, dropout=dropout,
                         activation=activation,
                         layer_type=layer_type, drop_feat=drop_feat, have_final=True)


def get_CoAFormerHead_UpSample(UpSamplename='Double_Crossing', Head_pretrain=None, freeze_head=False, mode='small',
                               feat_dim=None, pos_weight=0.1, feat_weight=1,
                               dropout=0.1, activation='relu', layer_type=None, drop_feat=0.1):
    if layer_type is None:
        layer_type = ['C', 'I', 'C', 'I', 'C', 'I']
    if feat_dim is None:
        feat_dim = [1024, 512, 256, 64]

    model = CoAFormerHead_UpSample(UpSamplename, Head_pretrain, mode=mode, feat_dim=feat_dim, pos_weight=pos_weight,
                                   feat_weight=feat_weight, dropout=dropout, activation=activation,
                                   layer_type=layer_type, drop_feat=drop_feat, have_final=False)
    if freeze_head:
        model.freeze_Head()
    return model


if __name__ == '__main__':
    # pass
    x1 = torch.FloatTensor(5, 3, 480, 480)
    x2 = torch.FloatTensor(5, 3, 480, 480)
    import Modules

    backbone, _ = Modules.get_backbone_resnet50_output5feature()
    netHead = get_CoAFormerHead_UpSample('Easy')
    backbone = backbone.cpu()
    netHead = netHead.cpu()

    backbone.eval()
    netHead.train()

    x1 = backbone(x1)
    x2 = backbone(x2)

    out = netHead(x1, x2)

    print(out[0].shape)
    # x1 = torch.FloatTensor(2, 3, 480, 480)
    # x2 = torch.FloatTensor(2, 3, 480, 480)
    # x3 = torch.randn(2, 1, 480, 480)
    # backbone,_ = Modules.get_backbone_resnet50_output5feature()
    # x1 = backbone(x1)
    # x2 = backbone(x2)
    # bce = torch.nn.BCELoss()
    #
    # model = get_CoAFormerHead_UpSample(Head_pretrain='../Checkpoints/CoAFormer/train_by_voc2012_noUpSample/BestIou.pth')
    # model.train()
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
    # for name, param in model.named_parameters():
    #     if 'CoAFormerHead' in name:
    #         param.requires_grad = False
