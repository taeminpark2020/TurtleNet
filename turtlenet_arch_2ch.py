import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
import SelfAttnLayer
#from .utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# class Bottleneck(nn.Module):
#     expansion = 4
#     __constants__ = ['downsample']
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out

# '''Attention'''
# class AttentionConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
#         super(AttentionConv, self).__init__()
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.groups = groups
#
#         assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
#
#         self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
#         self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)
#
#         self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#
#         self.reset_parameters()
#
#     def forward(self, x):
#         batch, channels, height, width = x.size()
#
#         padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
#         q_out = self.query_conv(x)
#         k_out = self.key_conv(padded_x)
#         v_out = self.value_conv(padded_x)
#
#         k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
#         v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
#
#         k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
#         k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
#
#         k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
#         v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
#
#         q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
#
#         out = q_out * k_out
#         out = F.softmax(out, dim=-1)
#         out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
#
#         return out
#
#     def reset_parameters(self):
#         init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
#         init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
#         init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
#
#         init.normal_(self.rel_h, 0, 1)
#         init.normal_(self.rel_w, 0, 1)
#
# class AttentionStem(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
#         super(AttentionStem, self).__init__()
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.groups = groups
#         self.m = m
#
#         assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
#
#         self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
#         self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
#         self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)
#
#         self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
#         self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])
#
#         self.reset_parameters()
#
#     def forward(self, x):
#         batch, channels, height, width = x.size()
#
#         padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
#
#         q_out = self.query_conv(x)
#         k_out = self.key_conv(padded_x)
#         v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)
#
#         k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
#         v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)
#
#         k_out = k_out[:, :, :height, :width, :, :]
#         v_out = v_out[:, :, :, :height, :width, :, :]
#
#         emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
#         emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
#         emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
#         emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)
#
#         v_out = emb * v_out
#
#         k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
#         v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
#         v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
#
#         q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
#
#         out = q_out * k_out
#         out = F.softmax(out, dim=-1)
#         out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)
#
#         return out
#
#     def reset_parameters(self):
#         init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
#         init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
#         for _ in self.value_conv:
#             init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')
#
#         init.normal_(self.emb_a, 0, 1)
#         init.normal_(self.emb_b, 0, 1)
#         init.normal_(self.emb_mix, 0, 1)
# '''Attention'''

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1024, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()

        self.attn_layer = SelfAttnLayer.AttentionConv(64, 64, kernel_size=5, padding=2)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.elu = nn.ELU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

        #Reconstruction
        self.norm_layer = nn.LayerNorm([512,8,8])
        self.upsample = nn.Upsample(scale_factor=2)
        # self.norm_layer = nn.LayerNorm([2048, 8, 8])

        #for basicblock
        self.transpose_conv_0 = nn.ConvTranspose2d(512,256,3,2,1,1)
        self.transpose_conv_1 = nn.ConvTranspose2d(512,128,3,2,1,1)
        self.transpose_conv_2 = nn.ConvTranspose2d(256,64,3,2,1,1)
        self.transpose_conv_3 = nn.ConvTranspose2d(128,16,3,2,1,1)
        #self.transpose_conv_4 = nn.ConvTranspose2d(32,8,3,2,1,1)

        #self.reduce_noise = nn.Conv2d(2, 2, 1)
        self.reduce_noise = nn.Conv2d(16,2,1)
        #Reduce filter
        #self.reduce_filter_0 = nn.ConvTranspose2d(512,128,3,2,1,1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.attn_layer(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.layer4(x3)

        x = self.norm_layer(x)

        x = self.transpose_conv_0(x)
        x = torch.cat((x, x3),1)
        x = self.elu(x)

        x = self.transpose_conv_1(x)
        x = torch.cat((x,x2),1)
        x = self.elu(x)

        x = self.transpose_conv_2(x)
        x = torch.cat((x,x1),1)
        x = self.elu(x)

        x = self.transpose_conv_3(x)
        x = self.elu(x)

        # x = self.transpose_conv_4(x)
        # x = self.elu(x)

        x = self.upsample(x)
        x = self.reduce_noise(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def TurtleNet(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('TurtleNet', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

#Resnet18
#[2, 2, 2, 2]
#Resnet34
#[3, 4, 6, 3]