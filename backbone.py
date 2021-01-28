import math
import pickle

import numpy as np
import torch
import torch.nn
import torch.nn.functional
import torch.nn.init

from conv2d import Conv2dX100


class OctaveConv(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size,
                 alpha_in=(0.5, 0.5), alpha_out=(0.5, 0.5),
                 stride=1, padding=1, dilation=1, groups=1,
                 bias=False, up_kwargs=None):
        super(OctaveConv, self).__init__()
        if up_kwargs is None:
            self.up_kwargs = {'mode': 'bilinear'}
        else:
            self.up_kwargs = up_kwargs
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, round(in_channels / self.groups), kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.h2g_pool = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.alpha_in = [0]
        tmpsum = 0
        for i in range(len(alpha_in)):
            tmpsum += alpha_in[i]
            self.alpha_in.append(tmpsum)
        self.alpha_out = [0]
        tmpsum = 0
        for i in range(len(alpha_out)):
            tmpsum += alpha_out[i]
            self.alpha_out.append(tmpsum)
        self.inbranch = len(alpha_in)
        self.outbranch = len(alpha_out)

        self.reset_parameters()

    def reset_parameters(self):
        # n = self.in_channels
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, xset):
        # X_h, X_l = x
        yset = []
        ysets = []
        for j in range(self.outbranch):
            ysets.append([])

        if isinstance(xset, torch.Tensor):
            xset = [
                xset,
            ]

        for i in range(self.inbranch):
            if xset[i] is None:
                continue
            if self.stride == 2:
                x = torch.nn.functional.avg_pool2d(xset[i], (2, 2), stride=2)
            else:
                x = xset[i]
            begin_x = int(
                round(self.in_channels * self.alpha_in[i] / self.groups))
            end_x = int(
                round(self.in_channels * self.alpha_in[i + 1] / self.groups))
            if begin_x == end_x:
                continue
            for j in range(self.outbranch):
                begin_y = int(round(self.out_channels * self.alpha_out[j]))
                end_y = int(round(self.out_channels * self.alpha_out[j + 1]))
                if begin_y == end_y:
                    continue
                scale_factor = 2 ** (i - j)
                if self.bias is not None:
                    this_bias = self.bias[begin_y:end_y]
                else:
                    this_bias = None

                this_weight = self.weight[begin_y:end_y, begin_x:end_x, :, :]

                if scale_factor > 1:
                    y = torch.nn.functional.conv2d(x, this_weight, this_bias, 1, self.padding,
                                                   self.dilation, self.groups)
                    y = torch.nn.functional.interpolate(y,
                                                        scale_factor=scale_factor,
                                                        mode='bilinear')
                elif scale_factor < 1:
                    x_resize = torch.nn.functional.max_pool2d(x,
                                                              int(round(1.0 / scale_factor)),
                                                              stride=int(
                                                                  round(1.0 / scale_factor)))
                    y = torch.nn.functional.conv2d(x_resize, this_weight, this_bias, 1,
                                                   self.padding, self.dilation, self.groups)
                else:
                    y = torch.nn.functional.conv2d(x, this_weight, this_bias, 1, self.padding,
                                                   self.dilation, self.groups)
                ysets[j].append(y)

        for j in range(self.outbranch):
            if len(ysets[j]) != 0:
                yset.append(sum(ysets[j]))
            else:
                yset.append(None)
        del ysets
        return yset


class Octconv(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=(3, 3),
                 alpha_in=(0.5, 0.5), alpha_out=(0.5, 0.5),
                 stride=1, padding=1, dilation=1, groups=1,
                 bias=False, up_kwargs=None,
                 norm_layer=torch.nn.BatchNorm2d):
        super(Octconv, self).__init__()
        if up_kwargs is None:
            self.up_kwargs = {'mode': 'bilinear'}
        else:
            self.up_kwargs = up_kwargs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.std_conv = False
        if len(alpha_in) == 1 and len(alpha_out) == 1:
            self.std_conv = True
            self.conv = Conv2dX100(in_channels, out_channels, kernel_size, stride,
                                   padding, dilation, groups, bias)
        else:
            self.conv = OctaveConv(in_channels, out_channels, kernel_size,
                                   alpha_in, alpha_out, stride, padding,
                                   dilation, groups, bias, up_kwargs)
        self.bns = torch.nn.ModuleList()
        self.prelus = torch.nn.ModuleList()
        for i in range(len(alpha_out)):
            if int(round(out_channels * alpha_out[i])) != 0:
                self.bns.append(
                    norm_layer(int(round(out_channels * alpha_out[i]))))
                self.prelus.append(
                    torch.nn.PReLU(int(round(out_channels * alpha_out[i]))))
            else:
                self.bns.append(None)
                self.prelus.append(None)
        self.outbranch = len(alpha_out)
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, xset):
        if self.std_conv:
            if isinstance(xset, torch.Tensor):
                xset = [
                    xset,
                ]

            xset = self.conv(xset[0])
            xset = self.prelus[0](self.bns[0](xset))
        else:
            xset = self.conv(xset)
            for i in range(self.outbranch):
                if xset[i] is not None:
                    xset[i] = self.prelus[i](self.bns[i](xset[i]))
        return xset


class dpconv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, alpha=(0.5, 0.5), padding=1, dilation=1, bias=False,
                 norm_layer=torch.nn.BatchNorm2d):
        super(dpconv, self).__init__()
        self.std_conv = False
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.prelus = torch.nn.ModuleList()
        for i in range(len(alpha)):
            if int(round(in_channels * alpha[i])) >= 1:
                self.convs.append(
                    Conv2dX100(int(round(in_channels * alpha[i])),
                               int(round(out_channels * alpha[i])),
                               kernel_size=(3, 3),
                               groups=int(round(out_channels * alpha[i])),
                               padding=padding,
                               dilation=dilation,
                               bias=bias))
                self.bns.append(norm_layer(int(round(out_channels *
                                                     alpha[i]))))
                self.prelus.append(
                    torch.nn.PReLU(int(round(out_channels * alpha[i]))))
            else:
                self.convs.append(None)
                self.bns.append(None)
                self.prelus.append(None)
        self.outbranch = len(alpha)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = [x, ]
        output = []
        for i in range(self.outbranch):
            if x[i] is not None:
                output.append(self.prelus[i](self.bns[i](self.convs[i](x[i]))))
            else:
                output.append(None)

        return output


class OCBlock(torch.nn.Module):
    def __init__(self, inlist, outlist, stride=1, first=False, decode=False):
        super(OCBlock, self).__init__()

        ninput = int(round(sum(inlist)))
        noutput = int(round(sum(outlist)))
        alpha_in = inlist * 1.0 / ninput
        alpha_out = outlist * 1.0 / noutput
        alpha_in = alpha_in.tolist()
        alpha_out = alpha_out.tolist()
        self.first = first
        self.decode = decode

        if self.first or stride == 2:
            self.conv1x1 = Octconv(ninput, noutput, kernel_size=(3, 3), padding=1, alpha_in=alpha_in,
                                   alpha_out=alpha_out, stride=stride)
        else:
            self.conv1x1 = Octconv(ninput, noutput, kernel_size=(1, 1), padding=0, alpha_in=alpha_in,
                                   alpha_out=alpha_out, stride=stride)
        self.conv3x3_1 = dpconv(noutput, noutput, padding=1, alpha=alpha_out)
        if not decode:
            self.conv3x3_2 = dpconv(noutput, noutput, padding=1, alpha=alpha_out)

    def forward(self, x):
        output = self.conv1x1(x)
        output = self.conv3x3_1(output)
        if not self.decode:
            output = self.conv3x3_2(output)
        return output


class backbone(torch.nn.Module):
    def __init__(self, layer_config, in_channel=3):
        super(backbone, self).__init__()

        self.stages = layer_config[-1]
        self.layer_config = layer_config
        index = 0

        # pre- stage0
        self.stage0 = torch.nn.ModuleList()
        self.stage0.append(
            OCBlock(np.array([in_channel]), self.layer_config[index][1], stride=1, first=True)
        )

        # en1
        index = index + 1
        self.stage1 = torch.nn.ModuleList()
        self.stage1.append(
            OCBlock(self.layer_config[index][0], self.layer_config[index][1]))
        index = index + 1
        for i in range(1, self.stages[0]):
            self.stage1.append(
                OCBlock(self.layer_config[index][0], self.layer_config[index][1]))
            index = index + 1

        # en2
        self.stage2 = torch.nn.ModuleList()
        self.stage2.append(
            OCBlock(self.layer_config[index][0], self.layer_config[index][1], stride=2))
        index = index + 1
        for i in range(1, self.stages[1]):
            self.stage2.append(
                OCBlock(self.layer_config[index][0], self.layer_config[index][1]))
            index = index + 1

        # en3
        self.stage3 = torch.nn.ModuleList()
        self.stage3.append(
            OCBlock(self.layer_config[index][0], self.layer_config[index][1], stride=2))
        index = index + 1
        for i in range(1, self.stages[2]):
            self.stage3.append(
                OCBlock(self.layer_config[index][0],
                        self.layer_config[index][1]))
            index = index + 1

        # en4
        self.stage4 = torch.nn.ModuleList()
        self.stage4.append(
            OCBlock(self.layer_config[index][0], self.layer_config[index][1], stride=2))
        index = index + 1
        for i in range(1, self.stages[3]):
            self.stage4.append(
                OCBlock(self.layer_config[index][0], self.layer_config[index][1]))
            index = index + 1

    def forward(self, x, non_local=None, key_feature=None, value_feature=None, un=None):

        x0 = self.stage0[0](x)
        x1 = x0
        for i in range(self.stages[0]):
            x1 = self.stage1[i](x1)
        x2 = x1
        for i in range(self.stages[1]):
            x2 = self.stage2[i](x2)
        x3 = x2
        for i in range(self.stages[2]):
            x3 = self.stage3[i](x3)
        if non_local is not None:
            un = torch.nn.functional.interpolate(un, scale_factor=0.25)
            encode_non_local_feature = non_local(torch.nn.functional.interpolate(x3[0] * un, scale_factor=0.5),
                                                 key_feature, value_feature)
            encode_non_local_feature = torch.nn.functional.interpolate(encode_non_local_feature, scale_factor=2)
            x3[0] = x3[0] + encode_non_local_feature
        x4 = x3
        for i in range(self.stages[3]):
            x4 = self.stage4[i](x4)

        return x0, x1, x2, x3, x4


def init_layers(basewidth, basic_split=[1, ], stages=[3, 4, 6, 4], in_channel=3):
    layer_config = []
    basic_split = [float(x) for x in basic_split]
    layer_config.append([np.array([
        in_channel,
    ]), basewidth * np.array(basic_split)])
    layer_config.append(
        [basewidth * np.array(basic_split), basewidth * np.array(basic_split)])
    # stage 1
    for i in range(1, stages[0]):
        layer_config.append([
            basewidth * np.array(basic_split),
            basewidth * np.array(basic_split)
        ])
    # stage 2
    layer_config.append([
        basewidth * np.array(basic_split),
        basewidth * 2 * np.array(basic_split)
    ])
    for i in range(1, stages[1] - 1):
        layer_config.append([
            basewidth * 2 * np.array(basic_split),
            basewidth * 2 * np.array(basic_split)
        ])
    layer_config.append([
        basewidth * 2 * np.array(basic_split), basewidth * 2 * np.array([
            1,
        ])
    ])
    # stage 3
    layer_config.append([
        basewidth * 2 * np.array([
            1,
        ]), basewidth * 4 * np.array(basic_split)
    ])
    for i in range(1, stages[2] - 1):
        layer_config.append([
            basewidth * 4 * np.array(basic_split),
            basewidth * 4 * np.array(basic_split)
        ])
    layer_config.append([
        basewidth * 4 * np.array(basic_split), basewidth * 4 * np.array([
            1,
        ])
    ])
    # stage 4
    layer_config.append([
        basewidth * 4 * np.array([
            1,
        ]), basewidth * 4 * np.array(basic_split)
    ])
    for i in range(1, stages[3] - 1):
        layer_config.append([
            basewidth * 4 * np.array(basic_split),
            basewidth * 4 * np.array(basic_split)
        ])
    layer_config.append([
        basewidth * 4 * np.array(basic_split), basewidth * 4 * np.array([
            1,
        ])
    ])
    side2 = basewidth * 2
    side3 = basewidth * 4
    side4 = basewidth * 4
    layer_config.append(np.array([side2, side3, side4]))

    for i in range(len(layer_config)):
        layer_config[i][0] = np.round(layer_config[i][0]).astype(np.int32)
        layer_config[i][1] = np.round(layer_config[i][1]).astype(np.int32)
    layer_config.append(stages)
    return layer_config


def load_layer_config(predefine):
    with open(predefine, "rb") as data:
        return pickle.load(data)


def build_model(basic_split=[1, ], expand=1.0, stages=[3, 4, 6, 4], in_channel=3):
    basewidth = 20
    if expand > 1:
        real_width = int(round(basewidth * expand))
    else:
        real_width = basewidth

    layer_config = init_layers(real_width, basic_split, stages=stages, in_channel=in_channel)

    newmodel = backbone(layer_config=layer_config, in_channel=in_channel)

    return newmodel, layer_config


if __name__ == '__main__':
    images = torch.rand(1, 3, 512, 512).cuda()

    # model, _ = build_model(predefine='./model/csnet-L-x2.bin', fusion=True, num_classes=3)
    model, config = build_model(basic_split=[0.5, 0.5], stages=[2, 2, 2, 2], expand=1)
    model.cuda()

    out = model(images)

    # print(out.shape)
    for o in out:
        for t in o:
            print(t.shape, end='')
        print('\n')
    for f in config:
        print(f)
