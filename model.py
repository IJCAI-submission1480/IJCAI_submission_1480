import numpy as np
import torch
import torch.nn

import backbone

use_bn = True


def conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=1, norm=True,
               norm_type=torch.nn.BatchNorm2d, relu_type=torch.nn.ReLU):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=not norm),
        torch.nn.Identity() if not norm else norm_type(out_channel),
        relu_type(inplace=True)
    )


class channel_attention(torch.nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(channel_attention, self).__init__()
        self.max_pooling = torch.nn.AdaptiveMaxPool2d(1)
        self.avg_pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.shared_mlp = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel // ratio, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channel // ratio, in_channel, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, feature):
        max_out = self.max_pooling(feature)
        avg_out = self.avg_pooling(feature)
        output = self.shared_mlp(max_out + avg_out)
        return output


class spatial_attention(torch.nn.Module):
    def __init__(self):
        super(spatial_attention, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        feature = torch.cat((avg_out, max_out), 1)
        output = self.sigmoid(self.conv(feature))
        return output


class short_cut(torch.nn.Module):
    def __init__(self, channel):
        super(short_cut, self).__init__()
        self.conv1 = backbone.OCBlock(np.array([channel]), np.array([channel]))
        # self.conv2 = backbone.OCBlock(np.array([channel]), np.array([channel]))

    def forward(self, feature):
        feature = self.conv1(feature)[0]
        # feature = self.conv2(feature)[0]
        return feature


class side_output(torch.nn.Module):
    def __init__(self, in_channel, img_size):
        super(side_output, self).__init__()
        self.conv = torch.nn.Conv2d(in_channel, 3, 1)

        self.img_size = img_size

    def forward(self, feature):
        feature = self.conv(feature)
        feature = torch.nn.functional.interpolate(feature, size=(self.img_size, self.img_size), mode='nearest')
        return feature


class non_local_attention(torch.nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels):
        super(non_local_attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.f_key = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(self.key_channels)
        )
        self.f_query = self.f_key
        self.f_value = torch.nn.Conv2d(self.in_channels, self.value_channels, kernel_size=1, stride=1, padding=0)
        self.w = torch.nn.Conv2d(self.value_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

        torch.nn.init.constant_(self.w.weight, 0)
        torch.nn.init.constant_(self.w.bias, 0)

    def forward(self, query_feature, key_feature, value_feature):
        batch_size, h, w = query_feature.size(0), query_feature.size(2), query_feature.size(3)

        value = self.f_value(value_feature).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(query_feature).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(key_feature).view(batch_size, self.key_channels, -1)

        query = query / (torch.norm(query, dim=2, keepdim=True) + 1e-7)
        key = key / (torch.norm(key, dim=1, keepdim=True) + 1e-7)
        sim_map = torch.matmul(query, key) + 1

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *query_feature.size()[2:])

        context = self.w(context)

        return context


def long_range_action(x, n, c, qh, ph, qw, pw):
    x = x.reshape(n, c, qh, ph, qw, pw)
    x = x.permute(0, 3, 5, 1, 2, 4)
    x = x.reshape(n * ph * pw, c, qh, qw)
    return x


def short_range_action(x, n, c, qh, ph, qw, pw):
    x = x.reshape(n, ph, pw, c, qh, qw)
    x = x.permute(0, 4, 5, 3, 1, 2)
    x = x.reshape(n * qh * qw, c, ph, pw)
    return x


class efficient_non_local_attention(torch.nn.Module):
    def __init__(self, ph=4, pw=4, long_attention=None, short_attention=None):
        super(efficient_non_local_attention, self).__init__()
        self.ph = ph
        self.pw = pw
        self.long_attention = long_attention
        self.short_attention = short_attention

    def forward(self, query_feature=None, key_feature=None, value_feature=None):
        n, c, h, w = query_feature.size()

        ph, pw = self.ph, self.pw
        qh, qw = h // ph, w // pw

        # long-range
        query_feature = long_range_action(query_feature, n, c, qh, ph, qw, pw)
        key_feature = long_range_action(key_feature, n, c, qh, ph, qw, pw)
        value_feature = long_range_action(value_feature, n, c, qh, ph, qw, pw)

        query_feature = self.long_attention(query_feature, key_feature, value_feature)

        # local
        query_feature = short_range_action(query_feature, n, c, qh, ph, qw, pw)
        key_feature = short_range_action(key_feature, n, c, qh, ph, qw, pw)
        value_feature = short_range_action(value_feature, n, c, qh, ph, qw, pw)

        query_feature = self.short_attention(query_feature, key_feature, value_feature)
        query_feature = query_feature.reshape(n, qh, qw, c, ph, pw)

        return query_feature.permute(0, 3, 1, 4, 2, 5).reshape(n, c, h, w)


class cf_module(torch.nn.Module):
    def __init__(self, channel, low_channel):
        super(cf_module, self).__init__()
        self.conv1 = backbone.OCBlock(np.array([channel + low_channel]), np.array([channel]))

    def forward(self, feature, low_feature):
        _, _, h, w = feature.shape
        low_feature = torch.nn.functional.interpolate(low_feature, size=(h, w))
        feature = torch.nn.functional.sigmoid(torch.mean(low_feature, dim=1, keepdim=True)) * feature
        feature = torch.cat((feature, low_feature), 1)
        feature = self.conv1(feature)[0]
        return feature


class matting_stage1_seg(torch.nn.Module):
    def __init__(self, args):
        super(matting_stage1_seg, self).__init__()

        self.backbone, self.layer_config = backbone.build_model(basic_split=[0.5, 0.5], expand=1, stages=[3, 4, 6, 4],
                                                                in_channel=3)

        self.args = args
        self.channels = self.layer_config[-2]

        self.conv1 = conv_block(self.channels[2], self.channels[1], kernel_size=3, norm=use_bn)

        self.ca = channel_attention(self.channels[2])
        self.s1 = cf_module(self.channels[0], self.channels[1])

        self.s2 = short_cut(self.channels[1])

        self.decode1 = backbone.OCBlock(np.array([self.channels[1]]), np.array([self.channels[1]]), decode=True)
        self.decode2 = backbone.OCBlock(np.array([self.channels[1] + self.channels[1]]), np.array([self.channels[0]]),
                                        decode=True)
        self.decode3 = backbone.OCBlock(np.array([self.channels[0] + self.channels[0]]), np.array([16]), decode=True)

        self.mid_conv = torch.nn.Conv2d(16, 3, kernel_size=1)
        self.up = torch.nn.UpsamplingNearest2d(scale_factor=2)

        self.side1 = side_output(self.channels[1], args.image_size)
        self.side2 = side_output(self.channels[0], args.image_size)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, image):
        x0, x1, x2, x3, x4 = self.backbone(image)
        x1, x2, x3, x4 = x1[0], x2[0], x3[0], x4[0]

        ca = self.ca(x4)
        x4 = x4 * ca
        x4 = self.conv1(x4)
        x2 = self.s1(x2, x4)
        x3 = self.s2(x3)

        x4 = self.up(self.decode1(x4)[0])
        x3 = self.up(self.decode2(torch.cat((x3, x4), 1))[0])
        x2 = self.decode3(torch.cat((x2, x3), 1))[0]

        x2 = self.mid_conv(x2)
        x2 = self.up(x2)
        s1 = self.side1(x4)
        s2 = self.side2(x3)

        return [x2, s1, s2]


class matting_stage2_refine(torch.nn.Module):
    def __init__(self, args):
        super(matting_stage2_refine, self).__init__()

        self.args = args
        self.encoder, self.layer_config = backbone.build_model(basic_split=[0.5, 0.5], expand=1, stages=[2, 2, 2, 2],
                                                               in_channel=6)
        self.channels = self.layer_config[-2]

        self.decode1 = backbone.OCBlock(np.array([self.channels[2]]), np.array([self.channels[1]]), decode=True)
        self.decode2 = backbone.OCBlock(np.array([self.channels[1] + self.channels[1]]), np.array([self.channels[1]]),
                                        decode=True)
        self.decode3 = backbone.OCBlock(np.array([self.channels[1] + self.channels[0]]), np.array([10]), decode=True)
        self.decode4 = backbone.OCBlock(np.array([10 + 10]), np.array([10]), decode=True)
        self.up = torch.nn.UpsamplingNearest2d(scale_factor=2)

        self.c = conv_block(10, 10, 1, padding=0, norm=use_bn)
        self.final_conv = torch.nn.Conv2d(10, 1, kernel_size=3, stride=1, padding=1)

        self.image_conv1 = backbone.OCBlock(np.array([3]), np.array([10, 10]), stride=1, first=True)
        self.image_conv2 = backbone.OCBlock(np.array([10, 10]), np.array([self.channels[1]]), stride=2)
        self.down2 = torch.nn.UpsamplingNearest2d(scale_factor=0.5)
        self.down4 = torch.nn.UpsamplingNearest2d(scale_factor=0.25)
        self.down8 = torch.nn.UpsamplingNearest2d(scale_factor=0.125)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        self.encoder_long_attention = non_local_attention(
            self.channels[1], self.channels[1] // 2, self.channels[1] // 2, self.channels[1])
        self.encoder_short_attention = non_local_attention(
            self.channels[1], self.channels[1] // 2, self.channels[1] // 2, self.channels[1])
        self.decoder_long_attention = non_local_attention(
            self.channels[1], self.channels[1] // 2, self.channels[1] // 2, self.channels[1])
        self.decoder_short_attention = non_local_attention(
            self.channels[1], self.channels[1] // 2, self.channels[1] // 2, self.channels[1])
        self.non_local1 = efficient_non_local_attention(4, 4, self.encoder_long_attention, self.encoder_short_attention)
        self.non_local2 = efficient_non_local_attention(4, 4, self.decoder_long_attention, self.decoder_short_attention)

    def forward(self, image, stage1_output):
        seg_softmax = torch.nn.functional.softmax(stage1_output, 1)
        _, pred_fg_softmax, pred_unknown_softmax = torch.split(seg_softmax, 1, dim=1)

        image_feature = self.down4(self.image_conv2(self.image_conv1(image))[0])
        key_feature = image_feature
        value_feature = image_feature

        x0, x1, x2, x3, x4 = self.encoder(torch.cat((image, seg_softmax), 1),
                                          self.non_local1, key_feature, value_feature, pred_unknown_softmax)

        x1, x2, x3, x4 = x1[0], x2[0], x3[0], x4[0]

        x4 = self.up(self.decode1(x4)[0])
        x3 = torch.cat((x3, x4), 1)

        x3 = self.decode2(x3)[0]
        decode_non_local_feature = self.up(self.non_local2(self.down2(x3) * self.down8(pred_unknown_softmax),
                                                           key_feature, value_feature))
        x3 = self.up(x3 + decode_non_local_feature)

        x2 = torch.cat((x2, x3), 1)
        x2 = self.up(self.decode3(x2)[0])
        x1 = torch.cat((x1, x2), 1)
        x1 = self.decode4(x1)[0]

        pred = self.final_conv(self.c(x1))

        final_pred = pred * pred_unknown_softmax + pred_fg_softmax

        return [final_pred, pred]


class matting_end_to_end(torch.nn.Module):
    def __init__(self, args):
        super(matting_end_to_end, self).__init__()
        self.stage1 = matting_stage1_seg(args)
        self.stage2 = matting_stage2_refine(args)

    def forward(self, image):
        stage1_output = self.stage1(image)
        stage2_output = self.stage2(image, stage1_output[0])
        return stage1_output, stage2_output


if __name__ == '__main__':
    class temp(object):
        def __init__(self):
            self.image_size = 512
            # ...


    images = torch.rand(1, 3, 512, 512).cuda()
    net = matting_end_to_end(temp()).cuda()
    output = net(images)
    for o in output:
        for t in o:
            print(t.shape, end='')
        print('\n')
    exit()
