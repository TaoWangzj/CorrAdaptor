import torch
import torch.nn as nn
from baselines.loss2 import batch_episym


class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, xa):
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return wei


class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        radix = 4
        self.conv_init1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels // radix, kernel_size=1), )
        self.conv_init2 = nn.Sequential(
            nn.InstanceNorm2d(out_channels // radix, eps=1e-3),
            nn.BatchNorm2d(out_channels // radix),
            nn.ReLU(),
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.conv_init3 = nn.Sequential(
            nn.InstanceNorm2d(out_channels // radix, eps=1e-3),
            nn.BatchNorm2d(out_channels // radix),
            nn.ReLU(),
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.conv_init4 = nn.Sequential(
            nn.InstanceNorm2d(out_channels // radix, eps=1e-3),
            nn.BatchNorm2d(out_channels // radix),
            nn.ReLU(),
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.conv_init5 = nn.Sequential(
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels // radix, kernel_size=1), )
        self.conv_init6 = nn.Sequential(
            nn.InstanceNorm2d(out_channels // radix, eps=1e-3),
            nn.BatchNorm2d(out_channels // radix),
            nn.ReLU(),
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.conv_init7 = nn.Sequential(
            nn.InstanceNorm2d(out_channels // radix, eps=1e-3),
            nn.BatchNorm2d(out_channels // radix),
            nn.ReLU(),
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.conv_init8 = nn.Sequential(
            nn.InstanceNorm2d(out_channels // radix, eps=1e-3),
            nn.BatchNorm2d(out_channels // radix),
            nn.ReLU(),
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.AFF = AFF(out_channels)

    def forward(self, x):
        init1 = self.conv_init1(x)
        init2 = self.conv_init2(init1)
        init3 = self.conv_init3(init2)
        init4 = self.conv_init4(init3)
        wei = self.AFF(torch.cat([init1, init2, init3, init4], dim=1))
        init = torch.cat([init1, init2, init3, init4], dim=1)
        out = wei * init
        out1 = self.conv_init5(out)
        out2 = self.conv_init6(out1)
        out3 = self.conv_init7(out2)
        out4 = self.conv_init8(out3)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        radix = 4
        self.conv_init1 = nn.Sequential(
            nn.Conv2d(channels, out_channels // radix, kernel_size=1), )
        self.conv_init2 = nn.Sequential(
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.conv_init3 = nn.Sequential(
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.conv_init4 = nn.Sequential(
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        # Spatial Correlation Layer
        self.conv_init5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // radix, kernel_size=1), )
        self.conv_init6 = nn.Sequential(
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.conv_init7 = nn.Sequential(
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.conv_init8 = nn.Sequential(
            nn.Conv2d(out_channels // radix, out_channels // radix, kernel_size=1), )
        self.conv_init9 = nn.Sequential(
            nn.Conv2d(points, points // radix, kernel_size=1), )
        self.conv_init10 = nn.Sequential(
            nn.Conv2d(points // radix, points // radix, kernel_size=1), )
        self.conv_init11 = nn.Sequential(
            nn.Conv2d(points // radix, points // radix, kernel_size=1), )
        self.conv_init12 = nn.Sequential(
            nn.Conv2d(points // radix, points // radix, kernel_size=1), )
        self.normalinit = nn.Sequential(
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.normalmiddle = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
        )
        self.normalend = nn.Sequential(
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        init1 = self.conv_init1(x)
        init2 = self.conv_init2(init1)
        init3 = self.conv_init3(init2)
        init4 = self.conv_init4(init3)
        out = (torch.cat([init1, init2, init3, init4], dim=1))
        out = self.normalinit(out)
        out = out.transpose(1, 2)
        middle1 = self.conv_init9(out)
        middle2 = self.conv_init10(middle1)
        middle3 = self.conv_init11(middle2)
        middle4 = self.conv_init12(middle3)
        out = out + self.normalmiddle(torch.cat([middle1, middle2, middle3, middle4], dim=1))
        out = out.transpose(1, 2)
        out1 = self.conv_init5(out)
        out2 = self.conv_init6(out1)
        out3 = self.conv_init7(out2)
        out4 = self.conv_init8(out3)
        out = self.normalend(torch.cat([out1, out2, out3, out4], dim=1))
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


# you can use this bottleneck block to prevent from overfiting when your dataset is small
class OAFilterBottleneck(nn.Module):
    def __init__(self, channels, points1, points2, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            trans(1, 2))
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points1),
            nn.ReLU(),
            nn.Conv2d(points1, points2, kernel_size=1),
            nn.BatchNorm2d(points2),
            nn.ReLU(),
            nn.Conv2d(points2, points1, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1)
        )

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        # x_up: b*c*n*1
        # x_down: b*c*k*1
        embed = self.conv(x_up)  # b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)  # b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class OANBlock(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)

        l2_nums = clusters

        self.l1_1 = []
        for _ in range(self.layer_num // 2):
            self.l1_1.append(PointCN(channels))

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num // 2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(PointCN(2 * channels, channels))
        for _ in range(self.layer_num // 2 - 1):
            self.l1_2.append(PointCN(channels))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)
        self.Aff = AFF(2 * channels)
        self.output = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, data, xs):
        # data: b*c*n*1
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x_down1 = self.down1(x1_1)
        x2_1 = self.l2(x_down1)
        x_up1 = self.up1(x1_1, x2_1)
        wei = self.Aff(torch.cat([x1_1, x_up1], dim=1))
        out = self.l1_2(torch.cat([x1_1, x_up1], dim=1) * wei)
        logits1 = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
        e_hat = weighted_8points(xs, logits1)

        x1, x2 = xs[:, 0, :, :2], xs[:, 0, :, 2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits1, e_hat, residual


class OANet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.iter_num = 1
        depth_each_stage = 12 // (1 + 1)
        self.side_channel = (0 == 2) + (0 == 2)
        self.weights_init = OANBlock(128, 4 + self.side_channel, depth_each_stage, 500)
        self.weights_iter = [OANBlock(128, 6 + self.side_channel, depth_each_stage, 500) for
                             _ in range(1)]
        self.weights_iter = nn.Sequential(*self.weights_iter)

    def forward(self, xs):
        # assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        # batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # xs: b*1*n*c
        input = xs.transpose(1, 3)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, xs)
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1), xs)
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        # e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')  # pytorch 2.0
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into model_zoo for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat