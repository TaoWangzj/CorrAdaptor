import torch
import torch.nn as nn
from baselines.loss1 import batch_episym
import torch.nn.functional as F
import copy
from timm.models.layers import DropPath

# 利用 FLOW 特性引入 Motion 的 FlowFormer
class Flow_Attention(nn.Module):
    """
    paper: Flowformer: Linearizing transformers with conservation flows.
    """
    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.m_proj = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def kernel(self, x):
        x = torch.sigmoid(x)
        return x

    def my_sum(self, a, b):
        # "nhld,nhd->nhl"
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, q, kv, motion):
        B_, N, C = q.shape
        # qkv = self.qkv(q).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = self.q_proj(q).reshape(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k_proj(kv).reshape(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_proj(kv).reshape(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        m = self.m_proj(motion).reshape(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # kernel
        q, k, m = self.kernel(q), self.kernel(k), self.kernel(m)
        # inject motion information
        eq = 1.0 / (self.my_sum(q + 1e-6, m.sum(dim=2) + 1e-6) + 1e-6)
        ek = 1.0 / (self.my_sum(k + 1e-6, m.sum(dim=2) + 1e-6) + 1e-6)
        eqq = self.my_sum(m + 1e-6, (q * eq[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        ekk = self.my_sum(m + 1e-6, (k * ek[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        # sigmoid
        eqq = torch.sigmoid(eqq)
        ekk = torch.sigmoid(ekk)
        # 相加
        q = q + q * (m * eqq[:, :, :, None])
        k = k + k * (m * ekk[:, :, :, None])
        # normalizer
        sink_incoming = 1.0 / (self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
        source_outgoing = 1.0 / (self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)
        conserved_sink = self.my_sum(q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = self.my_sum(k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        # competition
        source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[2])
        # multiply
        kv = k.transpose(-2, -1) @ (v * source_competition[:, :, :, None])
        x_update = ((q @ kv) * sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
        x = (x_update).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]

    return idx[:, :, :]

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        try:
            # e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
            e,v = torch.linalg.eigh(X[batch_idx,:,:].squeeze(), UPLO='U') # pytorch 2.0
        except Exception as result:
            return None
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.GELU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return F.gelu(out)


def weighted_8points(x_in, logits):
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    if v == None:
        return None
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class GAB(nn.Module):
    """
    Dynamic graph / knn-base graph attention block
    """
    def __init__(self, in_channels=128, k=9):
        super(GAB, self).__init__()
        self.knn_num = k
        self.reduction = 3

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=True),
            nn.InstanceNorm2d(in_channels, eps=1e-5),
            nn.BatchNorm2d(in_channels)
        )

        self.pointcn1 = nn.Sequential(
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        self.pointcn2 = nn.Sequential(
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        if self.knn_num == 9:
            self.spatial_att = nn.Sequential(
                nn.Conv2d(self.knn_num, self.reduction, kernel_size=1),
                nn.BatchNorm2d(self.reduction), nn.GELU(),
                nn.Conv2d(self.reduction, self.knn_num, kernel_size=1),
                nn.BatchNorm2d(self.knn_num),
            )
        if self.knn_num == 6:
            self.spatial_att = nn.Sequential(
                nn.Conv2d(self.knn_num, self.reduction, kernel_size=1),
                nn.BatchNorm2d(self.reduction), nn.GELU(),
                nn.Conv2d(self.reduction, self.knn_num, kernel_size=1),
                nn.BatchNorm2d(self.knn_num),
            )
        self.neighbor_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4), nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4), nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, knn_graph):
        knn_graph = self.embed(knn_graph)
        residual0 = knn_graph
        att1_0 = knn_graph.mean(dim=1).unsqueeze(dim=1)
        att2_0 = knn_graph.max(dim=1)[0].unsqueeze(dim=1)
        att0 = att1_0 + att2_0
        att0 = self.spatial_att(att0.transpose(1, 3))
        att0 = torch.sigmoid(att0).transpose(1, 3)
        knn_graph = knn_graph * att0
        knn_graph = knn_graph + residual0

        residual1 = knn_graph
        knn_graph = self.pointcn1(knn_graph)
        att1_1 = knn_graph.mean(dim=2).unsqueeze(dim=2)
        att2_1 = knn_graph.max(dim=2)[0].unsqueeze(dim=2)
        att1 = att1_1 + att2_1
        att1 = self.neighbor_att(att1)
        att1 = torch.sigmoid(att1)
        knn_graph = knn_graph * att1
        knn_graph = knn_graph + residual1

        residual2 = knn_graph
        knn_graph = self.pointcn2(knn_graph)
        att1_2 = knn_graph.mean(dim=3).unsqueeze(dim=3)
        att2_2 = knn_graph.max(dim=3)[0].unsqueeze(dim=3)
        att2 = att1_2 + att2_2
        att2 = self.channel_att(att2)
        att2 = torch.sigmoid(att2)
        knn_graph = knn_graph * att2
        knn_graph = knn_graph + residual2

        return knn_graph


class AnnularConv(nn.Module):
    """
    Neighborhood aggregation
    """
    def __init__(self, in_channels=128, k=9):
        super(AnnularConv, self).__init__()
        self.in_channel = in_channels
        self.knn_num = k

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )
        if self.knn_num == 6:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),
                nn.BatchNorm2d(self.in_channel), nn.GELU(),
            )

    def forward(self, features):
        B, C, N, _ = features.shape
        out = self.conv1(features)
        out = self.conv2(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class My_BlockLayerScale(nn.Module):
    def __init__(self, dim, nhead, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, init_values=1e-5):
        super().__init__()
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.norm1_3 = norm_layer(dim)
        self.attn = Flow_Attention(dim, nhead)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, q, kv, motion):
        x = self.norm1_1(q)
        y = self.norm1_2(kv)
        motion = self.norm1_3(motion)
        q = q + self.drop_path(self.attn(x, y, motion))
        x = self.norm2(q)
        q = q + self.drop_path(self.gamma * self.mlp(x))
        return q


class LinearTransformers(nn.Module):
    def __init__(self, d_model, nhead, layer_names):
        super(LinearTransformers, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        encoder_layer = My_BlockLayerScale(d_model, self.nhead)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, motion):
        assert self.d_model == feat0.size(2)

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, motion)
                feat1 = layer(feat1, feat1, motion)
            elif name == 'cross':
                feat0 = layer(feat0, feat1)
                feat1 = layer(feat1, feat0)
            else:
                raise KeyError

        return feat0 + feat1


class trans(nn.Module):

    def __init__(self, dim1, dim2):
        super(trans, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):

    def __init__(self, channels, points, out_channels=None):
        super(OAFilter, self).__init__()
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)

        self.conv1 = nn.Sequential(nn.InstanceNorm2d(channels, eps=1e-3),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(),
                                   nn.Conv2d(channels, out_channels, kernel_size=1),
                                   trans(1, 2))

        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
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
        super(diff_pool, self).__init__()
        self.output_points = output_points # clusters
        self.conv = nn.Sequential(
                    nn.InstanceNorm2d(in_channel, eps=1e-3),
                    nn.BatchNorm2d(in_channel),
                    nn.ReLU(),
                    nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        super(diff_unpool, self).__init__()
        self.output_points = output_points
        self.conv = nn.Sequential(nn.InstanceNorm2d(in_channel, eps=1e-3),
                                  nn.BatchNorm2d(in_channel),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        #x_up: b*c*n*1
        #x_down: b*c*k*1
        embed = self.conv(x_up)# b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)# b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class DS_Block(nn.Module):
    def __init__(self, config, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channel = 4 if self.initial is True else 6
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
            nn.InstanceNorm2d(self.out_channel),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.motion_embed = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(1, 1)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, self.out_channel, kernel_size=(1, 1))
        )

        self.gab1 = GAB(self.out_channel, k=self.k_num)
        self.aggregator1 = AnnularConv(self.out_channel, k_num)
        self.gab2 = GAB(self.out_channel, k=self.k_num)
        self.aggregator2 = AnnularConv(self.out_channel, k_num)

        self.ltr1 = LinearTransformers(self.out_channel, 4, ['self', 'self'] * 2)  # recommend a change to 1
        self.ltr2 = LinearTransformers(self.out_channel, 4, ['self', 'self'] * 2)

        self.pool1 = diff_pool(out_channel, 500)
        self.pool2 = diff_pool(out_channel, 500)
        self.oafilter1 = nn.Sequential(OAFilter(out_channel, 500),
                                       OAFilter(out_channel, 500),
                                       OAFilter(out_channel, 500))
        self.oafilter2 = nn.Sequential(OAFilter(out_channel, 500),
                                       OAFilter(out_channel, 500),
                                       OAFilter(out_channel, 500))
        self.unpool1 = diff_unpool(out_channel, 500)
        self.unpool2 = diff_unpool(out_channel, 500)

        self.fusion1 = ResNet_Block(out_channel * 2, out_channel, pre=True)
        self.fusion2 = ResNet_Block(out_channel * 2, out_channel, pre=True)

        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))

        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        if self.predict == True:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, self.out_channel, 1, 1))
            return x_out, y_out, w_out, feature_out

    def trans_shape(self, x, flag):
        if flag == "in":
            return x.transpose(1, 2).squeeze(dim=-1)
        else:
            return x.transpose(1, 2).unsqueeze(dim=-1)

    def forward(self, x, y):
        B, _, N , _ = x.size()
        coor1, coor2 = x[:, :, :, :2], x[:, :, :, 2:4]
        motion = coor2 - coor1  # motion -> [2, 1, 2000, 2]
        motion = self.motion_embed(motion.transpose(1, 3).contiguous())  # motion -> [2, 128, 2000, 1]
        motion = self.trans_shape(motion, "in")  # motion -> [2, 2000, 128] 

        initial_x = x
        x = x.transpose(1, 3).contiguous()  # x -> [2, 4, 2000, 1]
        x = self.conv(x)  # x -> [2, 128, 2000, 1]

        # stack 2 times

        # implict branch
        i_local_graphs = self.pool1(x)  # i_local_graphs -> [B, 128, 500, 1]
        i_local_graphs = self.oafilter1(i_local_graphs)  # i_local_graphs -> [B, 128, 500, 1]
        i_local_context = self.unpool1(x, i_local_graphs)  # i_local_graphs -> [B, 128, 2000, 1]
        # i_gl_context = self.ltr1(self.trans_shape(x, "in"), self.trans_shape(i_local_context, "in"))
        # i_gl_context = self.trans_shape(i_gl_context, "out")  # global & local context
        # explict branch
        e_local_graphs = get_graph_feature(x, k=self.k_num)  # e_local_graphs -> [B, 256, 2000, 9]
        e_local_graphs = self.gab1(e_local_graphs)  # e_local_graphs -> [B, 128, 2000, 9] 
        e_local_context = self.aggregator1(e_local_graphs)  # e_local_context -> [B, 128, 2000, 1]
        # e_gl_context = self.ltr1(self.trans_shape(x, "in"), self.trans_shape(e_local_context, "in"))

        gl_context = self.ltr1(self.trans_shape(i_local_context, "in"), self.trans_shape(e_local_context, "in"), motion)  # gl_context -> [2, 2000, 128]

        gl_context = self.trans_shape(gl_context, "out")  # global & local context  # gl_context -> [2, 128, 2000, 1]
        # fusion
        # gl_context = self.fusion1(torch.cat([i_gl_context, e_gl_context], dim=1))
        w0 = self.linear_0(gl_context).view(B, -1)  # w0 -> [2, 2000]

        # implict branch
        i_local_graphs = self.pool2(gl_context)  # output -> [B, 128, 500, 1]
        i_local_graphs = self.oafilter2(i_local_graphs)
        i_local_context = self.unpool2(gl_context, i_local_graphs)
        # i_gl_context = self.ltr2(self.trans_shape(gl_context, "in"), self.trans_shape(i_local_context, "in"))
        # i_gl_context = self.trans_shape(i_gl_context, "out")  # global & local context
        # explict branch
        e_local_graphs = get_graph_feature(gl_context, k=self.k_num)
        e_local_graphs = self.gab2(e_local_graphs)
        e_local_context = self.aggregator2(e_local_graphs)  
        # e_gl_context = self.ltr2(self.trans_shape(gl_context, "in"), self.trans_shape(e_local_context, "in"))

        gl_context = self.ltr2(self.trans_shape(i_local_context, "in"), self.trans_shape(e_local_context, "in"), motion)

        gl_context = self.trans_shape(gl_context, "out")
        # fusion
        # gl_context = self.fusion2(torch.cat([i_gl_context, e_gl_context], dim=1))
        # gl_context = self.embed_1(gl_context)
        w1 = self.linear_1(gl_context).view(B, -1)

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)  # w1_ds -> [2, 2000], w1 的降序
            w1_ds = w1_ds[:, :int(N*self.sr)]  # w1_ds -> [2, 1000], 取前 1000 个
            x_ds, y_ds, w0_ds = self.down_sampling(initial_x, y, w0, indices, None, self.predict)  # x_ds -> [2, 1, 1000, 4]; y_ds -> [2, 1000], w0_ds -> [2, 1000]
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)  # w1_ds -> [2, 1000], w1 的降序
            w1_ds = w1_ds[:, :int(N*self.sr)]  # w1_ds -> [2, 500], 取前 1000 个
            x_ds, y_ds, w0_ds, out = self.down_sampling(initial_x, y, w0, indices, gl_context, self.predict)  # x_ds -> [2, 1, 500, 4]; y_ds -> [2, 500], w0_ds -> [2, 500], out -> [2, 128, 500, 1]
            out = self.embed_2(out)  # out -> [2, 128, 500, 1]
            w2 = self.linear_2(out)  # out -> [2, 2, 500, 1]
            e_hat = weighted_8points(x_ds, w2)  # e_hat -> [2, 9]

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat

class CLNet(nn.Module):
    def __init__(self, config):
        super(CLNet, self).__init__()

        self.ds_0 = DS_Block(config, initial=True, predict=False, out_channel=128, k_num=9, sampling_rate=0.5)  # sampling_rate=0.5
        self.ds_1 = DS_Block(config, initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=0.5)

    def forward(self, x, y):
        # x[32,1,2000,4], y[32,2000]
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0 = self.ds_0(x, y)  # x1 -> [2, 1, 1000, 4], y1 -> [2, 1000], ws0 -> [[2, 2000], [2, 2000]], w_ds0 -> [[2, 1000], [2, 1000]]

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)  # w_ds0[0] -> [2, 1, 1000, 1]
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)  # w_ds0[1] -> [2, 1, 1000, 1]
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)  # x_ -> [2, 1, 1000, 6]

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1)  # x2 -> [2, 1, 500, 4]; y2 -> [2, 500]; ws1 -> [[2, 1000], [2, 1000], [2, 500]]

        if e_hat == None:
            return ws0 + ws1, [y, y, y1, y1, y2], [None], None, [x, x, x1, x1, x2]

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)  # y_hat -> [2, 2000]

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat, [x, x, x1, x1, x2]