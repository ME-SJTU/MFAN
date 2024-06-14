import torch
import torch.nn as nn
from torch.nn import functional as F


class AsymmetricConvolution(nn.Module):

    def __init__(self, in_dims=64, out_dims=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dims, out_dims, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_dims, out_dims, kernel_size=(1, 3), padding=(0, 1))

        self.shortcut = lambda x: x

        if in_dims != out_dims:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dims, out_dims, kernel_size=(1, 1), bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x, sparse_mask):
        shortcut = self.shortcut(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_ = self.activation(x2 + x1) * sparse_mask.unsqueeze(1)

        return x_ + shortcut


class SpatialFeatures(nn.Module):
    def __init__(self, in_dims=64, out_dim=1, conv_num=7):
        super(SpatialFeatures, self).__init__()

        self.dims = 2
        self.conv_num = conv_num

        self.conv = nn.ModuleList()
        for k in range(conv_num):
            self.conv.append(AsymmetricConvolution(in_dims=self.dims, out_dims=self.dims))

        self.conv_out = AsymmetricConvolution(in_dims=self.dims, out_dims=1)

        self.spatial_agg = nn.Sequential(
            nn.Conv2d(in_dims, out_dim, kernel_size=(1, 1), bias=False),
            nn.PReLU()
        )
        self.embedding = nn.Linear(in_dims, out_dim, bias=False)

        self.avg_pooling = nn.AdaptiveAvgPool2d((None, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((None, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, features, mask, threshold=0.5):

        agg_features01 = self.avg_pooling(features).permute(0, 3, 1, 2)
        agg_features02 = self.max_pooling(features).permute(0, 3, 1, 2)

        spatial_features = torch.cat((agg_features01, agg_features02), dim=1)
        # spatial_features = features.permute(0, 3, 1, 2)

        for k in range(self.conv_num):
            spatial_features = F.dropout(self.conv[k](spatial_features, mask), p=0.0)

        agg_features = self.conv_out(spatial_features, mask).permute(0, 2, 3, 1)

        spatial_scores = self.sigmoid(agg_features)

        sparse_features = spatial_scores * features

        return sparse_features


class ChannelFeatures(nn.Module):
    def __init__(self, embedding_dims=64, conv_num=7):
        super(ChannelFeatures, self).__init__()

        self.dims = embedding_dims
        self.conv_num = conv_num

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()

        for k in range(conv_num):
            self.conv1.append(
                nn.Sequential(
                    nn.Conv2d(self.dims, self.dims, kernel_size=(1, 1)),
                    nn.PReLU())
            )

            self.conv2.append(
                nn.Sequential(
                    nn.Conv2d(self.dims, self.dims, kernel_size=(1, 1)),
                    nn.PReLU())
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, features):

        agg_features = features.permute(0, 3, 1, 2)

        agg_features01 = self.max_pooling(agg_features)
        agg_features02 = self.avg_pooling(agg_features)

        for k in range(self.conv_num):
            agg_features01 = self.conv1[k](agg_features01) + agg_features01
            agg_features02 = self.conv1[k](agg_features02) + agg_features02

        channel_features01 = agg_features01.permute(0, 2, 3, 1)
        channel_features02 = agg_features02.permute(0, 2, 3, 1)

        channel_features = channel_features01 + channel_features02

        channel_scores = self.sigmoid(channel_features)

        attention_features = features * channel_scores

        return attention_features


class EdgeFeatureAttention(nn.Module):

    def __init__(self, in_dims=2, in_edge_dims=5, out_dims=32, conv_num=7):
        super(EdgeFeatureAttention, self).__init__()

        self.in_dims = in_dims
        self.embedding_dims = out_dims
        self.fusion_dims = 2 * out_dims

        self.embedding = nn.Linear(in_dims, self.embedding_dims)
        self.hn1_embedding = nn.Linear(self.embedding_dims, self.embedding_dims)
        self.hn2_embedding = nn.Linear(self.embedding_dims, self.embedding_dims)

        self.edge_embedding = nn.Sequential(
            nn.Linear(in_edge_dims, out_dims, bias=False),
        )

        self.fusion_embedding = nn.Sequential(
            nn.Linear(self.fusion_dims, self.fusion_dims, bias=False),
            nn.PReLU()
        )

        self.spatial_features = SpatialFeatures(self.fusion_dims, 1, conv_num)
        self.channel_features = ChannelFeatures(self.fusion_dims, 1)

        # self.scaled_factor = torch.sqrt(torch.Tensor([self.fusion_dims])).cuda()

        self.fusion_out = nn.Linear(self.fusion_dims, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, node, edge, mask):

        assert len(node.shape) == 3
        assert len(edge.shape) == 4

        node_embedding = self.embedding(node)
        edge_embedding = self.edge_embedding(edge)

        hn1 = self.hn1_embedding(node_embedding)
        hn2 = self.hn2_embedding(node_embedding)

        hn1 = hn1.repeat(1, 1, hn1.shape[-2]).view(
            hn1.shape[0], hn1.shape[1], -1, hn1.shape[-1])
        hn2 = hn2.repeat(1, hn2.shape[-2], 1).view(
            hn2.shape[0], hn2.shape[1], -1, hn2.shape[-1])

        hn = self.fusion_embedding(torch.cat((hn1 * hn2 * mask.unsqueeze(-1), edge_embedding), dim=-1))

        spatial_features = self.spatial_features(hn, mask)
        channel_features = self.channel_features(hn)

        fusion_features = self.fusion_out(spatial_features + channel_features).squeeze()

        # sparse_features = (torch.sum(mixing_features, dim=-1)) / self.scaled_factor
        # scale_factor = torch.sum(mixing_features, dim=-1).unsqueeze(-1)

        # attention = mixing_features / scale_factor

        # sparse_scores = self.sparse_scores(mixing_attention)
        # sparse_features = mixing_attention * sparse_scores

        zero_vec = -9e20 * torch.ones_like(fusion_features).cuda()

        attention = torch.where(fusion_features != 0, fusion_features, zero_vec)

        attention = self.softmax(attention)
        # print(attention)

        return attention


class SpatialTemporalSparseAttention(nn.Module):

    def __init__(self, spat_edge_dims=5, temp_edge_dims=4, embedding_dims=32, conv_num=5):
        super(SpatialTemporalSparseAttention, self).__init__()

        self.spat_sparse_attention = EdgeFeatureAttention(
            in_edge_dims=spat_edge_dims, out_dims=embedding_dims, conv_num=conv_num)

        self.temp_sparse_attention = EdgeFeatureAttention(
            in_edge_dims=temp_edge_dims, out_dims=embedding_dims, conv_num=conv_num)

    def forward(self, node, spat_edge, temp_edge, mask):

        spat_node = node.squeeze()
        temp_node = spat_node.permute(1, 0, 2)

        spat_edge = spat_edge.squeeze()
        temp_edge = temp_edge.squeeze()

        spat_mask = mask[0].squeeze()
        temp_mask = mask[1].squeeze()

        spat_sparse_attention = self.spat_sparse_attention(spat_node, spat_edge, spat_mask)
        temp_sparse_attention = self.temp_sparse_attention(temp_node, temp_edge, temp_mask)

        return spat_sparse_attention, temp_sparse_attention


class GraphAttentionNetwork(nn.Module):

    def __init__(self, in_dims=32, embedding_dims=32, dropout=0):
        super(GraphAttentionNetwork, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()

        self.dropout = dropout

    def forward(self, node, attention):

        gat_features = torch.matmul(attention, node)
        gat_features = F.dropout(gat_features, p=self.dropout)

        return gat_features


class SpatialTemporalGAT(nn.Module):

    def __init__(self, in_dims=2, embedding_dims=32, gat_num=1):
        super(SpatialTemporalGAT, self).__init__()

        self.spatial_temporal_gat = nn.ModuleList()
        self.temporal_spatial_gat = nn.ModuleList()

        self.spat_embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.temp_embedding = nn.Linear(in_dims, embedding_dims, bias=False)

        self.gat_num = gat_num
        for k in range(gat_num):
            self.spatial_temporal_gat.append(GraphAttentionNetwork(embedding_dims, embedding_dims))
            self.temporal_spatial_gat.append(GraphAttentionNetwork(embedding_dims, embedding_dims))

        self.spatial_relu = nn.PReLU()
        self.temporal_relu = nn.PReLU()

    def forward(self, node, spat_attention, temp_attention):

        assert len(node.shape) == 4
        assert len(spat_attention.shape) == 3
        assert len(temp_attention.shape) == 3

        spat_node = node.squeeze()
        temp_node = spat_node.permute(1, 0, 2)

        spatial_features = self.spat_embedding(spat_node)
        temporal_features = self.temp_embedding(temp_node)

        for k in range(self.gat_num):
            spatial_features = self.spatial_relu(self.spatial_temporal_gat[k](spatial_features, spat_attention)
                                                 + spatial_features)
            temporal_features = self.temporal_relu(self.temporal_spatial_gat[k](temporal_features, temp_attention)
                                                   + temporal_features)

        spatial_features = spatial_features.permute(1, 0, 2)  # [num_node num_heads seq_len feat_dims]

        return spatial_features, temporal_features


class PredictionModel(nn.Module):

    def __init__(self, embedding_dims=32, obs_len=8, pred_len=12,
                 num_tcn=5, out_dims=5, dropout=0, conv_num=5, gat_num=1):
        super(PredictionModel, self).__init__()

        self.num_tcn = num_tcn
        self.dropout = dropout

        self.spat_temp_attention = SpatialTemporalSparseAttention(embedding_dims=embedding_dims, conv_num=conv_num)

        self.spat_temp_gat = SpatialTemporalGAT(embedding_dims=embedding_dims, gat_num=gat_num)

        self.st_fusion = nn.Conv2d(obs_len, obs_len, kernel_size=(1, 1))

        self.tcn = nn.ModuleList()
        self.tcn.append(nn.Sequential(
            nn.Conv2d(obs_len, pred_len, kernel_size=(3, 3), padding=(1, 1)),
            nn.PReLU()))
        for k in range(1, num_tcn):
            self.tcn.append(nn.Sequential(
                nn.Conv2d(pred_len, pred_len, kernel_size=(3, 3), padding=(1, 1)),
                nn.PReLU()))

        self.output = nn.Linear(embedding_dims, out_dims)

    def forward(self, node, spat_edge, temp_edge, mask):

        node = node[:, :, :, 1:]
        spat_edge = spat_edge.squeeze()
        temp_edge = temp_edge.squeeze()

        st_attention, ts_attention = self.spat_temp_attention(node, spat_edge, temp_edge, mask)

        st_features, ts_features = self.spat_temp_gat(node, st_attention, ts_attention)

        st_features = self.st_fusion(st_features.unsqueeze(2))
        gat_features = st_features + ts_features.unsqueeze(2)

        features = self.tcn[0](gat_features)
        for k in range(1, self.num_tcn):
            features = F.dropout(self.tcn[k](features) + features, p=self.dropout)

        prediction = self.output(features.squeeze()).permute(1, 0, 2)

        return prediction.contiguous()



