import torch
import torch.nn as nn

from utils.myaml import load_config
import torch.nn.functional as F
import time
import math
from models.tcn_model import TemporalBlock, TemporalConvNet
import numpy as np
from .DCN_model import Conv2dWithConstraint
from einops import rearrange
from .EmT import GraphEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.adj_matrix = adj_matrix
        self.out_features = out_features

        # 初始化权重
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: (batch_size, num_channels, sequence_len, in_features)
        batch_size, num_channels, seq_len, _ = x.size()

        # Reshape x to (batch_size * num_channels, sequence_len, in_features)
        x_reshaped = x.reshape(batch_size * num_channels, seq_len, -1)

        # GCN forward
        D = torch.diag(torch.sum(self.adj_matrix, dim=1))
        D_inv_sqrt = torch.inverse(torch.sqrt(D))
        normalized_adj = torch.matmul(torch.matmul(D_inv_sqrt, self.adj_matrix), D_inv_sqrt)

        # Apply GCN
        support = torch.matmul(x_reshaped, self.weight)  # (batch_size * num_channels, seq_len, out_features)
        # 假设 support 的形状为 (batch_size * num_channels, seq_len, out_features)
        # 我们需要调整它的形状
        # 将 support 形状调整为 (batch_size, num_channels, seq_len, out_features)
        support = support.view(batch_size, num_channels, seq_len, self.out_features)

        # 现在将 support 的形状调整为 (batch_size, num_channels, seq_len, out_features)
        # 可以使用以下的方式进行矩阵乘法
        # 由于 normalized_adj 是 (num_channels, num_channels)，我们需要对 support 进行合并和转置操作

        # 转置 support，使其变为 (batch_size, seq_len, num_channels, out_features)
        support = support.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_channels, out_features)

        # 现在进行矩阵乘法
        # 将 normalized_adj 乘到 support 上，得到 (batch_size, seq_len, num_channels, out_features)
        output = torch.matmul(normalized_adj, support)  # (batch_size, seq_len, num_channels, out_features)

        # 需要调整 output 的形状为 (batch_size, seq_len, out_features)
        output = output.permute(0, 2, 1, 3)  # (batch_size, num_channels, seq_len, out_features)
        output = output.contiguous()
        return F.relu(output)


class GaitGraph(nn.Module):
    """
    Modified GaitGraph with GraphNet block (GCN).
    """

    def __init__(self, config, init_adj=True):
        super(GaitGraph, self).__init__()
        self.ts = config.eegnet.eeg.time_step
        self.config = config.eegnet
        self.drop_out = self.config.dropout
        self.mask = self.config.mask
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((self.config.blk1_kernel // 2 - 1, self.config.blk1_kernel // 2, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=self.config.F1,
                kernel_size=(1, self.config.blk1_kernel),
                bias=False
            ),
            nn.BatchNorm2d(self.config.F1)
        )
        self.gcn = GCNLayer(self.config.F1, self.config.F1, self.config.adj_matrix)
        graph2token = 'Flatten'
        encoder_type = 'GCN'
        layers_graph = [1, 2]
        K = 4

        if init_adj:

            repeated_adj_matrix = self.config.adj_matrix.unsqueeze(0).repeat(len(layers_graph), 1, 1)
            self.adjs = nn.Parameter(repeated_adj_matrix, requires_grad=True)
        else:
            self.adjs = nn.Parameter(torch.FloatTensor(len(layers_graph), self.config.num_chan_eeg, self.config.num_chan_eeg), requires_grad=True)
        nn.init.xavier_uniform_(self.adjs)

        self.GE1 = GraphEncoder(
            num_layers=layers_graph[0], num_node=self.config.num_chan_eeg, in_features=self.config.F1,
            out_features=self.config.F1, K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.GE2 = GraphEncoder(
            num_layers=layers_graph[1], num_node=self.config.num_chan_eeg, in_features=self.config.F1,
            out_features=self.config.F1, K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.block_2_gcn = nn.Sequential(

            nn.Conv2d(
                in_channels=self.config.F1,  # input shape (8, C, T)
                out_channels=self.config.D * self.config.F1,  # num_filters
                kernel_size=(self.config.num_chan_eeg, 1),  # filter size
                groups=self.config.F1,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(self.config.D * self.config.F1),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            # Variance_layer(2, (1,4)),  # output shape (16, 1, T//4)
            # nn.Linear(self.config.eeg.time_step, int(self.config.eeg.time_step / 4)),
            nn.Dropout(self.drop_out * 2)  # output shape (16, 1, T//4)
        )
        self.block_2 = nn.Sequential(

            nn.Conv2d(
                in_channels=self.config.F1,  # input shape (8, C, T)
                out_channels=self.config.D * self.config.F1,  # num_filters
                kernel_size=(self.config.num_chan_eeg, 1),  # filter size
                groups=self.config.F1,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(self.config.D * self.config.F1),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            # Variance_layer(2, (1,4)),  # output shape (16, 1, T//4)
            # nn.Linear(self.config.eeg.time_step, int(self.config.eeg.time_step / 4)),
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )
        # GCN Layer instead of block_2
        # self.gcn = GCNConv(self.config.F1, self.config.D * self.config.F1)
        self.temporal_atten = nn.MultiheadAttention(self.config.F2, self.config.F2)
        self.spatial_atten = nn.MultiheadAttention(self.ts // 4, self.ts // 4)

        self.block_5 = nn.Sequential(
            nn.ZeroPad2d(((self.config.blk5_kernel + 1) // 2 - 1, (self.config.blk5_kernel + 1) // 2, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, 32, T//4)
                out_channels=self.config.F2 * 2,  # num_filters
                kernel_size=(self.config.F2 * 2, self.config.blk5_kernel),  # filter size
                # groups=self.config.F2,
                bias=False
            ),  # output shape (32, 1, T//4)

            nn.BatchNorm2d(self.config.F2 * 2),  # output shape (32, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (32, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Sequential(
            # nn.Linear(int(self.config.F2 * 2 * int((self.config.eeg.time_step - self.mask) / 16)), self.config.num_chan_kin)
            nn.Conv2d(
                in_channels=1,  # input shape (1, 32, T//4)
                out_channels=self.config.num_chan_kin,  # num_filters
                kernel_size=(self.config.F2 * 2, int((self.config.eeg.time_step - self.mask) // 32)),  # filter size
                # groups=self.config.F2,
                bias=False
            ),  # output shape (32, 1, T//4)
        )

    def forward(self, x):
        if self.config.mask != 0:
            x = x[:, :, :, :-1 * self.config.mask]

        x = self.block_1(x)## output shape (B, 8, C, T)
        self.out1 = x
        self.out1.retain_grad()
        basic_gcn = 0
        mvp_gcn = 1
        original = 1 - basic_gcn - mvp_gcn
        # num_nodes = self.config.adj_matrix
        # adj = F.relu(self.adjs + self.adjs.transpose(2, 1))
        # return adj
        if basic_gcn:
            x_gcn = self.out1.permute(0, 2, 3, 1)  # 交换 channel 和 kernel_size 维度
            x_gcn = self.gcn(x_gcn)
            x_gcn = x_gcn.permute(0, 3, 1, 2)  # output shape (F1*D, 59, 180)
        elif mvp_gcn:

            x_gcn = self.out1.permute(0, 3, 2, 1)#(B, T, C, 8)
            B, T, C, _ = x_gcn.shape
            x_reshaped = rearrange(x_gcn, 'b t c f -> (b t) c f')
            adjs = self.get_adj()
            # x_reshaped = x_gcn.reshape(B * T, C, self.config.F1)
            x_gcn1 = self.GE1(x_reshaped, adjs[0])
            x_gcn1 = x_gcn1.reshape(B, T * C * self.config.F1)
            x_gcn1 = x_gcn1.reshape(B, T, C, self.config.F1)
            x_gcn1 = x_gcn1.permute(0, 3, 2, 1)#(B, F1, C, T)

            x_gcn2 = self.GE2(x_reshaped, adjs[1])#(B*T, C*F1)
            x_gcn2 = x_gcn2.reshape(B, T * C * self.config.F1)
            x_gcn2 = x_gcn2.reshape(B, T, C, self.config.F1)
            x_gcn2 = x_gcn2.permute(0, 3, 2, 1)#(B, F1/2, C, T)

            x_gcn = x_gcn1 + x_gcn2
            # x_gcn = x_gcn1

        if not original:
            x_gcn = self.block_2_gcn(x_gcn) # output shape (F1*D, 1, 180)
            x = self.block_2(self.out1)
            x = x + x_gcn
            #

            # x = self.block_2(self.out1 + x_gcn)
        else:
            x = self.block_2(x)
        x = torch.squeeze(x, 2)  # output shape (B,16, T//4)
        tmp_x = x
        x = x.permute(0, 2, 1)#(B, T//4, 16)
        x_tempo, _ = self.temporal_atten(x, x, x)  # output shape (T, F2)
        x_tempo = x_tempo.permute(0, 2, 1)  # output shape (F2, T)
        x = torch.cat((tmp_x, x_tempo), dim=1)  # output shape (F2 * 2, T//4)
        x = torch.unsqueeze(x, 1)  # output shape (1, F2 * 2, T//4)
        x = self.block_5(x)
        self.feature = x
        x = x.permute(0, 2, 1, 3)

        x = self.out(x)
        x = x.squeeze()
        # x = self.final(x)
        return x
    def get_adj(self, self_loop=True):
        # self.adjs : n, node, node
        num_nodes = self.adjs.shape[-1]
        adj = F.relu(self.adjs + self.adjs.transpose(2, 1))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to('cuda:0')
        return adj
class GaitGraph_tmp(nn.Module):
    """
    Modified GaitGraph with GraphNet block (GCN).
    """

    def __init__(self, config, init_adj=True):
        super(GaitGraph_tmp, self).__init__()
        self.config = config.eegnet
        nChan = self.config.num_chan_eeg
        nTime = self.config.eeg.time_step
        pool_width = 3
        poolSize = {
            "LocalLayers": [(1, pool_width), (1, pool_width), (1, pool_width)],
            "GlobalLayers": (1, pool_width),
        }
        kernel_width = 10
        localKernalSize = {
            "LocalLayers": [(1, kernel_width), (1, kernel_width), (1, kernel_width)],
            "GlobalLayers": (1, kernel_width),
        }
        nClass = config.eegnet.num_chan_kin
        dropoutP = 0.5
        nFilt_FirstLayer = 25
        self.nFilt_FirstLayer = nFilt_FirstLayer
        nFiltLaterLayer = [25, 50, 100, 200]

        self.firstLayer = nn.Sequential(

            nn.ZeroPad2d((localKernalSize["LocalLayers"][0][1] // 2 - 1, localKernalSize["LocalLayers"][0][1] // 2, 0, 0)),
            Conv2dWithConstraint(
                1, nFilt_FirstLayer, localKernalSize["LocalLayers"][0], padding=0, max_norm=2),

        )

        graph2token = 'Flatten'
        encoder_type = 'GCN'
        layers_graph = [1, 2]
        K = 4
        if init_adj:

            repeated_adj_matrix = self.config.adj_matrix.unsqueeze(0).repeat(len(layers_graph), 1, 1)
            self.adjs = nn.Parameter(repeated_adj_matrix, requires_grad=True)
        else:
            self.adjs = nn.Parameter(torch.FloatTensor(len(layers_graph), self.config.num_chan_eeg, self.config.num_chan_eeg), requires_grad=True)
        nn.init.xavier_uniform_(self.adjs)

        self.GE1 = GraphEncoder(
            num_layers=layers_graph[0], num_node=self.config.num_chan_eeg, in_features=nFilt_FirstLayer,
            out_features=nFilt_FirstLayer, K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.GE2 = GraphEncoder(
            num_layers=layers_graph[1], num_node=self.config.num_chan_eeg, in_features=nFilt_FirstLayer,
            out_features=nFilt_FirstLayer, K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.spatialLayer = nn.Sequential(
            Conv2dWithConstraint(25, 25, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(nFilt_FirstLayer),
            nn.ELU(),
            nn.MaxPool2d(poolSize["LocalLayers"][0], stride=poolSize["LocalLayers"][0]),
        )
        # middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, localKernalSize)
        #     for inF, outF in zip(nFiltLaterLayer[:-1], nFiltLaterLayer[1:-1])])
        self.middleLayers = nn.Sequential(
            *[
                self.convBlock(inF, outF, dropoutP, kernalS, poolS)
                for inF, outF, kernalS, poolS in zip(
                    nFiltLaterLayer[:-1],
                    nFiltLaterLayer[1:-1],
                    localKernalSize["LocalLayers"][1:],
                    poolSize["LocalLayers"][1:],
                )
            ]
        )
        self.tmplayers = nn.Sequential(
            *[
                self.convBlock(inF, outF, dropoutP, kernalS, poolS)
                for inF, outF, kernalS, poolS in zip(
                    nFiltLaterLayer[:-2],
                    nFiltLaterLayer[1:-2],
                    localKernalSize["LocalLayers"][1:],
                    poolSize["LocalLayers"][1:],
                )
            ]
        )
        self.tmpdropout = nn.Dropout(p=dropoutP)
        self.firstGlobalLayer = self.convBlock(
            nFiltLaterLayer[-2],
            nFiltLaterLayer[-1],
            dropoutP,
            localKernalSize["GlobalLayers"],
            poolSize["GlobalLayers"],
        )

        self.allButLastLayers = nn.Sequential(
            self.firstLayer, self.middleLayers, self.firstGlobalLayer
        )

        # self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, nTime // 81))

        # self.weight_keys =[['allButLastLayers.0.0.weight','allButLastLayers.0.0.bias','allButLastLayers.0.1.weight'],
        #                     ['allButLastLayers.1.0.1.weight'],
        #                     ['allButLastLayers.1.1.1.weight'],
        #                     ['allButLastLayers.1.2.1.weight'],
        #                     ['lastLayer.0.weight', 'lastLayer.0.bias']
        #                     ]

        self.weight_keys = [
            [
                "allButLastLayers.0.0.weight",
                "allButLastLayers.0.0.bias",
                "allButLastLayers.0.1.weight",
            ],
            ["allButLastLayers.1.0.1.weight"],
            ["allButLastLayers.1.1.1.weight"],
            ["allButLastLayers.2.1.weight"],
            ["lastLayer.0.weight", "lastLayer.0.bias"],
        ]
        self.temporalGlobalLayer = nn.MultiheadAttention(nTime // 81, nTime // 81)
        self.temporalLastLayer = Conv2dWithConstraint(nFiltLaterLayer[-1], 6, (1, nTime // 81 * 2), padding=0, max_norm=2)

    def get_adj(self, self_loop=True):
        # self.adjs : n, node, node
        num_nodes = self.adjs.shape[-1]
        adj = F.relu(self.adjs + self.adjs.transpose(2, 1))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to('cuda:0')
        return adj

    def convBlock(self, inF, outF, dropoutP, kernalSize, poolSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(
                inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs
            ),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d(poolSize, stride=poolSize),
        )


    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(

            # nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs),
            # nn.LogSoftmax(dim=1),
        )

    def calculateOutSize(self, model, nChan, nTime):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def forward(self, x):
        # x = self.allButLastLayers(x)
        x = self.firstLayer(x) # (B, 25, C, T)
        x_gcn = x.permute(0, 3, 2, 1)  # (B, T, C, 8)
        B, T, C, _ = x_gcn.shape
        x_reshaped = rearrange(x_gcn, 'b t c f -> (b t) c f')
        adjs = self.get_adj()
        # x_reshaped = x_gcn.reshape(B * T, C, self.config.F1)
        x_gcn1 = self.GE1(x_reshaped, adjs[0])
        x_gcn1 = x_gcn1.reshape(B, T * C * self.nFilt_FirstLayer)
        x_gcn1 = x_gcn1.reshape(B, T, C, self.nFilt_FirstLayer)
        x_gcn1 = x_gcn1.permute(0, 3, 2, 1)  # (B, F1, C, T)

        x_gcn2 = self.GE2(x_reshaped, adjs[1])  # (B*T, C*F1)
        x_gcn2 = x_gcn2.reshape(B, T * C * self.nFilt_FirstLayer)
        x_gcn2 = x_gcn2.reshape(B, T, C, self.nFilt_FirstLayer)
        x_gcn2 = x_gcn2.permute(0, 3, 2, 1)  # (B, F1/2, C, T)

        x = (x_gcn1 + x_gcn2) / 2 + x# (B, 25, C, T)
        x = self.spatialLayer(x)# (B, 25, 1, T // 3)

        # #V1
        # x = self.tmplayers(x)# (B, 50, 1, T // 9)
        # x = torch.squeeze(x, dim=2)
        # # x = x.permute(0, 2, 1)
        # x = self.tmpdropout(x)
        # x_tempo, _ = self.temporalGlobalLayer(x, x, x)# (B, 50, T // 9)
        # x = self.tmpdropout(x)
        # x = torch.unsqueeze(torch.cat((x, x_tempo), dim=2), dim=1)# (B, 1, 50, T *  2 // 9)
        # x = self.temporalLastLayer(x)

        # #V2
        # x = self.middleLayers(x)# (B, 100, 1, T // 27)
        # x = torch.squeeze(x, dim=2)
        # # x = x.permute(0, 2, 1)
        # x = self.tmpdropout(x)
        # x_tempo, _ = self.temporalGlobalLayer(x, x, x)# (B, 100, T // 27)
        # x = self.tmpdropout(x)
        # x = torch.unsqueeze(torch.cat((x, x_tempo), dim=2), dim=1)# (B, 1, 50, T *  2 // 27)
        # x = self.temporalLastLayer(x)

        #V3
        x = self.middleLayers(x)# (B, 100, 1, T // 27)
        x = self.firstGlobalLayer(x)  # (B, 200, 1, T // 81)
        x = torch.squeeze(x, dim=2)
        # x = x.permute(0, 2, 1)
        x = self.tmpdropout(x)
        x_tempo, _ = self.temporalGlobalLayer(x, x, x)# (B, 200, T // 81)
        x = self.tmpdropout(x)
        x = torch.unsqueeze(torch.cat((x, x_tempo), dim=2), dim=1)# (B, 1, 200, T *  2 // 81)

        x = x.permute(0, 2, 1, 3)
        x = self.temporalLastLayer(x)

        #Original
        # x = self.middleLayers(x)# (B, 100, 1, T // 27)
        # x = self.firstGlobalLayer(x)# (B, 200, 1, T // 81)
        # x = self.lastLayer(x)

        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x

class GaitCNN_tmp(nn.Module):
    """
    Modified GaitGraph with GraphNet block (GCN).
    """

    def __init__(self, config, init_adj=True):
        super(GaitCNN_tmp, self).__init__()
        self.config = config.eegnet
        nChan = self.config.num_chan_eeg
        nTime = self.config.eeg.time_step
        pool_width = 3
        poolSize = {
            "LocalLayers": [(1, pool_width), (1, pool_width), (1, pool_width)],
            "GlobalLayers": (1, pool_width),
        }
        kernel_width = 10
        localKernalSize = {
            "LocalLayers": [(1, kernel_width), (1, kernel_width), (1, kernel_width)],
            "GlobalLayers": (1, kernel_width),
        }
        nClass = config.eegnet.num_chan_kin
        dropoutP = 0.5
        nFilt_FirstLayer = 25
        self.nFilt_FirstLayer = nFilt_FirstLayer
        nFiltLaterLayer = [25, 50, 100, 200]

        self.firstLayer = nn.Sequential(

            nn.ZeroPad2d((localKernalSize["LocalLayers"][0][1] // 2 - 1, localKernalSize["LocalLayers"][0][1] // 2, 0, 0)),
            Conv2dWithConstraint(
                1, nFilt_FirstLayer, localKernalSize["LocalLayers"][0], padding=0, max_norm=2),

        )

        graph2token = 'Flatten'
        encoder_type = 'GCN'
        layers_graph = [1, 2]
        K = 4
        if init_adj:

            repeated_adj_matrix = self.config.adj_matrix.unsqueeze(0).repeat(len(layers_graph), 1, 1)
            self.adjs = nn.Parameter(repeated_adj_matrix, requires_grad=True)
        else:
            self.adjs = nn.Parameter(torch.FloatTensor(len(layers_graph), self.config.num_chan_eeg, self.config.num_chan_eeg), requires_grad=True)
        nn.init.xavier_uniform_(self.adjs)

        self.GE1 = GraphEncoder(
            num_layers=layers_graph[0], num_node=self.config.num_chan_eeg, in_features=nFilt_FirstLayer,
            out_features=nFilt_FirstLayer, K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.GE2 = GraphEncoder(
            num_layers=layers_graph[1], num_node=self.config.num_chan_eeg, in_features=nFilt_FirstLayer,
            out_features=nFilt_FirstLayer, K=K, graph2token=graph2token, encoder_type=encoder_type
        )
        self.spatialLayer = nn.Sequential(
            Conv2dWithConstraint(25, 25, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(nFilt_FirstLayer),
            nn.ELU(),
            nn.MaxPool2d(poolSize["LocalLayers"][0], stride=poolSize["LocalLayers"][0]),
        )
        # middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, localKernalSize)
        #     for inF, outF in zip(nFiltLaterLayer[:-1], nFiltLaterLayer[1:-1])])
        self.middleLayers = nn.Sequential(
            *[
                self.convBlock(inF, outF, dropoutP, kernalS, poolS)
                for inF, outF, kernalS, poolS in zip(
                    nFiltLaterLayer[:-1],
                    nFiltLaterLayer[1:-1],
                    localKernalSize["LocalLayers"][1:],
                    poolSize["LocalLayers"][1:],
                )
            ]
        )
        self.tmplayers = nn.Sequential(
            *[
                self.convBlock(inF, outF, dropoutP, kernalS, poolS)
                for inF, outF, kernalS, poolS in zip(
                    nFiltLaterLayer[:-2],
                    nFiltLaterLayer[1:-2],
                    localKernalSize["LocalLayers"][1:],
                    poolSize["LocalLayers"][1:],
                )
            ]
        )
        self.tmpdropout = nn.Dropout(p=dropoutP)
        self.firstGlobalLayer = self.convBlock(
            nFiltLaterLayer[-2],
            nFiltLaterLayer[-1],
            dropoutP,
            localKernalSize["GlobalLayers"],
            poolSize["GlobalLayers"],
        )

        self.allButLastLayers = nn.Sequential(
            self.firstLayer, self.middleLayers, self.firstGlobalLayer
        )

        # self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, nTime // 81))

        # self.weight_keys =[['allButLastLayers.0.0.weight','allButLastLayers.0.0.bias','allButLastLayers.0.1.weight'],
        #                     ['allButLastLayers.1.0.1.weight'],
        #                     ['allButLastLayers.1.1.1.weight'],
        #                     ['allButLastLayers.1.2.1.weight'],
        #                     ['lastLayer.0.weight', 'lastLayer.0.bias']
        #                     ]

        self.weight_keys = [
            [
                "allButLastLayers.0.0.weight",
                "allButLastLayers.0.0.bias",
                "allButLastLayers.0.1.weight",
            ],
            ["allButLastLayers.1.0.1.weight"],
            ["allButLastLayers.1.1.1.weight"],
            ["allButLastLayers.2.1.weight"],
            ["lastLayer.0.weight", "lastLayer.0.bias"],
        ]
        self.temporalGlobalLayer = nn.MultiheadAttention(nTime // 81, nTime // 81)
        self.temporalLastLayer = Conv2dWithConstraint(nFiltLaterLayer[-1], 6, (1, nTime // 81 * 2), padding=0, max_norm=2)

    def get_adj(self, self_loop=True):
        # self.adjs : n, node, node
        num_nodes = self.adjs.shape[-1]
        adj = F.relu(self.adjs + self.adjs.transpose(2, 1))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to('cuda:0')
        return adj

    def convBlock(self, inF, outF, dropoutP, kernalSize, poolSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(
                inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs
            ),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d(poolSize, stride=poolSize),
        )


    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(

            # nn.ZeroPad2d((kernalSize[1] // 2 - 1, kernalSize[1] // 2, 0, 0)),
            Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs),
            # nn.LogSoftmax(dim=1),
        )

    def calculateOutSize(self, model, nChan, nTime):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def forward(self, x):
        # x = self.allButLastLayers(x)
        x = self.firstLayer(x) # (B, 25, C, T)
        # x_gcn = x.permute(0, 3, 2, 1)  # (B, T, C, 8)
        # B, T, C, _ = x_gcn.shape
        # x_reshaped = rearrange(x_gcn, 'b t c f -> (b t) c f')
        # adjs = self.get_adj()
        # # x_reshaped = x_gcn.reshape(B * T, C, self.config.F1)
        # x_gcn1 = self.GE1(x_reshaped, adjs[0])
        # x_gcn1 = x_gcn1.reshape(B, T * C * self.nFilt_FirstLayer)
        # x_gcn1 = x_gcn1.reshape(B, T, C, self.nFilt_FirstLayer)
        # x_gcn1 = x_gcn1.permute(0, 3, 2, 1)  # (B, F1, C, T)
        #
        # x_gcn2 = self.GE2(x_reshaped, adjs[1])  # (B*T, C*F1)
        # x_gcn2 = x_gcn2.reshape(B, T * C * self.nFilt_FirstLayer)
        # x_gcn2 = x_gcn2.reshape(B, T, C, self.nFilt_FirstLayer)
        # x_gcn2 = x_gcn2.permute(0, 3, 2, 1)  # (B, F1/2, C, T)
        #
        # x = (x_gcn1 + x_gcn2) / 2 + x# (B, 25, C, T)
        x = self.spatialLayer(x)# (B, 25, 1, T // 3)

        # #V1
        # x = self.tmplayers(x)# (B, 50, 1, T // 9)
        # x = torch.squeeze(x, dim=2)
        # # x = x.permute(0, 2, 1)
        # x = self.tmpdropout(x)
        # x_tempo, _ = self.temporalGlobalLayer(x, x, x)# (B, 50, T // 9)
        # x = self.tmpdropout(x)
        # x = torch.unsqueeze(torch.cat((x, x_tempo), dim=2), dim=1)# (B, 1, 50, T *  2 // 9)
        # x = self.temporalLastLayer(x)

        # #V2
        # x = self.middleLayers(x)# (B, 100, 1, T // 27)
        # x = torch.squeeze(x, dim=2)
        # # x = x.permute(0, 2, 1)
        # x = self.tmpdropout(x)
        # x_tempo, _ = self.temporalGlobalLayer(x, x, x)# (B, 100, T // 27)
        # x = self.tmpdropout(x)
        # x = torch.unsqueeze(torch.cat((x, x_tempo), dim=2), dim=1)# (B, 1, 50, T *  2 // 27)
        # x = self.temporalLastLayer(x)

        #V3
        x = self.middleLayers(x)# (B, 100, 1, T // 27)
        x = self.firstGlobalLayer(x)  # (B, 200, 1, T // 81)
        x = torch.squeeze(x, dim=2)
        # x = x.permute(0, 2, 1)
        x = self.tmpdropout(x)
        x_tempo, _ = self.temporalGlobalLayer(x, x, x)# (B, 200, T // 81)
        x = self.tmpdropout(x)
        x = torch.unsqueeze(torch.cat((x, x_tempo), dim=2), dim=1)# (B, 1, 200, T *  2 // 81)

        x = x.permute(0, 2, 1, 3)
        x = self.temporalLastLayer(x)

        #Original
        # x = self.middleLayers(x)# (B, 100, 1, T // 27)
        # x = self.firstGlobalLayer(x)# (B, 200, 1, T // 81)
        # x = self.lastLayer(x)

        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x

if __name__ == "__main__":
    # 当脚本直接执行时，以下代码会运行
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(100, 1, 59, 200).to(device)
    # Channel names
    ch_names = [
        'Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
        'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2',
        'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
        'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
        'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4',
        'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2'
    ]

    # Neighbors as a dictionary
    channel_neighbors = {
        # order: up-down-left-right
        "Fp1": ["AF7", "AF3", "AFz"],
        "Fp2": ["AF4", "AF8", "AFz"],
        "AF7": ["Fp1", "AF3", "F7"],
        "AF3": ["Fp1", "F3", "AF7", "AFz"],
        "AFz": ["Fp1", 'Fp2', "Fz", "AF3", "AF4"],
        "AF4": ["Fp2", "F4", "AFz", "AF8"],
        "AF8": ["Fp2", "AF4", "F8"],
        "F7": ["AF7", "FT7", "F5"],
        "F5": ["AF7", "AF3", "FC5", "F7", "F3"],
        "F3": ["AF3", "FC3", "F5", "F1"],
        "F1": ["AF3", "AFz", "FC1", "F3", "Fz"],
        "Fz": ["AFz", "F1", "F2"],
        "F2": ["AFz", "AF4", "FC2", "Fz", "F4"],
        "F4": ["AF4", "FC4", "F2", "F6"],
        "F6": ["AF4", "AF8", "FC6", "F4", "F8"],
        "F8": ["AF8", "FT8", "F6"],
        "FT7": ["F7", "T7", "FC5"],
        "FC5": ["F5", "C5", "FT7", "FC3"],
        "FC3": ["F3", "C3", "FC5", "FC1"],
        "FC1": ["F1", "C1", "FC3"],
        "FC2": ["F2", "C2", "FC4"],
        "FC4": ["F4", "C4", "FC2", "FC6"],
        "FC6": ["F6", "C6", "FC4", "FT8"],
        "FT8": ["F8", "T8", "FC6"],
        "T7": ["FT7", "TP7", "C5"],
        "C5": ["FC5", "CP5", "T7", "C3"],
        "C3": ["FC3", "CP3", "C5", "C1"],
        "C1": ["FC1", "CP1", "C3", "Cz"],
        "Cz": ["CPz", "C1", "C2"],
        "C2": ["FC2", "CP2", "Cz", "C4"],
        "C4": ["FC4", "CP4", "C2", "CP6"],
        "C6": ["FC6", "CP6", "C4", "T8"],
        "T8": ["FT8", "TP8", "C6"],
        "TP7": ["T7", "P7", "CP5"],
        "CP5": ["C5", "P5", "TP7", "CP3"],
        "CP3": ["C3", "P3", "CP5", "CP1"],
        "CP1": ["C1", "P1", "CP3", "CPz"],
        "CPz": ["Cz", "Pz", "CP1", "CP2"],
        "CP2": ["C2", "P2", "CPz", "CP4"],
        "CP4": ["C4", "P4", "CP2", "CP6"],
        "CP6": ["C6", "P6", "CP6", "TP8"],
        "TP8": ["T8", "P8", "CP6"],
        "P7": ["TP7", "PO7", "P5"],
        "P5": ["CP5", "PO7", "PO3", "P7", "P3"],
        "P3": ["CP3", "PO3", "P5", "P1"],
        "P1": ["CP1", "PO3", "POz", "P3", "Pz"],
        "Pz": ["CPz", "POz", "P1", "P2"],
        "P2": ["CP2", "POz", "PO4", "Pz", "P4"],
        "P4": ["CP4", "PO4", "P2", "P6"],
        "P6": ["CP6", "PO4", "PO8", "P4", "P8"],
        "P8": ["TP8", "PO8", "P6"],
        "PO7": ["P7", "O1", "PO3"],
        "PO3": ["P3", "O1", "Oz", "PO7", "POz"],
        "POz": ["Pz", "Oz", "PO3", "PO4"],
        "PO4": ["P4", "Oz", "O2", "POz", "PO8"],
        "PO8": ["P8", "O2", "PO4"],
        "O1": ["PO7", "PO3", "Oz"],
        "Oz": ["POz", "O1", "O2"],
        "O2": ["PO8", "PO4", "Oz"],
    }

    # # Create edge_index
    # edges = []
    #
    # # Loop through each channel and its neighbors to create edges
    # for channel, neighbors in channel_neighbors.items():
    #     for neighbor in neighbors:
    #         edges.append((ch_names.index(channel), ch_names.index(neighbor)))
    #
    # # Convert edges to tensor and create edge_index
    # edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 初始化邻接矩阵
    num_channels = len(ch_names)
    adj_matrix = np.zeros((num_channels, num_channels))

    # 填充邻接矩阵
    for channel, neighbors in channel_neighbors.items():
        channel_index = ch_names.index(channel)
        for neighbor in neighbors:
            neighbor_index = ch_names.index(neighbor)
            adj_matrix[channel_index, neighbor_index] = 1
            adj_matrix[neighbor_index, channel_index] = 1  # 无向图，双向连接

    # 将邻接矩阵转换为tensor
    adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float).to(device)

    config = load_config('../config.yaml')
    config.eegnet.adj_matrix = adj_matrix_tensor
    model = GaitGraph(config).to(device)
    out = model(x)
    a = 0