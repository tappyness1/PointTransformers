import torch
import torch.nn as nn
from src.model.PTv2.ptv2_utils import PointTransformerV2Block, TransitionDownBlock, TransitionUpBlock

class PTV2Classifier(nn.Module):

    def __init__(self, n_classes = 10, in_dim = 6):
        super().__init__()
        self.linear = nn.Linear(in_dim, 48)
        self.ptb_0 = PointTransformerV2Block(in_dim = 48, out_dim = 48)

        self.tdb_1 = TransitionDownBlock(in_dim = 48, out_dim = 96, grid_size = [0.06] * 3)
        self.ptb_1 = PointTransformerV2Block(in_dim = 96, out_dim = 96, K = 4)

        self.tdb_2 = TransitionDownBlock(in_dim = 96, out_dim = 192, grid_size = [0.13] * 3)
        self.ptb_2 = PointTransformerV2Block(in_dim = 192, out_dim = 192, K = 2)

        self.tdb_3 = TransitionDownBlock(in_dim = 192, out_dim = 384, grid_size = [0.325] * 3)
        self.ptb_3 = PointTransformerV2Block(in_dim = 384, out_dim = 384, K = 1)

        self.tdb_4 = TransitionDownBlock(in_dim = 384, out_dim = 512, grid_size = [0.8125] * 3)
        self.ptb_4 = PointTransformerV2Block(in_dim = 512, out_dim = 512, K = 1)
        
        # self.avg_pool = nn.AvgPool1d(1)
        self.mlp = nn.Linear(512, n_classes)

    def forward(self, points):

        points_xyz, points_features = points[:, :, :3], points
        out = self.linear(points_features)
        out_xyz, out_features = self.ptb_0(points_xyz, out)

        out_xyz, out_features = self.tdb_1(out_xyz, out_features)
        out_xyz, out_features = self.ptb_1(out_xyz, out_features)

        out_xyz, out_features = self.tdb_2(out_xyz, out_features)
        out_xyz, out_features = self.ptb_2(out_xyz, out_features)

        out_xyz, out_features = self.tdb_3(out_xyz, out_features)
        out_xyz, out_features = self.ptb_3(out_xyz, out_features)

        out_xyz, out_features = self.tdb_4(out_xyz, out_features)
        out_xyz, out_features = self.ptb_4(out_xyz, out_features)

        # out = self.avg_pool(out_features.permute(0, 2, 1))
        out = torch.mean(out_features, dim=1) # average pooling here because we don't how many L is left

        out = self.mlp(out.squeeze(-1))

        return out

class PTV2Segmentation(nn.Module):
    
    def __init__(self, n_classes = 10, in_dim = 6):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, 48)
        self.ptb_0 = PointTransformerV2Block(in_dim = 48, out_dim = 48)

        self.tdb_1 = TransitionDownBlock(in_dim = 48, out_dim = 96, grid_size = [0.06] * 3)
        self.ptb_1 = PointTransformerV2Block(in_dim = 96, out_dim = 96, K = 4)

        self.tdb_2 = TransitionDownBlock(in_dim = 96, out_dim = 192, grid_size = [0.13] * 3)
        self.ptb_2 = PointTransformerV2Block(in_dim = 192, out_dim = 192, K = 2)

        self.tdb_3 = TransitionDownBlock(in_dim = 192, out_dim = 384, grid_size = [0.325] * 3)
        self.ptb_3 = PointTransformerV2Block(in_dim = 384, out_dim = 384, K = 1)

        self.tdb_4 = TransitionDownBlock(in_dim = 384, out_dim = 512, grid_size = [0.8125] * 3)
        self.ptb_4 = PointTransformerV2Block(in_dim = 512, out_dim = 512, K = 1)
        
        # self.avg_pool = nn.AvgPool1d(1)
        self.linear_2 = nn.Linear(512, 512)
        self.ptb_5 = PointTransformerV2Block(in_dim = 512, out_dim = 512, K = 1)

        self.tub_6 = TransitionUpBlock(in_dim = 512, out_dim = 384)
        self.ptb_6 = PointTransformerV2Block(in_dim = 384, out_dim = 384, K = 1)

        self.tub_7 = TransitionUpBlock(in_dim = 384, out_dim = 192)
        self.ptb_7 = PointTransformerV2Block(in_dim = 192, out_dim = 192, K = 2)

        self.tub_8 = TransitionUpBlock(in_dim = 192, out_dim = 96)
        self.ptb_8 = PointTransformerV2Block(in_dim = 96, out_dim = 96, K = 4)

        self.tub_9 = TransitionUpBlock(in_dim = 96, out_dim = 48)
        self.ptb_9 = PointTransformerV2Block(in_dim = 48, out_dim = 48, K = 8)

        self.mlp = nn.Linear(48, n_classes)

    def forward(self, points):

        points_xyz, points_features = points[:, :, :3], points
        out = self.linear_1(points_features)

        out_xyz, out_features = self.ptb_0(points_xyz, out)
        skipped_0_xyz, skipped_0_features = torch.clone(out_xyz), torch.clone(out_features)

        out_xyz, out_features = self.tdb_1(out_xyz, out_features)
        out_xyz, out_features = self.ptb_1(out_xyz, out_features)
        skipped_1_xyz, skipped_1_features = torch.clone(out_xyz), torch.clone(out_features)

        out_xyz, out_features = self.tdb_2(out_xyz, out_features)
        out_xyz, out_features = self.ptb_2(out_xyz, out_features)
        skipped_2_xyz, skipped_2_features = torch.clone(out_xyz), torch.clone(out_features)

        out_xyz, out_features = self.tdb_3(out_xyz, out_features)
        out_xyz, out_features = self.ptb_3(out_xyz, out_features)
        skipped_3_xyz, skipped_3_features = torch.clone(out_xyz), torch.clone(out_features)

        out_xyz, out_features = self.tdb_4(out_xyz, out_features)
        out_xyz, out_features = self.ptb_4(out_xyz, out_features)

        out_features = self.linear_2(out_features)
        out_xyz, out_features = self.ptb_5(out_xyz, out_features)

        out_xyz, out_features = self.tub_6(out_xyz, out_features, skipped_3_xyz, skipped_3_features)
        out_xyz, out_features = self.ptb_6(out_xyz, out_features)

        out_xyz, out_features = self.tub_7(out_xyz, out_features, skipped_2_xyz, skipped_2_features)
        out_xyz, out_features = self.ptb_7(out_xyz, out_features)

        out_xyz, out_features = self.tub_8(out_xyz, out_features, skipped_1_xyz, skipped_1_features)
        out_xyz, out_features = self.ptb_8(out_xyz, out_features)

        out_xyz, out_features = self.tub_9(out_xyz, out_features, skipped_0_xyz, skipped_0_features)
        out_xyz, out_features = self.ptb_9(out_xyz, out_features)

        out = self.mlp(out_features)

        return points_xyz, out 