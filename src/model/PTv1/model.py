import torch
import torch.nn as nn
from src.model.PTv1.point_transformer_utils import PointTransformerBlock, TransitionDownBlock, TransitionUpBlock

class PointTransformerClassifier(nn.Module):

    def __init__(self, npoints = 1000, n_classes = 10, in_dim = 6):

        super().__init__()

        self.linear = nn.Linear(in_dim, 32)
        self.ptb_0 = PointTransformerBlock(in_dim = 32, out_dim = 32)

        self.tdb_1 = TransitionDownBlock(npoints = npoints// 4 ** 1, in_dim = 32, out_dim = 64, K = 16)
        self.ptb_1 = PointTransformerBlock(in_dim = 64, out_dim = 64, K = 16)

        self.tdb_2 = TransitionDownBlock(npoints = npoints// 4**2, in_dim = 64, out_dim = 128, K = 8)
        self.ptb_2 = PointTransformerBlock(in_dim = 128, out_dim = 128, K = 8)

        self.tdb_3 = TransitionDownBlock(npoints = npoints// 4**3, in_dim = 128, out_dim = 256, K = 4)
        self.ptb_3 = PointTransformerBlock(in_dim = 256, out_dim = 256, K = 4)

        self.tdb_4 = TransitionDownBlock(npoints = npoints// 4**4, in_dim = 256, out_dim = 512, K = 2)
        self.ptb_4 = PointTransformerBlock(in_dim = 512, out_dim = 512, K = 2)

        self.avg_pool = nn.AvgPool1d(npoints // 4 ** 4)

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

        out = self.avg_pool(out_features.permute(0, 2, 1))
        out = self.mlp(out.squeeze(-1))

        return out
    
class PointTransformerSegmentation(nn.Module):

    def __init__(self, npoints = 1000, n_classes = 10, in_dim = 6):

        super().__init__()

        self.linear = nn.Linear(in_dim, 32)
        self.ptb_0 = PointTransformerBlock(in_dim = 32, out_dim = 32)

        self.tdb_1 = TransitionDownBlock(npoints = npoints// 4 ** 1, in_dim = 32, out_dim = 64, K = 16)
        self.ptb_1 = PointTransformerBlock(in_dim = 64, out_dim = 64, K = 16)

        self.tdb_2 = TransitionDownBlock(npoints = npoints// 4**2, in_dim = 64, out_dim = 128, K = 8)
        self.ptb_2 = PointTransformerBlock(in_dim = 128, out_dim = 128, K = 8)

        self.tdb_3 = TransitionDownBlock(npoints = npoints// 4**3, in_dim = 128, out_dim = 256, K = 4)
        self.ptb_3 = PointTransformerBlock(in_dim = 256, out_dim = 256, K = 4)

        self.tdb_4 = TransitionDownBlock(npoints = npoints// 4**4, in_dim = 256, out_dim = 512, K = 2)
        self.ptb_4 = PointTransformerBlock(in_dim = 512, out_dim = 512, K = 2)

        self.linear_2 = nn.Linear(512, 512)
        self.ptb_5 = PointTransformerBlock(in_dim = 512, out_dim = 512, K = 2)

        self.tub_6 = TransitionUpBlock(in_dim = 512, out_dim = 256)
        self.ptb_6 = PointTransformerBlock(in_dim = 256, out_dim = 256, K = 4)

        self.tub_7 = TransitionUpBlock(in_dim = 256, out_dim = 128)
        self.ptb_7 = PointTransformerBlock(in_dim = 128, out_dim = 128, K = 8)

        self.tub_8 = TransitionUpBlock(in_dim = 128, out_dim = 64)
        self.ptb_8 = PointTransformerBlock(in_dim = 64, out_dim = 64, K = 16)

        self.tub_9 = TransitionUpBlock(in_dim = 64, out_dim = 32)
        self.ptb_9 = PointTransformerBlock(in_dim = 32, out_dim = 32, K = 16)

        self.mlp = nn.Linear(32, n_classes)


    def forward(self, points):
        
        points_xyz, points_features = points[:, :, :3], points
        out = self.linear(points_features)

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
    
if __name__ == "__main__":
    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 1000, 6).astype('float32')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(X).to(device)

    model = PointTransformerSegmentation(npoints = 1000, n_classes = 10, in_dim = 6)

    model(X)