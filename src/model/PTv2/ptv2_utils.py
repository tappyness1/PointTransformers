from __future__ import annotations
import torch
import torch.nn as nn
import torch_cluster

class PositionalEncoding(nn.Module):

    def __init__(self, out_dim):
        super().__init__()

        self.linear_1 = nn.Linear(3, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)
        self.batch_norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, p_i: torch.Tensor, p_j: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            p_i (torch.Tensor): BxNx3
            p_j (torch.Tensor): BxNxMx3

        Returns:
            torch.Tensor: _description_
        """
        # in the pointcept code - https://github.com/Pointcept/PointTransformerV2/blob/main/pcr/models/point_transformer2/point_transformer_v2m2_base.py
        # linear -> batchnorm -> relu -> linear
        # but in the original paper, it's linear -> relu -> linear
        # can you please put it in writing, original authors....
        diff = p_i - p_j
        out = self.linear_1(diff)
        out = out.permute(0,3,1,2)
        out = self.batch_norm(out)
        out = out.permute(0,2,3,1)
        out = self.relu(out)
        out = self.linear_2(out)

        return out

def partition_based_pooling(points: torch.Tensor, points_features: torch.Tensor, grid_size: list[float, float, float] = [0.1,0.1,0.1]) -> tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        points (torch.Tensor): _description_
        points_features (torch.Tensor): _description_
        grid_size (list[float, float, float], optional): _description_. Defaults to [0.1,0.1,0.1].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: _description_
    """

    cluster_indices = torch_cluster.grid.grid_cluster(points, grid_size)
    unique_clusters, inverse_indices = cluster_indices.unique(return_inverse=True)
    num_clusters = unique_clusters.size(0)
    num_features = points_features.size(1)
    pooled_features = torch.zeros((num_clusters, num_features), device=points_features.device)
    pooled_features = torch.zeros_like(pooled_features).scatter_reduce_(0, inverse_indices[:, None].expand(-1, num_features), points_features, reduce="amax")
    cluster_points = points[torch.unique(cluster_indices)]

    return cluster_points, pooled_features

class GroupVectorAttention(nn.Module):
    # note here we only implement the Grouped Linear (GL) version of GVA.
    # based on the ablation study, we need to implement: GL -> Batch Norm -> ReLU -> Linear
    def __init__(self, in_dim, out_dim, groups):
        super().__init__()
        self.q,self.k,self.v = nn.Linear(in_dim, in_dim), nn.Linear(in_dim, in_dim), nn.Linear(in_dim, in_dim)
        self.conv_weights = nn.Conv2d(in_dim, out_dim, 1, groups = groups, bias = False)
        self.bn = nn.BatchNorm2d(in_dim)
        self.delta_mult = PositionalEncoding(in_dim)
        self.delta_bias = PositionalEncoding(in_dim)
        self.softmax_1d = nn.Softmax(dim=1)
        self.linear = nn.Linear(in_dim, in_dim)
        self.out_dim = out_dim
        self.groups = groups
        
    def forward(self, points_xyz, points_features, neighbours_xyz, neighbours_features):
        delta_mult_out = self.delta_mult(points_xyz.unsqueeze(-2), neighbours_xyz)
        delta_bias_out = self.delta_bias(points_xyz.unsqueeze(-2), neighbours_xyz)

        q_out = self.q(points_features.unsqueeze(-2))
        k_out = self.k(neighbours_features)
        v_out = self.v(neighbours_features)
        gamma_out = q_out - k_out
        vector_attention = (delta_mult_out * gamma_out) + delta_bias_out
        omega_out = self.conv_weights(vector_attention.permute(0,3,1,2))
        omega_out = self.softmax_1d(omega_out)
        b,c,h,w = omega_out.shape
        
        # very awkward code here. TODO: cleaner
        weight_encoding = (omega_out.permute(0,2,3,1).unsqueeze(-1) * v_out.reshape(b,h,w,c,self.groups)).reshape(b,h,w,-1).permute(0,3,1,2)
        out = self.bn(weight_encoding)
        out = out.permute(0,2,3,1)
        out = torch.relu(out)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    
    gva = GroupVectorAttention(4, 2, 2)
    points = torch.randn(1, 16, 7)
    neighbours = torch.randn(1, 16, 14, 7)
    out = gva(points[..., :3], points[...,3:], neighbours[..., :3], neighbours[...,3:])
    print(out.shape)