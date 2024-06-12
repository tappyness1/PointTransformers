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
    
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

def partition_based_pooling(points: torch.Tensor, points_features: torch.Tensor, grid_size: list[float, float, float] = [0.02,0.02,0.02]) -> tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        points (torch.Tensor): xyz values
        points_features (torch.Tensor): features of points
        grid_size (list[float, float, float], optional): _description_. 

    Returns:
        tuple[torch.Tensor, torch.Tensor]: _description_
    """
    batch_size = points.size(0)
    pooled_cluster_points_list = []
    batch_pooled_features_list = []

    for i in range(batch_size):
        cluster_indices = torch_cluster.grid.grid_cluster(points[i], torch.tensor(grid_size, device=points[i].device))
        unique_clusters, inverse_indices = cluster_indices.unique(return_inverse=True)
        num_clusters = unique_clusters.size(0)
        num_features = points_features[i].size(1)

        pooled_features = torch.zeros((num_clusters, num_features), device=points[i].device)
        pooled_features.scatter_reduce_(0, inverse_indices[:, None].expand(-1, num_features), points_features[i], reduce="amax", include_self=False)

        summed_points = torch.zeros((num_clusters, 3), device=points[i].device)
        summed_points.scatter_add_(0, inverse_indices[:, None].expand(-1, 3), points[i])
        cluster_counts = torch.bincount(inverse_indices, minlength=num_clusters).float().unsqueeze(1)
        mean_cluster_points = summed_points / cluster_counts

        pooled_cluster_points_list.append(mean_cluster_points)
        batch_pooled_features_list.append(pooled_features)

    min_size = min([x.size(0) for x in pooled_cluster_points_list])
    pooled_cluster_points_list = [x[:min_size] for x in pooled_cluster_points_list]
    batch_pooled_features_list = [x[:min_size] for x in batch_pooled_features_list]

    pooled_cluster_points = torch.stack(pooled_cluster_points_list)
    batch_pooled_features = torch.stack(batch_pooled_features_list)

    return pooled_cluster_points, batch_pooled_features

class TransitionDownBlock(nn.Module):

    def __init__(self, in_dim, out_dim, grid_size = [0.02, 0.02, 0.02]):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.grid_size = grid_size

    def forward(self, points_xyz, points_features):
        points_features_out = self.linear(points_features)
        points_features_out = self.norm(points_features_out.permute(0,2,1)).permute(0,2,1)
        points_features_out = self.relu(points_features_out)
        cluster_points, pooled_features = partition_based_pooling(points_xyz, points_features_out, grid_size = self.grid_size)
        return cluster_points, pooled_features

class GroupVectorAttention(nn.Module):
    # note here we only implement the Grouped Linear (GL) version of GVA.
    # based on the ablation study, we need to implement: GL -> Batch Norm -> ReLU -> Linear
    def __init__(self, in_dim, out_dim, groups):
        super().__init__()
        self.q,self.k,self.v = nn.Linear(in_dim, in_dim), nn.Linear(in_dim, in_dim), nn.Linear(in_dim, in_dim)
        self.conv_weights = nn.Conv2d(in_dim, out_dim, 1, groups = groups, bias = False)
        self.bn = nn.BatchNorm1d(in_dim)
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
        omega_out = self.softmax_1d(omega_out) # ??
        b,h,w,c = v_out.shape

        # very awkward code here. TODO: clean up
        weight_encoding = (omega_out.permute(0,2,3,1).unsqueeze(-1) * v_out.reshape(b,h,w,self.groups,int(c/self.groups))).reshape(b,h,w,-1)
        out = torch.sum(weight_encoding, dim=2)
        out = self.bn(out.permute(0,2,1)).permute(0,2,1)
        out = torch.relu(out)
        out = self.linear(out)
        
        return out

class PointTransformerV2Block(nn.Module):

    def __init__(self, in_dim, out_dim, groups = 2, K=16):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.gva = GroupVectorAttention(out_dim, groups, groups)
        self.linear_2 = nn.Linear(out_dim, in_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.k = K

    def forward(self, points_xyz, points_features):
        residual = points_features.clone() # may contain xyz still
        # get the neighbours of each point
        distances = torch.cdist(points_xyz, points_xyz)
        _, indices = torch.topk(distances, self.k, largest=False)
        neighbours_xyz = index_points(points_xyz, indices)

        out = self.linear_1(points_features)
        out_neighbours_features = index_points(out, indices)

        out = self.gva(points_xyz, out, neighbours_xyz, out_neighbours_features)
        out = self.linear_2(out)
        out += residual

        return points_xyz, out 

if __name__ == "__main__":
    
    gva = GroupVectorAttention(4, 2, 2)
    points = torch.randn(1, 16, 7)
    neighbours = torch.randn(1, 16, 14, 7)
    out = gva(points[..., :3], points[...,3:], neighbours[..., :3], neighbours[...,3:])
    print(out.shape)