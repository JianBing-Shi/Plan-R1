import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter_min

from utils import transform_point_to_local_coordinate
from utils import drop_edge_between_samples

class ProgressReward(nn.Module):
    # 导航进度奖励函数，计算ego车辆的行驶进度与专家轨迹的行驶进度的比值
    def __init__(self, num_historical_intervals=4, interval=5, progress_threshold=2.0):
        self.progress_threshold = progress_threshold                # 防止专家进度过小导致除以零或者比例过大
        self.interval = interval                                    # 专家轨迹按照N个数据点取一个位置
        self.num_historical_intervals = num_historical_intervals    # 前N个历史轨迹
        super(ProgressReward, self).__init__()

    def forward(self, data):
        # get the point on route
        polyline_batch = data['polyline']['batch']                                                      # 导航线场景
        polyline_position = data['polyline']['position']                                                # 导航线参考位置
        polyline_heading = data['polyline']['heading']                                                  # 导航线方向角
        polyline_to_polygon_edge_index = data['polyline', 'polygon']['polyline_to_polygon_edge_index']  # 导航线与行驶区域的图连接关系
        polygon_on_route_mask = data['polygon']['on_route_mask']                                        # 行驶区域的有效掩码
        polyline_on_route_mask = polygon_on_route_mask[polyline_to_polygon_edge_index[1]]               # 当前导航线

        # compute the progress of trajectory
        # 得到ego当前时刻和上一时刻的位置信息
        ego_index = data['agent']['ptr'][:-1]
        ego_pos = data['agent']['infer_position'][ego_index, self.num_historical_intervals:].reshape(-1, 2)    # [Num * T, 2]
        ego_pos_pre = data['agent']['infer_position'][ego_index, self.num_historical_intervals-1:-1].reshape(-1, 2)    # [Num * T, 2]
        T = data['agent']['infer_position'].size(1) - self.num_historical_intervals
        ego_batch = data['agent']['batch'][ego_index].repeat_interleave(T)    # [Num * T]

        # 建立ego与路线的polyline的有效配对
        ego_pos_to_route_mask = torch.ones(ego_pos.size(0), device=ego_pos.device, dtype=torch.bool).unsqueeze(1) & polyline_on_route_mask.unsqueeze(0)    # [Num * T, (R1,...,Rb)]
        ego_pos_to_route_mask = drop_edge_between_samples(ego_pos_to_route_mask, batch=(ego_batch, polyline_batch)) # 删除不同场景之间的配对
        ego_pos_to_route_edge_index = dense_to_sparse(ego_pos_to_route_mask)[0]                                     # 得到有效配对边
        # 把ego当前时刻和上一时刻的位置信息转换到polyline局部坐标系
        ego_pos_to_route = transform_point_to_local_coordinate(ego_pos[ego_pos_to_route_edge_index[0]], polyline_position[ego_pos_to_route_edge_index[1]], polyline_heading[ego_pos_to_route_edge_index[1]])
        ego_pos_pre_to_route = transform_point_to_local_coordinate(ego_pos_pre[ego_pos_to_route_edge_index[0]], polyline_position[ego_pos_to_route_edge_index[1]], polyline_heading[ego_pos_to_route_edge_index[1]])

        # 选择每个位置最近的路线 polyline
        ego_pos_to_route_edge_dist = (
            torch.abs(ego_pos_to_route[:, 1]) * 10      # 横向距离 * 10，更重视车辆是否靠近正确的道路横向位置
            + torch.abs(ego_pos_to_route[:, 0])         # 纵向距离
        )
        ego_pos_to_route_edge_mask = ego_pos_to_route[:, 0] > 0
        ego_pos_to_route_edge_dist[ego_pos_to_route_edge_mask] += 1000
        # 从所有候选路线 polyline 中选择距离最小的一个。
        _, ego_pos_to_route_dist_min_index = scatter_min(ego_pos_to_route_edge_dist, ego_pos_to_route_edge_index[0], dim_size=ego_pos.size(0))
        # 判断该位置是否找到了有效路线 polyline
        valid_index_mask = ego_pos_to_route_dist_min_index != ego_pos_to_route_edge_index.size(1)

        # 计算模型轨迹的总进度
        progress_per_step = torch.zeros(ego_pos.size(0), device=ego_pos.device)
        progress_per_step[valid_index_mask] = ego_pos_to_route[ego_pos_to_route_dist_min_index[valid_index_mask], 0] - ego_pos_pre_to_route[ego_pos_to_route_dist_min_index[valid_index_mask], 0]
        progress_per_step = progress_per_step.reshape(-1, T)
        progress = progress_per_step.sum(dim=-1)    # [Num]

        # compute the progress of expert trajectory
        # 计算专家轨迹进度
        expert_pos = data['agent']['position'][ego_index, ::self.interval]
        expert_pos_pre = expert_pos[:, -T-1:-1].reshape(-1, 2)   # [Num * T, 2]
        expert_pos = expert_pos[:, -T:].reshape(-1, 2)           # [Num * T, 2]
        expert_batch = data['agent']['batch'][ego_index].repeat_interleave(T)        # [Num * T]

        expert_pos_to_route_mask = torch.ones(expert_pos.size(0), device=expert_pos.device, dtype=torch.bool).unsqueeze(1) & polyline_on_route_mask.unsqueeze(0)    # [Num * T, (R1,...,Rb)]
        expert_pos_to_route_mask = drop_edge_between_samples(expert_pos_to_route_mask, batch=(expert_batch, polyline_batch))
        expert_pos_to_route_edge_index = dense_to_sparse(expert_pos_to_route_mask)[0]
        expert_pos_to_route = transform_point_to_local_coordinate(expert_pos[expert_pos_to_route_edge_index[0]], polyline_position[expert_pos_to_route_edge_index[1]], polyline_heading[expert_pos_to_route_edge_index[1]])
        expert_pos_pre_to_route = transform_point_to_local_coordinate(expert_pos_pre[expert_pos_to_route_edge_index[0]], polyline_position[expert_pos_to_route_edge_index[1]], polyline_heading[expert_pos_to_route_edge_index[1]])
        
        expert_pos_to_route_edge_dist = torch.abs(expert_pos_to_route[:, 1]) * 10 + torch.abs(expert_pos_to_route[:, 0])
        expert_pos_to_route_mask = expert_pos_to_route[:, 0] > 0
        expert_pos_to_route_edge_dist[expert_pos_to_route_mask] += 1000
        _, expert_pos_to_route_dist_min_index = scatter_min(expert_pos_to_route_edge_dist, expert_pos_to_route_edge_index[0], dim_size=expert_pos.size(0))
        valid_index_mask = expert_pos_to_route_dist_min_index != expert_pos_to_route_edge_index.size(1)

        expert_progress_per_step = torch.zeros(expert_pos.size(0), device=expert_pos.device)
        expert_progress_per_step[valid_index_mask] = expert_pos_to_route[expert_pos_to_route_dist_min_index[valid_index_mask], 0] - expert_pos_pre_to_route[expert_pos_to_route_dist_min_index[valid_index_mask], 0]
        expert_progress_per_step = expert_progress_per_step = expert_progress_per_step.reshape(-1, T)
        expert_progress = expert_progress_per_step.sum(dim=-1)    # [Num]

        # compute the progress reward
        # 生成最终进度奖励（模型总进度/专家总进度）
        # progress_reward = (progress.clamp(min=self.progress_threshold) / (expert_progress * 1.0).clamp(min=self.progress_threshold)).clamp(max=1.2)
        # 先判断哪些场景达到最低进度阈值
        valid_trajectory_mask = (progress >= self.progress_threshold) & (expert_progress >= self.progress_threshold)

        # 默认所有场景、所有时间步奖励为 0，输出形状为 [Num, T]
        progress_reward = torch.zeros_like(progress_per_step)

        # 只对整条轨迹达到阈值的场景，逐时间步计算模型进度/专家进度
        eps = torch.finfo(expert_progress_per_step.dtype).eps
        valid_step_mask = (
            valid_trajectory_mask.unsqueeze(1)
            & (expert_progress_per_step > eps)
        )
        progress_reward[valid_step_mask] = (
            progress_per_step[valid_step_mask].clamp(min=0.0)
            / expert_progress_per_step[valid_step_mask]
        ).clamp(max=1.2)

        return progress_reward
