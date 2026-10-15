import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter_max

from utils import transform_point_to_local_coordinate
from utils import compute_corner_positions
from utils import drop_edge_between_samples

class TTCReward(nn.Module):
    # TTC奖励函数，计算ego车辆与周围车辆的碰撞时间是否在安全范围内
    def __init__(self, long_threshold=0.1, lat_threshold=0.1, num_historical_intervals=4, least_min_ttc=0.95, time_span_per_interval=0.5, min_ttc=0.8, max_ttc=1.8):
        self.long_threshold = long_threshold                        # 纵向阈值
        self.lat_threshold = lat_threshold                          # 横向阈值
        self.num_historical_intervals = num_historical_intervals    # 历史时间间隔数
        self.least_min_ttc = least_min_ttc                          # 最小TTC时间阈值
        self.time_span_per_interval = time_span_per_interval        # 时间间隔
        self.danger_ttc = min_ttc                                   # 危险阈值，单位：秒
        self.safe_ttc = max_ttc                                     # 安全阈值，单位：秒
        super(TTCReward, self).__init__()

    def forward(self, data):
        # get the ego vehicle's trajectory and surrounding agents' trajectory
        # 得到ego车辆的轨迹和周围车辆的轨迹
        vel = data['agent']['infer_position'][:, self.num_historical_intervals:] - data['agent']['infer_position'][:, self.num_historical_intervals-1:-1]    # [Num, T, 2]
        vel = vel / self.time_span_per_interval    # [Num, T, 2]
        batch = data['agent']['batch']    # [Num]
        ego_index = data['agent']['ptr'][:-1]    # [Num_ego]
        num_ego = len(ego_index)
        N, T, _ = vel.size()
        device = batch.device
        horizons = [
            self.danger_ttc + 0.1 * i
            for i in range(int(round((self.safe_ttc - self.danger_ttc) / 0.1)) + 1)
        ]
        active_mask = torch.ones(num_ego, T, device=device, dtype=torch.bool)
        ttc_reward = torch.ones(num_ego, T, device=device)

        for horizon_index, horizon in enumerate(horizons):
            if not active_mask.any():
                break
            # pos_ttc = data['agent']['infer_position'][:, self.num_historical_intervals:] + vel * self.least_min_ttc    # [Num, T, 2]
            pos_ttc = data['agent']['infer_position'][:, self.num_historical_intervals:] + vel * horizon    # [Num, T, 2]
            yaw_ttc = data['agent']['infer_heading'][:, self.num_historical_intervals:]    # [Num, T]
            box = data['agent']['box']    # [Num, 4]
            corner_pos_ttc = compute_corner_positions(      # 计算车辆的四个角点在TTC时间戳下的位置，不再只看车辆中心点，而是考虑车辆实际尺寸
                pos_ttc,
                yaw_ttc,
                box.unsqueeze(-1)
            )    # [Num, T, 4, 2]

            # 得到ego车辆的位姿信息，将其转换为TTC时间戳下的位姿信息，并将其与周围车辆的位姿信息进行碰撞检测
            ego_pos_ttc = pos_ttc[ego_index].transpose(0, 1).reshape(-1, 2)     # [T*Num_ego, 2]
            ego_yaw_ttc = yaw_ttc[ego_index].transpose(0, 1).reshape(-1)        # [T*Num_ego]
            ego_mask = data['agent']['infer_valid_mask'][ego_index, self.num_historical_intervals:].transpose(0, 1)    # [T, Num_ego]
            ego_batch = batch[ego_index]                                        # [Num_ego]
            ego_front = box[ego_index, 0].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num_ego]   # 车辆前端位置
            ego_rear = box[ego_index, 1].unsqueeze(0).expand(T, -1).reshape(-1)     # [T*Num_ego]   # 车辆后端位置
            ego_left = box[ego_index, 2].unsqueeze(0).expand(T, -1).reshape(-1)     # [T*Num_ego]   # 车辆左侧位置
            ego_right = box[ego_index, 3].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num_ego]   # 车辆右侧位置
            ego_corner_pos_ttc = corner_pos_ttc[ego_index].transpose(0, 1).reshape(-1, 4, 2)    # [T*Num_ego, 4, 2] # 车辆四个焦点发生TTC碰撞

            # 得到agent车辆的位置信息，将其转换为TTC时间戳下的位姿信息
            agent_pos_ttc = pos_ttc.transpose(0, 1).reshape(-1, 2)    # [T*Num, 2]
            agent_yaw_ttc = yaw_ttc.transpose(0, 1).reshape(-1)    # [T*Num]
            agent_mask = data['agent']['infer_valid_mask'][:, self.num_historical_intervals:].transpose(0, 1)    # [T, Num]
            agent_batch = batch    # [Num]
            agent_front = box[..., 0].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
            agent_rear = box[..., 1].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
            agent_left = box[..., 2].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
            agent_right = box[..., 3].unsqueeze(0).expand(T, -1).reshape(-1)    # [T*Num]
            agent_corner_pos_ttc = corner_pos_ttc.transpose(0, 1).reshape(-1, 4, 2)    # [T*Num, 4, 2]

            # 建立有效的 ego-agent 配对
            edge_mask = ego_mask.unsqueeze(2) & agent_mask.unsqueeze(1)    # [T, Num_ego, Num]  # 某一时间戳的自车和某个agent是否都有效
            edge_mask = drop_edge_between_samples(edge_mask, batch=(ego_batch, agent_batch))    # [T, Num_ego, Num] # 这一行删除不同场景之间的配对，保证场景 A 的自车只与场景 A 的 agent 比较
            # 删除自车与自己之间的配对，保证场景 A 的自车只与场景 A 的 其他agent 比较
            self_mask = ego_index.unsqueeze(-1) != torch.arange(N, device=device).unsqueeze(0)    # [1, Num_ego, Num]
            edge_mask = edge_mask & self_mask.unsqueeze(0)    # [T, Num_ego, Num]
            edge_mask = edge_mask & active_mask.transpose(0, 1).unsqueeze(2)

            # 当某场景没有其他有效agent时，则认为所有时间戳安全
            ego_in_agent_edge_index = dense_to_sparse(edge_mask)[0]    # [2, E]
            if ego_in_agent_edge_index.size(1) == 0:
                # return torch.ones(num_ego, T, device=device)
                if horizon_index == 0:
                    return torch.ones(num_ego, T, device=device)
                break
            agent_in_ego_edge_index = dense_to_sparse(edge_mask.transpose(1,2))[0]    # [2, E]

            # 将角点变换到对方 agent 坐标系，对方 agent 的角点转换到自车坐标系
            # 为什么要做双向转换？
            # 因为两个车辆可能有不同航向。通过在双方局部坐标系下分别检查，可以判断：自车角点是否进入对方矩形，对方角点是否进入自车矩形
            ego_corner_in_agent_pos_ttc = torch.stack([transform_point_to_local_coordinate(ego_corner_pos_ttc[ego_in_agent_edge_index[0], i], agent_pos_ttc[ego_in_agent_edge_index[1]], agent_yaw_ttc[ego_in_agent_edge_index[1]]) for i in range(4)], dim=1)    # [E, 4, 2]
            agent_corner_in_ego_pos_ttc = torch.stack([transform_point_to_local_coordinate(agent_corner_pos_ttc[agent_in_ego_edge_index[0], i], ego_pos_ttc[agent_in_ego_edge_index[1]], ego_yaw_ttc[agent_in_ego_edge_index[1]]) for i in range(4)], dim=1)    # [E, 4, 2]

            # 计算 agent 角点相对于自车的侵入程度
            agent_corner_in_ego_ttc = torch.stack([torch.min(torch.stack([
                    torch.clamp((ego_front[agent_in_ego_edge_index[1]] + self.long_threshold - agent_corner_in_ego_pos_ttc[..., i, 0]), min=0),
                    torch.clamp((ego_rear[agent_in_ego_edge_index[1]] + self.long_threshold + agent_corner_in_ego_pos_ttc[..., i, 0]), min=0),
                    torch.clamp((ego_left[agent_in_ego_edge_index[1]] + self.lat_threshold - agent_corner_in_ego_pos_ttc[..., i, 1]), min=0),
                    torch.clamp((ego_right[agent_in_ego_edge_index[1]] + self.lat_threshold + agent_corner_in_ego_pos_ttc[..., i, 1]), min=0)
            ], dim=0), dim=0).values for i in range(4)], dim=0)    # [4, E]
            ego_corner_in_agent_ttc = torch.stack([torch.min(torch.stack([
                    torch.clamp((agent_front[ego_in_agent_edge_index[1]] + self.long_threshold - ego_corner_in_agent_pos_ttc[..., i, 0]), min=0),
                    torch.clamp((agent_rear[ego_in_agent_edge_index[1]] + self.long_threshold + ego_corner_in_agent_pos_ttc[..., i, 0]), min=0),
                    torch.clamp((agent_left[ego_in_agent_edge_index[1]] + self.lat_threshold - ego_corner_in_agent_pos_ttc[..., i, 1]), min=0),
                    torch.clamp((agent_right[ego_in_agent_edge_index[1]] + self.lat_threshold + ego_corner_in_agent_pos_ttc[..., i, 1]), min=0)
            ], dim=0), dim=0).values for i in range(4)], dim=0)    # [4, E]

            # 合并两个方向的几何结果
            ttc_loss = torch.max(torch.cat([agent_corner_in_ego_ttc, ego_corner_in_agent_ttc]), dim=0).values
            _, ttc_loss_max_index = scatter_max(ttc_loss, ego_in_agent_edge_index[0], dim_size=num_ego * T)
            # 生成当前 horizon 的二值碰撞结果
            valid_index_mask = ttc_loss_max_index != len(ego_in_agent_edge_index[0])
            # ttc_reward = torch.ones(num_ego * T, device=device, dtype=torch.bool)
            # ttc_reward[valid_index_mask] = ttc_loss[ttc_loss_max_index[valid_index_mask]] <= 0
            collision = torch.zeros(num_ego * T, device=device, dtype=torch.bool)
            collision[valid_index_mask] = ttc_loss[ttc_loss_max_index[valid_index_mask]] > 0
            collision = collision.reshape(T, -1).transpose(0, 1)

            first_collision = active_mask & collision
            current_reward = (horizon - self.danger_ttc) / (self.safe_ttc - self.danger_ttc)
            ttc_reward[first_collision] = current_reward
            active_mask[first_collision] = False
            
        return ttc_reward
