import torch
import torch.nn as nn

from utils import wrap_angle

class SpeedLimitReward(nn.Module):
    # 超速奖励函数，计算ego车辆是否超速
    def __init__(self, time_span_per_interval=0.5, num_historical_intervals=4, max_overspeed_value_threshold=2.23, max_overspeed_ratio=0.2):
        self.time_span_per_interval = time_span_per_interval                    # 相邻轨迹点之间的时间间隔
        self.num_historical_intervals = num_historical_intervals                # 历史轨迹时间间隔数量
        self.max_overspeed_value_threshold = max_overspeed_value_threshold      # 允许用于归一化的最大平均超速量
        self.max_overspeed_ratio = max_overspeed_ratio                          # 允许超速比例，按照不同道路限速进行调整
        super(SpeedLimitReward, self).__init__()

    def forward(self, data):
        # Get the ego vehicle's speed
        ego_index = data['agent']['ptr'][:-1]                                       # ego车辆索引
        ego_pos = data['agent']['infer_position'][ego_index]                        # ego车辆位置
        ego_vel = (ego_pos[:, 1:] - ego_pos[:, :-1]) / self.time_span_per_interval  # ego车辆速度
        T = ego_pos.size(1) - self.num_historical_intervals                         # 未来预测时间戳数量
        ego_speed = torch.norm(ego_vel, dim=-1)[:, -T:]                             # 取速度序列最后 T 个时间点，使它与模型预测轨迹长度对应

        # Get the speed limit 
        # 获取车辆信息
        ego_pos = data['agent']['infer_position'][ego_index, self.num_historical_intervals:].reshape(-1, 2)  # [Num * T, 2]
        ego_yaw = data['agent']['infer_heading'][ego_index, self.num_historical_intervals:].reshape(-1)  # [Num * T]
        # 获取地图信息
        polyline_position = data['polyline']['position']
        polyline_heading = data['polyline']['heading']
        polyline_batch = data['polyline']['batch']
        ego_pos_to_polyline_dist = torch.norm(ego_pos.unsqueeze(1) - polyline_position.unsqueeze(0), dim=-1)            # ego车辆与导航线位置偏差
        ego_pos_to_polyline_angle_dist = torch.abs(wrap_angle(ego_yaw.unsqueeze(1) - polyline_heading.unsqueeze(0)))    # ego车辆与导航线角度偏差

        # filter invalid polyline in different batch
        # 过滤非同一场景下的导航线，删除不同场景的导航线
        ego_batch = data['agent']['batch'][ego_index].repeat_interleave(T)  # [Num * T]
        batch_mask = ego_batch.unsqueeze(1) == polyline_batch.unsqueeze(0)
        # 设置距离和角度偏差设置为无穷大，让模型不会选择其他场景的导航线
        ego_pos_to_polyline_dist[~batch_mask] = float('inf')
        ego_pos_to_polyline_angle_dist[~batch_mask] = float('inf')

        # lane only
        # 建立polyline类型
        polygon_type = data['polygon']['type']
        polyline_to_polygon_edge_index = data['polyline', 'polygon']['polyline_to_polygon_edge_index']
        polyline_type = torch.zeros(polyline_position.size(0), device=polyline_position.device, dtype=torch.uint8)
        polyline_type[polyline_to_polygon_edge_index[0]] = polygon_type[polyline_to_polygon_edge_index[1]]
        # 筛选车道polyline
        type_mask = polyline_type == 0
        ego_pos_to_polyline_dist[:, ~type_mask] = float('inf')
        ego_pos_to_polyline_angle_dist[:, ~type_mask] = float('inf')

        # get the closest polyline
        # 选择最近的polyline，匹配代价 = 位置偏差 + 5 * 角度偏差
        ego_pos_to_polyline_min_index = torch.argmin(ego_pos_to_polyline_dist + 5 * ego_pos_to_polyline_angle_dist, dim=1)

        # get the speed limit of the closest polyline
        # 获得对应的道路限速
        polygon_speed_limit = data['polygon']['speed_limit']
        polygon_speed_limit_valid_mask = data['polygon']['speed_limit_valid_mask']
        # 将行驶区域的限速映射到导航线中
        polyline_speed_limit = polygon_speed_limit[polyline_to_polygon_edge_index[1]]
        polyline_speed_limit_valid_mask = polygon_speed_limit_valid_mask[polyline_to_polygon_edge_index[1]]
        # 取出最近polyline的限速
        closest_polyline_speed_limit = polyline_speed_limit[ego_pos_to_polyline_min_index].reshape(-1, T)
        closest_polyline_speed_limit_valid_mask = polyline_speed_limit_valid_mask[ego_pos_to_polyline_min_index].reshape(-1, T)

        # 按固定超速量累计整条轨迹奖励。
        # speed_limit_reward = (ego_speed - closest_polyline_speed_limit).clamp(min=0.0) * closest_polyline_speed_limit_valid_mask.float() * self.time_span_per_interval
        # speed_limit_reward = 1 - speed_limit_reward.sum(-1) / ((self.max_overspeed_value_threshold * T * self.time_span_per_interval))
        # speed_limit_reward = speed_limit_reward.clamp(min=0.0)

        # 按相对超速比例计算指数奖励。
        # 归一化后 x=1 表示超速达到 max_overspeed_ratio，奖励为 0；x>1 时奖励继续下降并区分严重程度。
        overspeed_excess = (ego_speed / closest_polyline_speed_limit.clamp(min=1e-6) - 1.0).clamp(min=0.0)
        normalized_overspeed = overspeed_excess / self.max_overspeed_ratio

        # 默认没有有效限速的时间步奖励为 1；只对有效限速位置写入指数奖励。
        speed_limit_reward = torch.ones_like(ego_speed)
        decay_at_threshold = torch.exp(torch.tensor(-4.0, device=ego_speed.device, dtype=ego_speed.dtype))
        speed_limit_reward[closest_polyline_speed_limit_valid_mask] = (torch.exp(-4 * normalized_overspeed[closest_polyline_speed_limit_valid_mask])- decay_at_threshold).div(1.0 - decay_at_threshold).clamp(min=0.0)
        
        return speed_limit_reward