import torch
import torch.nn as nn

from utils import wrap_angle

class ComfortReward(nn.Module):
    # 舒适性奖励函数，计算ego车辆的加速度和角速度是否在舒适范围内
    def __init__(self, time_span_per_interval=0.5, num_historical_intervals=4):
        self.time_span_per_interval = time_span_per_interval            # 时间间隔
        self.num_historical_intervals = num_historical_intervals        # 历史时间间隔数
        # 奖励阈值设置
        self.min_lon_accel_threshold = -4.05                            # 最小纵向加速度阈值
        self.max_lon_accel_threshold = 2.40                             # 最大纵向加速度阈值    
        self.max_abs_lat_accel_threshold = 4.89                         # 最大绝对横向加速度阈值
        self.max_abs_yaw_accel_threshold = 1.93                         # 最大绝对yaw加速度阈值
        self.max_abs_yaw_rate_threshold = 0.95                          # 最大绝对yaw速率阈值
        super(ComfortReward, self).__init__()

    def forward(self, data):
        # Get the ego vehicle's position and orientation
        # 得到ego车辆的位置信息和航向角信息
        ego_index = data['agent']['ptr'][:-1]                   # ego车辆的索引
        ego_pos = data['agent']['infer_position'][ego_index]    # ego车辆的位置信息
        ego_yaw = data['agent']['infer_heading'][ego_index]     # ego车辆的航向角信息
        T = ego_pos.size(1) - self.num_historical_intervals     # 计算时间步数

        # compute the ego vehicle's acceleration and angular velocity
        ego_vel = (ego_pos[:, 1:] - ego_pos[:, :-1]) / self.time_span_per_interval                              # 计算ego车辆的速度
        ego_acc = (ego_vel[:, 1:] - ego_vel[:, :-1]) / self.time_span_per_interval                              # 计算ego车辆的加速度
        ego_angular_vel = wrap_angle((ego_yaw[:, 1:] - ego_yaw[:, :-1])) / self.time_span_per_interval          # 计算ego车辆的角速度
        ego_angular_acc = (ego_angular_vel[:, 1:] - ego_angular_vel[:, :-1]) / self.time_span_per_interval      # 计算ego车辆的角加速度

        ego_acc = ego_acc[:, -T:]
        ego_angular_vel = ego_angular_vel[:, -T:]
        ego_angular_acc = ego_angular_acc[:, -T:]

        # rotation matrix to transform from global to ego's local coordinates
        # 旋转矩阵，将全局坐标系转换为ego车辆的局部坐标系
        rotation_matrix = torch.stack([
            torch.cos(ego_yaw[:, -T:]), torch.sin(ego_yaw[:, -T:]),
            -torch.sin(ego_yaw[:, -T:]), torch.cos(ego_yaw[:, -T:])
        ], dim=-1).view(-1, T, 2, 2).to(ego_pos.device)

        # transform to ego's local coordinates
        # 将加速度转换为ego车辆的局部坐标系
        ego_acc_local = torch.matmul(rotation_matrix, ego_acc.unsqueeze(-1)).squeeze(-1)
        ego_acc_long = ego_acc_local[..., 0]    # 纵向加速度：沿车辆前进方向
        ego_acc_lat = ego_acc_local[..., 1]     # 横向加速度：垂直车辆前进方向

        # compute the comfort reward
        # # 计算舒适性奖励 Flase或True，满足条件为True，不满足条件为False
        # comfort_reward = (ego_acc_long > self.min_lon_accel_threshold) & \
        #     (ego_acc_long < self.max_lon_accel_threshold) & \
        #     (torch.abs(ego_acc_lat) < self.max_abs_lat_accel_threshold) & \
        #     (torch.abs(ego_angular_acc) < self.max_abs_yaw_accel_threshold) & \
        #     (torch.abs(ego_angular_vel) < self.max_abs_yaw_rate_threshold)
        # comfort_reward = comfort_reward.float()    # [Num, T]
        # 相比与上面的代码，下面的代码将舒适性奖励计算一个连续值，让奖励函数更加平滑，使得对边界附近的行为有更好的惩罚效果
        long_violation = torch.clamp(ego_acc_long - self.max_lon_accel_threshold, min = 0) + torch.clamp(self.min_lon_accel_threshold - ego_acc_long, min = 0)
        lat_violation = torch.clamp(torch.abs(ego_acc_lat) - self.max_abs_lat_accel_threshold, min = 0)
        yaw_acc_violation = torch.clamp(torch.abs(ego_angular_acc) - self.max_abs_yaw_accel_threshold, min = 0)
        yaw_rate_violation = torch.clamp(torch.abs(ego_angular_vel) - self.max_abs_yaw_rate_threshold, min = 0)
        comfort_reward = torch.exp(-long_violation - lat_violation - yaw_acc_violation - yaw_rate_violation) # [Num, T]

        return comfort_reward