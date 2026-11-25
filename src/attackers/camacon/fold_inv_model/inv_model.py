import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attackers.camacon.fold_inv_model.model import GraphLayer
from src.attackers.camacon.fold_inv_model.utils import index_points, knn


class CouplingLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Invertible_decoder(nn.Module):
    def __init__(self, in_channel=512, num_points=1024):
        super(Invertible_decoder, self).__init__()
        self.num_points = num_points
        self.F0 = CouplingLayer(in_dim=3, hidden_dim=256, out_dim=512)  # U(3维) → 512维
        self.G0 = CouplingLayer(in_dim=512, hidden_dim=256, out_dim=3)  # Y1(512维) → 3维
        self.F1 = CouplingLayer(in_dim=3, hidden_dim=256, out_dim=512)  # Y2(3维) → 512维
        self.G1 = CouplingLayer(in_dim=512, hidden_dim=256, out_dim=3)  # θ_U(512维) → 3维
        self.sphere_points = self.create_sphere_points()

    def create_sphere_points(self):  # 生成球形点云
        indices = torch.arange(0, self.num_points, dtype=torch.float32)
        phi = torch.acos(1 - 2 * indices / self.num_points)
        theta = torch.pi * (1 + 5 ** 0.5) * indices

        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        points = torch.stack([x, y, z], dim=0)  # [3, num_points]
        return points

    def forward(self, theta_P, U=None):
        batch_size = theta_P.shape[0]
        if U is None:
            U = self.sphere_points.unsqueeze(0).repeat(batch_size, 1, 1).to(theta_P.device)
        else:
            U = U.to(theta_P.device)

        if U.shape[-1] != self.num_points:
            raise ValueError(f"球形点云点数{U.shape[-1]}与输入点数{self.num_points}不一致")
        theta_P_pts = theta_P.permute(0, 2, 1)
        U_pts = U.permute(0, 2, 1)

        Y1 = theta_P_pts + self.F0(U_pts)
        Y2 = U_pts + self.G0(Y1)
        theta_U = Y1 + self.F1(Y2)
        P = Y2 + self.G1(theta_U)

        P = P.permute(0, 2, 1)
        theta_U = theta_U.permute(0, 2, 1)

        return P, theta_U

    def inverse(self, theta_U, P):
        # 重塑为点维度在最后
        theta_U_pts = theta_U.permute(0, 2, 1)  # [B, N, 512]
        P_pts = P.permute(0, 2, 1)  # [B, N, 3]

        # 反向映射
        Y2 = P_pts - self.G1(theta_U_pts)  # [B, N, 3]
        Y1 = theta_U_pts - self.F1(Y2)  # [B, N, 512]
        U = Y2 - self.G0(Y1)  # [B, N, 3]
        theta_P = Y1 - self.F0(U)  # [B, N, 512]

        # 转回标准格式
        theta_P = theta_P.permute(0, 2, 1)
        U = U.permute(0, 2, 1)

        return theta_P, U


class New_Encoder(nn.Module):

    def __init__(self, num_points=1024):
        super(New_Encoder, self).__init__()

        self.conv1 = nn.Conv1d(12, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.graph_layer1 = GraphLayer(in_channel=64, out_channel=128, k=16)
        self.graph_layer2 = GraphLayer(in_channel=128, out_channel=1024, k=16)

        self.conv4 = nn.Conv1d(1024, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        self.num_points = num_points

    def forward(self, x):
        b, c, n = x.size()

        # get the covariances, reshape and concatenate with x
        knn_idx = knn(x, k=16)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        covariances = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1).permute(0, 2, 1)
        x = torch.cat([x, covariances], dim=1)  # (B, 12, N)

        # three layer MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # two consecutive graph layers
        x = self.graph_layer1(x)
        x = self.graph_layer2(x)

        x = self.bn4(self.conv4(x))
        global_feature = torch.max(x, dim=-1)[0]
        return x, global_feature


class InvertibleAutoEncoder(nn.Module):
    def __init__(self, num_points=1024, pretrained_path=None):
        super(InvertibleAutoEncoder, self).__init__()
        self.num_points = num_points
        self.encoder = New_Encoder(num_points)
        self.decoder = Invertible_decoder(in_channel=512, num_points=num_points)

        # 加载预训练权重
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        """加载原始自编码器的预训练权重"""
        try:
            # 加载原始模型权重
            pretrained_dict = torch.load(pretrained_path, weights_only=True)
            current_dict = self.state_dict()

            # 筛选可以加载的权重（主要是编码器部分）
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k in current_dict:
                    if current_dict[k].shape == v.shape:
                        filtered_dict[k] = v
                    else:
                        print(f"跳过 {k}: 形状不匹配 {v.shape} -> {current_dict[k].shape}")
                else:
                    # 处理编码器权重的键名映射
                    if k.startswith('encoder.'):
                        # 直接匹配
                        if k in current_dict and current_dict[k].shape == v.shape:
                            filtered_dict[k] = v
                    elif any(k.startswith(prefix) for prefix in ['conv', 'bn', 'graph_layer']):
                        # 添加encoder前缀
                        new_key = f'encoder.{k}'
                        if new_key in current_dict and current_dict[new_key].shape == v.shape:
                            filtered_dict[new_key] = v

            # 加载匹配的权重
            current_dict.update(filtered_dict)
            self.load_state_dict(current_dict, strict=False)
            self.pretrained_load_report = {
                'path': pretrained_path,
                'loaded_keys': len(filtered_dict),
                'checkpoint_keys': len(pretrained_dict),
                'decoder_keys_loaded': sum(k.startswith('decoder.') for k in filtered_dict),
                'loaded_key_names': sorted(filtered_dict),
            }
            print(f"成功加载 {len(filtered_dict)}/{len(pretrained_dict)} 个参数")
            decoder_loaded = self.pretrained_load_report['decoder_keys_loaded']
            print(f"其中已加载解码器参数 {decoder_loaded} 项")

        except Exception as e:
            self.pretrained_load_report = {'path': pretrained_path, 'error': repr(e)}
            raise RuntimeError(f"加载预训练权重失败: {e}") from e

    def forward(self, original_points, sphere_pc=None):

        # 编码器返回局部特征 [B, 512, N] 和全局特征 [B, 512]
        local_feature, global_feature = self.encoder(original_points)

        # 解码器使用局部特征进行可逆变换
        P, theta_U = self.decoder(local_feature, sphere_pc)

        return P, theta_U, global_feature, local_feature

    def inverse(self, theta_U, P):

        theta_P, U = self.decoder.inverse(theta_U, P)
        return theta_P, U


class InvertibleAELoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.01):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, original_pc, reconstructed_pc):
        recon_loss = self.mse_loss(reconstructed_pc, original_pc)
        return recon_loss
