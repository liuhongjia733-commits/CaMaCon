"""Implementation of CaMaCon."""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch3d.ops import knn_points, knn_gather, sample_farthest_points

from .dist_utils import ChamferDist, HausdorffDist
from .conflict_diagnostics import (
    clean_geometry, conflict_metrics, correspondence_metrics,
    operator_losses, remove_negative_component, tensor_dict_to_records,
)
from src.utils.save_utils import pre_check_path

class UntargetedLogitsAdvLoss(torch.nn.Module):
    """无目标攻击损失函数"""

    def __init__(self, device, kappa=0.0):
        super(UntargetedLogitsAdvLoss, self).__init__()
        self.device = device
        self.kappa = kappa

    def forward(self, logits, targets):
        B, K = logits.shape
        if len(targets.shape) == 1:
            targets = targets.view(-1, 1)
        targets = targets.long()
        one_hot_targets = (
            torch.zeros(B, K, device=self.device).scatter(1, targets, 1).float()
        )
        real_logits = torch.sum(one_hot_targets * logits, dim=1)
        other_logits = torch.max(
            (1.0 - one_hot_targets) * logits - one_hot_targets * 10000.0, dim=1
        )[0]
        # [FIXED] 返回 [B] 形状的张量，而不是标量，为了配合C&W逐样本控制
        loss = torch.clamp(real_logits - other_logits + self.kappa, min=0.0)
        return loss


class CaMaCon:
    """Class for CaMaCon Attack"""

    def __init__(
            self,
            model,
            ae_model,
            device,
            **kwargs
    ):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.ae_model = ae_model.to(self.device)
        self.ae_model.eval()

        default_params = {
            'attack_lr': 0.009745891853290296,
            'init_weight': 14.43601042042758,
            'max_weight': 49.846468064858904,
            'binary_step': 13,
            'num_iter': 50,
            'clip_func': None,
            'haus_weight': 0.0001421500937849392,
            'cd_weight': 0.0006320109469140246,
            'curv_weight': 4.948068608530099e-05,
            # Explicit clean-normal preservation term used by the
            # CD+HD+normal comparison. Zero preserves all legacy behaviour.
            'normal_weight': 0.0,
            'ker_weight': 0.5598656905996824,
            'hide_weight': 1.1637463197209648,
            'curv_loss_knn': 16,
            'central_num': 192,
            'total_central_num': 256,
            'max_sigm': 1.2,
            'min_sigm': 0.1,
            'budget': 0.22474794364994255,
            'alpha': 1,
            'latent_loss_weight': 19.783428681750156,
            # Mechanism-ablation controls. Defaults preserve the current attack.
            'constraint_space': 'sphere',  # sphere | point | none
            'mask_mode': 'grad',           # uniform | detach | grad
            'latent_reference': 'standard',# standard | clean_mapping
            'point_reference': 'clean',     # clean | reconstruction
            'constraint_kind': 'local',     # local | radial | projected_local
            'mapping_mode': 'nonlinear',    # nonlinear | linearized
            'distance_mode': 'hard',        # hard | smooth
            'soft_temperature': 0.01,
            'candidate_rule': 'transform', # transform | cd | cd_hd | point_l2
            'candidate_hd_weight': 0.1,     # only used by candidate_rule='cd_hd'
            'perturbation_kernel': 'gaussian',
            'matched_displacement_control': False,
            'sphere_angle_weight': 0.0,
            # Centroid alignment: forbid the "global translation" shortcut. When
            # enabled, every candidate adversarial cloud is re-centered to the
            # clean cloud's centroid before the forward pass. This is a linear,
            # differentiable operation, so gradients still flow back to the
            # perturbation parameters; only the translation degree of freedom is
            # removed. Default False preserves all legacy behaviour.
            'centroid_align': False,
            'record_diagnostics': False,
            'diagnostic_interval': 10,
            'fixed_scale_const': None,
            # Phase 7--12 conflict diagnostics. Disabled by default.
            'conflict_diagnostics': False,
            'conflict_interval': 1,
            'record_pointwise': False,
            'pointwise_sample_limit': 2,
            'center_selection_mode': 'original',  # original | high_complexity | low_complexity | mixed
            'intervention_mode': 'none',  # none | all | cd | hd | translation | scale | normal | tangent | lowfreq | highfreq | high_complexity | low_complexity
            'intervention_gate': 'always',  # always | pre_success | post_success | switch_event | complexity_dominant
            'restore_intervention_norm': True,
            # [NEW] 可视化对比图相关参数
            'sphere_vis_dir': 'figures/camacon_sphere_v2',
            'snapshot_iters': [10, 20, 40],
            'generate_comparison': False,
            'total_samples': 50,  # 总共生成多少份不同点云的对比图/折线图
        }

        default_params.update(kwargs)
        if default_params['perturbation_kernel'] not in (
            'gaussian', 'wendland_c2', 'adaptive_gradient', 'dfs_gaussian'
        ):
            raise ValueError(f"Unknown perturbation kernel: {default_params['perturbation_kernel']}")
        if default_params['perturbation_kernel'] == 'adaptive_gradient' and (
            not default_params['matched_displacement_control']
            or default_params['intervention_mode'] != 'none'
            or default_params['conflict_diagnostics']
            or default_params['record_diagnostics']
        ):
            raise ValueError('Adaptive gradient requires matched displacement control and standard optimization.')
        if default_params['candidate_rule'] not in ('transform', 'cd', 'cd_hd', 'point_l2'):
            raise ValueError(
                "candidate_rule must be one of transform | cd | cd_hd | point_l2"
            )

        for key, value in default_params.items():
            setattr(self, key, value)

        self.adv_func = UntargetedLogitsAdvLoss(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.losses = {
            'adv_loss': [],
            'dist_loss': [],
            'chamfer_loss': [],
            'transformation_loss': [],
            'hide_loss': [],
            'latent_space_loss': [],
        }

    # ============================================================
    # [NEW] 可视化辅助方法
    # ============================================================
    def _to_bnc_numpy(self, points):
        """将任意形式的批量点云转换为 [B, N, 3] 形状的 numpy 数组。"""
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().float().numpy()
        points = np.asarray(points)
        if points.ndim != 3:
            raise ValueError(f"Expected a batched point cloud with 3 dims, got {points.shape}")
        if points.shape[1] == 3:
            points = np.transpose(points, (0, 2, 1))
        if points.shape[-1] != 3:
            raise ValueError(f"Expected point coordinates on the last dim, got {points.shape}")
        return points

    def _set_equal_3d_axes(self, ax, points, limit=None):
        """为 3D 散点子图设置等比例坐标轴。"""
        if limit is None:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            center = (mins + maxs) / 2.0
            radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
        else:
            center = np.zeros(3)
            radius = float(limit)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    def compute_sphere_knn_metrics(self, U_orig, U_recon, k=16):
        """计算球面点云的 KNN 平均距离和平均角度（每样本指标）。

        Args:
            U_orig (torch.Tensor): 原始球面点云 [B, 3, N]
            U_recon (torch.Tensor): 重建球面点云 [B, 3, N]
            k (int): KNN 邻居数

        Returns:
            knn_dist_mean: [B] — 每个样本中所有点邻域的 Chamfer 距离均值
            knn_angle_mean: [B] — 每个样本中所有点邻域的 (1 - cos_sim) 均值
        """
        B, C, N = U_orig.shape
        U_orig_knn = U_orig.transpose(1, 2)
        U_recon_knn = U_recon.transpose(1, 2)

        # U_orig 每个点在 U_orig 自身找 k 个邻居
        _, _, knn_pts_orig = knn_points(
            U_orig_knn, U_orig_knn, K=k, return_nn=True
        )  # [B, N, k, C]

        # U_orig 每个点在 U_recon 中找 k 个邻居
        _, _, knn_pts_recon = knn_points(
            U_orig_knn, U_recon_knn, K=k, return_nn=True
        )  # [B, N, k, C]

        # 计算邻域 Chamfer 距离（每点）
        pts_recon = knn_pts_recon.reshape(B * N, k, C)
        pts_orig = knn_pts_orig.reshape(B * N, k, C)
        dist = torch.cdist(pts_recon, pts_orig, p=2)  # [B*N, k, k]
        min_dist1 = torch.min(dist, dim=2)[0]  # [B*N, k]
        min_dist2 = torch.min(dist.transpose(1, 2), dim=2)[0]  # [B*N, k]
        local_cd = (torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1))  # [B*N]
        knn_dist_mean = local_cd.view(B, N).mean(dim=1)  # [B]

        # 计算邻域余弦相似度（每点）
        local_vectors_orig = knn_pts_orig - U_orig_knn.unsqueeze(2)
        local_vectors_recon = knn_pts_recon - U_orig_knn.unsqueeze(2)
        flat_vectors_orig = local_vectors_orig.reshape(B, N, -1)
        flat_vectors_recon = local_vectors_recon.reshape(B, N, -1)
        cos_sim = F.cosine_similarity(flat_vectors_orig, flat_vectors_recon, dim=2)  # [B, N]
        knn_angle_mean = (1.0 - cos_sim).mean(dim=1)  # [B]

        return knn_dist_mean, knn_angle_mean

    def _visualize_iteration_comparison(self, ori_data, U_orig, snapshots, vis_dir):
        """生成迭代对比图：左列显示原始点云 + 标准球，右侧每列对应一次迭代的
        对抗样本 + 球形自编码器映射结果，对抗样本与球形之间显示 CD/HD 数值。

        Args:
            ori_data: 原始点云 [B, 3, N]
            U_orig: 标准球面点云 [B, 3, N]
            snapshots: dict, {iteration: {'adv_data', 'U_recon', 'cd', 'hd'}}
            vis_dir: 输出目录
        """
        ori_np = self._to_bnc_numpy(ori_data)
        U_orig_np = self._to_bnc_numpy(U_orig)

        sorted_iters = sorted(snapshots.keys())
        num_snaps = len(sorted_iters)
        B = ori_np.shape[0]  # batch size

        for sample_idx in range(B):
            cols = 1 + num_snaps  # clean + snapshots
            fig = plt.figure(figsize=(5 * cols, 5), dpi=150)

            # --- 第 1 列：clean point cloud + standard sphere ---
            ax_top = fig.add_subplot(2, cols, 1, projection="3d")
            pts = ori_np[sample_idx]
            ax_top.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, c="#2f6df6", alpha=0.8, linewidths=0)
            ax_top.set_title("clean point cloud", fontsize=9)
            self._set_equal_3d_axes(ax_top, pts, limit=None)
            ax_top.view_init(elev=18, azim=35)

            ax_bot = fig.add_subplot(2, cols, 1 + cols, projection="3d")
            pts_s = U_orig_np[sample_idx]
            ax_bot.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2], s=10, c="#2ca02c", alpha=0.8, linewidths=0)
            ax_bot.set_title("standard sphere", fontsize=9)
            self._set_equal_3d_axes(ax_bot, pts_s, limit=1.05)
            ax_bot.view_init(elev=18, azim=35)

            # --- 后续列：adv @ iter_X + mapped sphere ---
            for snap_col, iter_num in enumerate(sorted_iters):
                col_idx = snap_col + 2  # 1-indexed
                snap = snapshots[iter_num]
                adv_np = self._to_bnc_numpy(snap['adv_data'])
                U_rec_np = self._to_bnc_numpy(snap['U_recon'])
                cd_val = snap['cd'][sample_idx].item()
                hd_val = snap['hd'][sample_idx].item()

                # 上方：对抗样本
                ax_top = fig.add_subplot(2, cols, col_idx, projection="3d")
                pts_a = adv_np[sample_idx]
                ax_top.scatter(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2], s=10, c="#d62728", alpha=0.8, linewidths=0)
                ax_top.set_title(f"adv @ iter {iter_num}", fontsize=9)
                self._set_equal_3d_axes(ax_top, pts_a, limit=None)
                ax_top.view_init(elev=18, azim=35)

                # 下方：球形映射结果
                ax_bot = fig.add_subplot(2, cols, col_idx + cols, projection="3d")
                pts_r = U_rec_np[sample_idx]
                ax_bot.scatter(pts_r[:, 0], pts_r[:, 1], pts_r[:, 2], s=10, c="#9467bd", alpha=0.8, linewidths=0)
                ax_bot.set_title(f"mapped sphere\nCD={cd_val:.4f}  HD={hd_val:.4f}", fontsize=8)
                self._set_equal_3d_axes(ax_bot, pts_r, limit=1.05)
                ax_bot.view_init(elev=18, azim=35)

            plt.tight_layout()
            batch_idx = getattr(self, '_batch_idx', 0)
            out_path = os.path.join(vis_dir, f"comparison_b{batch_idx:04d}_s{sample_idx:02d}.png")
            os.makedirs(vis_dir, exist_ok=True)
            plt.savefig(pre_check_path(out_path), bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"[COMPARISON] saved {out_path}", flush=True)

    def _plot_metrics_correlation(self, vis_dir):
        """生成单坐标系折线图：CD、HD、球形局部 KNN 平均距离 vs 迭代次数。
        所有曲线绘制在同一坐标系内，KNN 角度 (1-cos) 不再单独成图。
        对当前 batch 内的每个样本各生成一张折线图。
        """
        if not self._metrics_history:
            return

        iterations = np.array([m['iteration'] for m in self._metrics_history])
        B = self._metrics_history[0]['cd'].shape[0]

        cd_arr = np.stack([m['cd'].cpu().numpy() for m in self._metrics_history], axis=1)      # [B, T]
        hd_arr = np.stack([m['hd'].cpu().numpy() for m in self._metrics_history], axis=1)      # [B, T]
        knn_d_arr = np.stack([m['knn_dist'].cpu().numpy() for m in self._metrics_history], axis=1)  # [B, T]

        batch_idx = getattr(self, '_batch_idx', 0)
        for sample_idx in range(B):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=120)
            fig.suptitle(
                f"Metrics vs Iteration — batch {batch_idx:04d} sample {sample_idx:02d}",
                fontsize=13,
            )

            ax.plot(iterations, cd_arr[sample_idx], 'b-o', markersize=3, linewidth=1.2,
                    label='Chamfer Distance (CD)')
            ax.plot(iterations, hd_arr[sample_idx], 'r-s', markersize=3, linewidth=1.2,
                    label='Hausdorff Distance (HD)')
            ax.plot(iterations, knn_d_arr[sample_idx], 'g-^', markersize=3, linewidth=1.2,
                    label='Sphere KNN Mean Distance')

            ax.set_xlabel("Iteration", fontsize=11)
            ax.set_ylabel("Distance", fontsize=11)
            ax.set_title("CD / HD / Sphere KNN Distance vs Iteration", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)

            plt.tight_layout()
            out_path = os.path.join(vis_dir, f"metrics_b{batch_idx:04d}_s{sample_idx:02d}.png")
            os.makedirs(vis_dir, exist_ok=True)
            plt.savefig(pre_check_path(out_path), bbox_inches="tight", dpi=120)
            plt.close(fig)
            print(f"[METRICS] saved {out_path}", flush=True)

    # ============================================================
    # [END] 可视化辅助方法
    # ============================================================

    def knn_latent_structural_loss(self, U_orig, U_recon, soft_mask, k=16):
        """
        [FIXED] 完全重写：使用可微软掩码与局部无序Chamfer约束
        """
        B, C, K_pts = U_recon.shape

        U_orig_knn = U_orig.transpose(1, 2).contiguous()  # [B, N, 3]
        U_recon_knn = U_recon.transpose(1, 2).contiguous()  # [B, K_pts, 3]

        # 1. 重建点云的局部邻居
        _, _, knn_pts_recon = knn_points(U_recon_knn, U_recon_knn, K=k, return_nn=True)  # [B, K_pts, k, 3]

        # 2. 从原始球体找到对应的局部参考结构
        _, _, knn_pts_orig = knn_points(U_recon_knn, U_orig_knn, K=k, return_nn=True)  # [B, K_pts, k, 3]

        # [FIXED] 避免展开(Flatten)导致的无序性坍塌，改为计算局部块之间的Chamfer Distance
        pts_recon = knn_pts_recon.view(B * K_pts, k, 3)
        pts_orig = knn_pts_orig.view(B * K_pts, k, 3)

        # 本地块的 pairwise L2 distance
        dist = torch.cdist(pts_recon, pts_orig, p=2) # [B*K_pts, k, k]
        min_dist1 = torch.min(dist, dim=2)[0] # [B*K_pts, k]
        min_dist2 = torch.min(dist.transpose(1, 2), dim=2)[0] # [B*K_pts, k]

        # 每个局部点域的形变代价
        local_cd = torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1) # [B*K_pts]
        local_cd = local_cd.view(B, K_pts)

        if self.sphere_angle_weight:
            match = dist.argmin(dim=2).view(B, K_pts, k)
            ref = knn_pts_orig.gather(2, match.unsqueeze(-1).expand(-1, -1, -1, 3))
            edge = knn_pts_recon - U_recon_knn.unsqueeze(2)
            ref_edge = ref - knn_pts_orig[:, :, :1]
            valid = (edge.norm(dim=-1) > 1e-8) & (ref_edge.norm(dim=-1) > 1e-8)
            angle = (1.0 - F.cosine_similarity(edge, ref_edge, dim=-1)) * valid
            local_cd = local_cd + self.sphere_angle_weight * (
                angle.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
            )

        # [FIXED] 应用软掩码 (全图处处可导)，返回 [B] 维度的 loss
        loss_per_batch = torch.sum(local_cd * soft_mask, dim=1) / (torch.sum(soft_mask, dim=1) + 1e-8)

        return loss_per_batch

    def attack(self, data, target, *, support_mask=None):
        self.losses = {
            'adv_loss': [], 'dist_loss': [], 'chamfer_loss': [],
            'transformation_loss': [], 'hide_loss': [], 'latent_space_loss': []
        }
        # [NEW] 用于迭代对比图与折线图的数据收集
        self._snapshots = {}  # iter -> {adv_data, U_recon, cd, hd}
        self._metrics_history = []  # list of {iteration, cd, hd, knn_dist, knn_angle}
        self.diagnostics = {'iterations': [], 'conflict_iterations': [], 'pointwise': [], 'config': {
            'constraint_space': self.constraint_space,
            'mask_mode': self.mask_mode,
            'latent_reference': self.latent_reference,
            'candidate_rule': self.candidate_rule,
            'point_reference': self.point_reference,
            'constraint_kind': self.constraint_kind,
            'mapping_mode': self.mapping_mode,
            'distance_mode': self.distance_mode,
            'soft_temperature': self.soft_temperature,
            'center_selection_mode': self.center_selection_mode,
            'intervention_mode': self.intervention_mode,
            'intervention_gate': self.intervention_gate,
            'centroid_align': self.centroid_align,
        }, 'centroid_shift': []}

        B, K = data.shape[:2]  # [B, N, 6]
        ori_data = data[:, :, :3].float().to(self.device).clone().detach()
        ori_normal = data[:, :, 3:].float().to(self.device).clone().detach()

        ori_data = ori_data.transpose(1, 2).contiguous()  # [B, 3, N]
        ori_normal = ori_normal.transpose(1, 2).contiguous()

        ori_data.requires_grad = False
        target = target.long().to(self.device).detach()
        fixed_geometry = (clean_geometry(ori_data, ori_normal, k=self.curv_loss_knn)
                          if self.conflict_diagnostics or self.center_selection_mode != 'original'
                          or self.intervention_mode != 'none' or self.normal_weight != 0 else None)

        adaptive = self.perturbation_kernel == 'adaptive_gradient'
        if support_mask is not None:
            if not adaptive or support_mask.shape != (B, K) or support_mask.dtype != torch.bool:
                raise ValueError('Expected a boolean [batch, points] support mask for adaptive gradient.')
            support = support_mask.to(self.device)
            if not support.any(dim=1).all():
                raise ValueError('The support mask must select points for each sample.')
        else:
            ori_kappa_std = self._get_kappa_std_ori(ori_data, ori_normal, k=self.curv_loss_knn)
            grad, success_num = self.get_gradient(ori_data, target)

            with torch.no_grad():
                center = torch.median(ori_data, dim=-1)[0].clone().detach()
                r = torch.sum((ori_data - center[:, :, None]) ** 2, dim=1) ** 0.5
                saliency = (-1.0 * (r ** (self.alpha)) * torch.sum((ori_data - center[:, :, None]) * grad, dim=1))
                saliency_min = saliency.min(dim=1, keepdim=True)[0]
                saliency_max = saliency.max(dim=1, keepdim=True)[0]
                std_min = ori_kappa_std.min(dim=1, keepdim=True)[0]
                std_max = ori_kappa_std.max(dim=1, keepdim=True)[0]
                normalized_saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-7)
                normalized_std = (ori_kappa_std - std_min) / (std_max - std_min + 1e-7)
                score = 0.001 * normalized_saliency + normalized_std

                far_points_idx = self.farthest_point_sample(ori_data.transpose(1, 2).contiguous(), self.total_central_num)
                far_points = self.index_points(ori_data.transpose(1, 2).contiguous(), far_points_idx)
                far_knn = knn_points(far_points, ori_data.permute(0, 2, 1), K=self.curv_loss_knn + 1)
                far_knn_points = knn_gather(ori_data.permute(0, 2, 1), far_knn.idx).permute(0, 3, 1, 2).contiguous()
                far_knn_score = self.index_points(score.unsqueeze(2), far_knn.idx)
                total_central_points_idx = far_knn_score.topk(k=1, dim=2)[1].squeeze(dim=-1)

                total_central_points = self.index_points(
                    far_knn_points.permute(0, 2, 3, 1).contiguous().view(-1, self.curv_loss_knn + 1, 3),
                    total_central_points_idx.view(-1, 1),
                )
                total_central_points = total_central_points.view(B, -1, 3).transpose(1, 2).contiguous()
                total_central_points_score = self.index_points(
                    far_knn_score.view(-1, self.curv_loss_knn + 1, 1),
                    total_central_points_idx.view(-1, 1),
                ).view(B, -1)

                central_points_score, tmp_idx = torch.topk(total_central_points_score, k=self.central_num)
                central_points = self.index_points(total_central_points.transpose(1, 2).contiguous(), tmp_idx).transpose(1, 2).contiguous()

                if self.center_selection_mode != 'original':
                    if fixed_geometry is None:
                        raise RuntimeError('Geometry fields are required for regional center selection')
                    complexity = fixed_geometry['complexity']
                    candidate_count = min(K, max(self.total_central_num, self.central_num))
                    if self.center_selection_mode in ('high_complexity', 'low_complexity'):
                        largest = self.center_selection_mode == 'high_complexity'
                        region_idx = complexity.topk(candidate_count, dim=1, largest=largest).indices
                        region_points = self.index_points(ori_data.transpose(1, 2), region_idx)
                        fps_local = self.farthest_point_sample(region_points, self.central_num)
                        chosen = self.index_points(region_idx.unsqueeze(2), fps_local).squeeze(2)
                    elif self.center_selection_mode == 'mixed':
                        high_n = self.central_num // 2
                        low_n = self.central_num - high_n
                        high_idx = complexity.topk(candidate_count, dim=1, largest=True).indices
                        low_idx = complexity.topk(candidate_count, dim=1, largest=False).indices
                        high_points = self.index_points(ori_data.transpose(1, 2), high_idx)
                        low_points = self.index_points(ori_data.transpose(1, 2), low_idx)
                        high_local = self.farthest_point_sample(high_points, high_n)
                        low_local = self.farthest_point_sample(low_points, low_n)
                        chosen = torch.cat([
                            self.index_points(high_idx.unsqueeze(2), high_local).squeeze(2),
                            self.index_points(low_idx.unsqueeze(2), low_local).squeeze(2)], dim=1)
                    else:
                        raise ValueError(f'Unknown center_selection_mode: {self.center_selection_mode}')
                    central_points = self.index_points(ori_data.transpose(1, 2), chosen).transpose(1, 2).contiguous()
                    central_points_score = complexity.gather(1, chosen)

        lower_bound = torch.zeros(B, device=self.device)
        initial_const = self.init_weight if self.fixed_scale_const is None else float(self.fixed_scale_const)
        scale_const = torch.ones(B, device=self.device) * initial_const
        upper_bound = torch.ones(B, device=self.device) * self.max_weight

        with torch.no_grad():
            U_standard = self.ae_model.decoder.sphere_points.unsqueeze(0).repeat(B, 1, 1).to(self.device)
            P_reconstruction, theta_U_orig_ref, _, _ = self.ae_model(ori_data)
            _, U_clean_mapping = self.ae_model.inverse(theta_U_orig_ref, ori_data)
            U_orig = U_standard if self.latent_reference == 'standard' else U_clean_mapping

        # weight factors for different distance loss
        haus_weight_v = self.haus_weight
        cd_weight_v = self.cd_weight
        ker_weight_v = self.ker_weight
        hide_weight_v = 0 if self.matched_displacement_control else self.hide_weight

        # [FIXED] 纯GPU张量化最佳记录器，防止CPU同步阻塞
        o_bestdist = torch.full((B,), 1e10, device=self.device)
        o_bestscore = torch.full((B,), -1, dtype=torch.long, device=self.device)
        o_bestattack = torch.zeros((B, 3, K), device=self.device)

        if support_mask is None:
            ori_kappa = self._get_kappa_ori(ori_data, ori_normal, k=self.curv_loss_knn)
            far_kappa_std = self.index_points(ori_kappa.unsqueeze(2), far_knn.idx)
            total_central_kappa_std = self.index_points(
                far_kappa_std.view(-1, self.curv_loss_knn + 1, 1),
                total_central_points_idx.view(-1, 1),
            ).view(B, -1, 1)
            central_kappa_std = self.index_points(total_central_kappa_std, tmp_idx)
            if self.center_selection_mode != 'original':
                center_nn = knn_points(central_points.transpose(1, 2), ori_data.transpose(1, 2), K=1).idx.squeeze(2)
                central_kappa_std = ori_kappa.gather(1, center_nn).unsqueeze(2)
        previous_diag_grads = None
        previous_nn = None
        previous_hd_witness = None
        if adaptive and support_mask is None:
            neighbors = knn_points(
                central_points.transpose(1, 2), ori_data.transpose(1, 2),
                K=min(self.curv_loss_knn, K),
            ).idx
            support = torch.zeros((B, K), dtype=torch.bool, device=self.device)
            support.scatter_(1, neighbors.flatten(1), True)
        dfs_gaussian = self.perturbation_kernel == 'dfs_gaussian'
        if dfs_gaussian:
            center_idx = knn_points(
                central_points.transpose(1, 2), ori_data.transpose(1, 2), K=1
            ).idx.squeeze(-1)
            same_point = center_idx[:, :, None] == center_idx[:, None, :]
            unique = ~same_point.tril(diagonal=-1).any(dim=2)
            # Only distinct GRASS-selected points can query or supply a perturbation.
            sources = unique[:, None, :] & ~same_point
            has_neighbors = sources.any(dim=2, keepdim=True)
            center_xyz = ori_data.gather(2, center_idx[:, None, :].expand(-1, 3, -1))
            center_dist2 = torch.cdist(center_xyz.transpose(1, 2), center_xyz.transpose(1, 2)).square()
            scatter_idx = center_idx[:, None, :].expand(-1, 3, -1)
            support = torch.zeros((B, K), dtype=torch.bool, device=self.device)
            support.scatter_(1, center_idx, True)

        # perform binary search
        for binary_step in range(self.binary_step):
            adv_data = ori_data.clone().detach()
            init_trans_scale = torch.tensor(self.budget)

            perturb_mat = (torch.rand(B, self.central_num, 3) * init_trans_scale).to(self.device)
            gauss_delta = torch.ones((B, self.central_num), device=self.device) * self.min_sigm + \
                          torch.rand((B, self.central_num), device=self.device) * (self.max_sigm - self.min_sigm)

            if self.matched_displacement_control:
                perturb_mat.zero_()
            if adaptive:
                perturb_mat = ori_data.new_zeros((B, K, 3))
            perturb_mat.requires_grad_()
            gauss_delta.requires_grad_(not adaptive)

            # [FIXED] GPU Tensor 记录当前迭代
            bestdist = torch.full((B,), 1e10, device=self.device)
            bestscore = torch.full((B,), -1, dtype=torch.long, device=self.device)

            opt = None if adaptive else optim.Adam([
                {"params": perturb_mat, "lr": self.attack_lr * 5},
                {"params": gauss_delta, "lr": self.attack_lr * 3},
            ], weight_decay=0.0)

            # one step in binary search
            for iteration in range(self.num_iter):
                with torch.no_grad():
                    perturb_mat.data = torch.clamp(perturb_mat.data, min=-self.budget, max=self.budget)
                    gauss_delta.data = torch.clamp(gauss_delta.data, min=self.min_sigm, max=self.max_sigm)

                if adaptive:
                    tmp_adv_data = adv_data + (perturb_mat * support.unsqueeze(-1)).transpose(1, 2)
                elif dfs_gaussian:
                    # Softmax is the normalized Gaussian regression over source points.
                    kernel_logits = -center_dist2 / (2 * gauss_delta[:, None, :].square())
                    kernel_logits = kernel_logits.masked_fill(~sources, -torch.inf)
                    kernel_logits = torch.where(has_neighbors, kernel_logits, 0.0)
                    weights = kernel_logits.softmax(dim=2).masked_fill(~sources, 0.0)
                    shifts = (weights @ perturb_mat) * unique.unsqueeze(-1)
                    tmp_adv_data = adv_data.scatter_add(2, scatter_idx, shifts.transpose(1, 2))
                else:
                    density = self.kernel_density(central_points, ori_data, gauss_delta)
                    denominator = density.sum(dim=1, keepdim=True) + 1e-8
                    weighted_shift = torch.einsum('bjc,bjk->bck', perturb_mat, density)
                    if self.perturbation_kernel == 'wendland_c2' or self.matched_displacement_control:
                        tmp_adv_data = adv_data + weighted_shift / denominator
                    else:
                        tmp_adv_data = (adv_data * density.sum(dim=1, keepdim=True) + weighted_shift) / denominator

                # ---- Centroid alignment -------------------------------------
                # Remove the global-translation degree of freedom so the attack
                # cannot "cheat" by rigidly shifting the whole cloud. We translate
                # every candidate cloud so its centroid matches the clean cloud's
                # centroid. This is a differentiable linear op (add/subtract a
                # per-sample constant vector), so Adam gradients still flow back
                # to perturb_mat/gauss_delta. Applied before forward pass so all
                # losses (adv/cd/hd/latent) and the saved o_bestattack see the
                # aligned, translation-free cloud.
                if self.centroid_align:
                    adv_centroid = tmp_adv_data.mean(dim=-1, keepdim=True)   # [B, 3, 1]
                    ori_centroid = ori_data.mean(dim=-1, keepdim=True)       # [B, 3, 1]
                    centroid_shift = (adv_centroid - ori_centroid).squeeze(-1)  # [B, 3]
                    tmp_adv_data = tmp_adv_data - (adv_centroid - ori_centroid)
                    diag_interval = max(1, int(self.diagnostic_interval))
                    if binary_step == 0 and (iteration % diag_interval == 0 or iteration == self.num_iter - 1):
                        self.diagnostics['centroid_shift'].append({
                            'iteration': iteration,
                            'mean_norm': float(centroid_shift.norm(dim=-1).mean().detach().cpu()),
                            'max_norm': float(centroid_shift.norm(dim=-1).max().detach().cpu()),
                        })
                # ---- End centroid alignment ----------------------------------

                # forward passing
                logits = self.model.forward(tmp_adv_data.permute([0, 2, 1]))["logit"]
                pred = torch.argmax(logits, dim=1)

                # [FIXED] 计算严格具有 [B] 维度的各类失真损失
                if self.matched_displacement_control:
                    transform_val_batch = (tmp_adv_data - ori_data).norm(dim=1).mean(dim=1)
                else:
                    transform_val_batch = self.transformation_loss(tmp_adv_data, perturb_mat, gauss_delta, batch_avg=False)
                cd_candidate_batch, hd_batch = self.distance_terms_batch(tmp_adv_data, ori_data)
                if self.candidate_rule == 'transform':
                    dist_val_batch = transform_val_batch
                elif self.candidate_rule == 'cd':
                    dist_val_batch = cd_candidate_batch
                elif self.candidate_rule == 'cd_hd':
                    # distance_terms_batch returns unsquared distances, so the
                    # weight is applied in the same scale as the CD term.
                    dist_val_batch = cd_candidate_batch + self.candidate_hd_weight * hd_batch
                elif self.candidate_rule == 'point_l2':
                    # Same best-snapshot criterion as ManifoldAdv: among all
                    # successful snapshots keep the smallest mean point-wise
                    # Euclidean displacement. This correlates directly with
                    # CD/HD instead of the kernel parameter norm.
                    dist_val_batch = (tmp_adv_data - ori_data).norm(dim=1).mean(dim=1)
                else:
                    raise ValueError(f'Unknown candidate_rule {self.candidate_rule}')

                # [FIXED] 纯GPU级别的矢量化状态更新，彻底移除 for CPU循环
                with torch.no_grad():
                    improve_mask = (dist_val_batch < bestdist) & (pred != target)
                    bestdist[improve_mask] = dist_val_batch[improve_mask]
                    bestscore[improve_mask] = pred[improve_mask]

                    o_improve_mask = (dist_val_batch < o_bestdist) & (pred != target)
                    o_bestdist[o_improve_mask] = dist_val_batch[o_improve_mask]
                    o_bestscore[o_improve_mask] = pred[o_improve_mask]
                    o_bestattack[o_improve_mask] = tmp_adv_data.detach()[o_improve_mask]

                # adv_loss is [B] tensor
                adv_loss_batch = self.adv_func(logits, target)
                dist_loss_batch = torch.zeros(B, device=self.device)

                # --- 潜空间局部结构软掩码计算 ---
                latent_space_loss_batch = torch.zeros(B, device=self.device)
                U_recon = None  # [NEW] 在需要时填充
                if self.latent_loss_weight > 0 and self.constraint_space != 'none':
                    # [FIXED] 计算可微的 soft_mask，保留反向传播能力
                    displacement = torch.norm(tmp_adv_data - ori_data, p=2, dim=1)  # [B, K]
                    max_disp = displacement.max(dim=1, keepdim=True)[0]
                    soft_mask = displacement / (max_disp + 1e-8)
                    if self.mask_mode == 'uniform':
                        soft_mask = torch.ones_like(soft_mask)
                    elif self.mask_mode == 'detach':
                        soft_mask = soft_mask.detach()
                    elif self.mask_mode != 'grad':
                        raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

                    if self.constraint_space == 'sphere':
                        if self.mapping_mode == 'nonlinear':
                            _, U_recon = self.ae_model.inverse(theta_U_orig_ref, tmp_adv_data)
                        elif self.mapping_mode == 'linearized':
                            def inverse_sphere(p):
                                return self.ae_model.inverse(theta_U_orig_ref, p)[1]
                            _, U_delta = torch.func.jvp(
                                inverse_sphere, (ori_data,), (tmp_adv_data - ori_data,))
                            U_recon = U_clean_mapping + U_delta
                        else:
                            raise ValueError(f"Unknown mapping_mode: {self.mapping_mode}")
                        constraint_ref = U_orig
                    elif self.constraint_space == 'point':
                        U_recon = tmp_adv_data
                        constraint_ref = (ori_data if self.point_reference == 'clean'
                                          else P_reconstruction.detach())
                    else:
                        raise ValueError(f"Unknown constraint_space: {self.constraint_space}")
                    if self.constraint_kind == 'radial':
                        radial = (torch.linalg.vector_norm(U_recon, dim=1) - 1.0).abs()
                        latent_space_loss_batch = (radial * soft_mask).sum(1) / (soft_mask.sum(1) + 1e-8)
                    else:
                        if self.constraint_kind == 'projected_local':
                            U_recon = F.normalize(U_recon, p=2, dim=1)
                            constraint_ref = F.normalize(constraint_ref, p=2, dim=1)
                        elif self.constraint_kind != 'local':
                            raise ValueError(f"Unknown constraint_kind: {self.constraint_kind}")
                        latent_space_loss_batch = self.knn_latent_structural_loss(
                            constraint_ref, U_recon, soft_mask, k=self.curv_loss_knn)

                    dist_loss_batch += latent_space_loss_batch * self.latent_loss_weight

                    # [NEW] 收集迭代快照与指标，仅在第一个 binary_step 收集（避免重复）
                    if self.generate_comparison and binary_step == 0:
                        snapshot_iters = getattr(self, 'snapshot_iters', [10, 20, 40])
                        with torch.no_grad():
                            adv_bnc = tmp_adv_data.detach().transpose(1, 2)  # [B, K, 3]
                            ori_bnc = ori_data.detach().transpose(1, 2)    # [B, K, 3]
                            chamfer_dist_obj = ChamferDist()
                            hausdorff_dist_obj = HausdorffDist()
                            cd_vals = chamfer_dist_obj(adv_bnc, ori_bnc, batch_avg=False)  # [B]
                            hd_vals = hausdorff_dist_obj(adv_bnc, ori_bnc, batch_avg=False)  # [B]
                            knn_d, knn_a = self.compute_sphere_knn_metrics(U_orig, U_recon, k=16)
                        # 收集快照
                        if iteration in snapshot_iters:
                            self._snapshots[iteration] = {
                                'adv_data': tmp_adv_data.detach().clone(),  # [B, 3, K]
                                'U_recon': U_recon.detach().clone(),        # [B, 3, N]
                                'cd': cd_vals,
                                'hd': hd_vals,
                            }
                            print(
                                f"[SNAPSHOT] b{binary_step} iter {iteration} saved "
                                f"CD={cd_vals.mean().item():.4f} HD={hd_vals.mean().item():.4f}",
                                flush=True,
                            )
                        # 收集每步指标
                        self._metrics_history.append({
                            'iteration': iteration,
                            'cd': cd_vals,         # [B]
                            'hd': hd_vals,         # [B]
                            'knn_dist': knn_d,     # [B]
                            'knn_angle': knn_a,    # [B]
                        })
                # [END NEW]

                # 添加其余失真函数 (全保障 [B] 输出)
                if haus_weight_v != 0:
                    dist_loss_batch += hd_batch * haus_weight_v
                if cd_weight_v != 0:
                    dist_loss_batch += cd_candidate_batch * cd_weight_v
                if self.normal_weight != 0:
                    normal_loss_batch = self.fixed_normal_structure_loss(
                        tmp_adv_data, ori_data, ori_normal, fixed_geometry['idx'])
                    dist_loss_batch += normal_loss_batch * float(self.normal_weight)
                if ker_weight_v != 0:
                    dist_loss_batch += transform_val_batch * ker_weight_v
                if hide_weight_v != 0:
                    # curv_std_loss 返回的是 [B, central_num], mean(1) -> [B]
                    hide_loss_batch = self.curv_std_loss(gauss_delta, central_kappa_std, self.max_sigm, self.min_sigm)
                    dist_loss_batch += hide_loss_batch * hide_weight_v

                # [FIXED] 完美C&W聚合机制：(批次对抗损失 + 按样本拉伸的几何损失).mean()
                loss = (adv_loss_batch + scale_const * dist_loss_batch).mean()

                diagnostic_interval = max(1, int(self.diagnostic_interval))
                take_diagnostic = (self.record_diagnostics and
                                   (iteration == 0 or iteration == self.num_iter - 1 or
                                    iteration % diagnostic_interval == 0))
                take_conflict = (self.conflict_diagnostics and
                                 (iteration == 0 or iteration == self.num_iter - 1 or
                                  iteration % max(1, int(self.conflict_interval)) == 0))
                conflict_rows = None
                nn_switch_gate = None
                if take_diagnostic:
                    adv_grads = torch.autograd.grad(adv_loss_batch.mean(), (perturb_mat, gauss_delta),
                                                    retain_graph=True, allow_unused=True)
                    geom_grads = torch.autograd.grad(dist_loss_batch.mean(), (perturb_mat, gauss_delta),
                                                     retain_graph=True, allow_unused=True)
                    cd_grads = torch.autograd.grad(cd_candidate_batch.mean(), (perturb_mat, gauss_delta),
                                                   retain_graph=True, allow_unused=True)
                    latent_grads = (torch.autograd.grad(latent_space_loss_batch.mean(),
                                                        (perturb_mat, gauss_delta), retain_graph=True,
                                                        allow_unused=True)
                                    if latent_space_loss_batch.requires_grad else (None, None))
                    def _norm(g):
                        return 0.0 if g is None else float(g.detach().norm().cpu())
                    def _cos(g1, g2):
                        if g1 is None or g2 is None:
                            return float('nan')
                        return float(F.cosine_similarity(g1.reshape(1, -1), g2.reshape(1, -1)).detach().cpu())
                    current_nn = torch.cdist(tmp_adv_data.transpose(1, 2), ori_data.transpose(1, 2)).argmin(2)
                    nn_switch = (float('nan') if previous_nn is None else
                                 float((current_nn != previous_nn).float().mean().detach().cpu()))
                    grad_stability_translation = (float('nan') if previous_diag_grads is None else
                                                  _cos(geom_grads[0], previous_diag_grads[0]))
                    self.diagnostics['iterations'].append({
                        'binary_step': binary_step, 'iteration': iteration,
                        'adv_loss': float(adv_loss_batch.mean().detach().cpu()),
                        'geometry_loss': float(dist_loss_batch.mean().detach().cpu()),
                        'latent_loss': float(latent_space_loss_batch.mean().detach().cpu()),
                        'cd': float(cd_candidate_batch.mean().detach().cpu()),
                        'transform': float(transform_val_batch.mean().detach().cpu()),
                        'scale_const_mean': float(scale_const.mean().detach().cpu()),
                        'grad_adv_translation': _norm(adv_grads[0]),
                        'grad_adv_scale': _norm(adv_grads[1]),
                        'grad_geom_translation': _norm(geom_grads[0]),
                        'grad_geom_scale': _norm(geom_grads[1]),
                        'attack_geometry_cos_translation': _cos(adv_grads[0], geom_grads[0]),
                        'attack_geometry_cos_scale': _cos(adv_grads[1], geom_grads[1]),
                        'latent_cd_cos_translation': _cos(latent_grads[0], cd_grads[0]),
                        'latent_cd_cos_scale': _cos(latent_grads[1], cd_grads[1]),
                        'geometry_grad_stability_translation': grad_stability_translation,
                        'nearest_neighbor_switch_rate': nn_switch,
                    })
                    previous_diag_grads = tuple(None if g is None else g.detach().clone() for g in geom_grads)
                if take_conflict:
                    # Point-space gradients remain separated by sample because every
                    # batch loss is separable before the final mean.
                    adv_x = torch.autograd.grad(adv_loss_batch.sum(), tmp_adv_data,
                                                retain_graph=True)[0]
                    cd_x = torch.autograd.grad(cd_candidate_batch.sum(), tmp_adv_data,
                                               retain_graph=True)[0]
                    hd_x = torch.autograd.grad(hd_batch.sum(), tmp_adv_data,
                                               retain_graph=True)[0]
                    latent_x = (torch.autograd.grad(latent_space_loss_batch.sum(), tmp_adv_data,
                                                    retain_graph=True)[0]
                                if latent_space_loss_batch.requires_grad else torch.zeros_like(tmp_adv_data))
                    point_geom_x = (cd_weight_v * cd_x + haus_weight_v * hd_x +
                                    self.latent_loss_weight * latent_x)
                    corr = correspondence_metrics(tmp_adv_data, ori_data, fixed_geometry,
                                                  previous_nn, previous_hd_witness)
                    nn_switch_gate = torch.nan_to_num(corr['nn_switch_rate'], nan=0.0) > 0
                    one_hot = F.one_hot(target, num_classes=logits.shape[1]).bool()
                    real = logits.gather(1, target[:, None]).squeeze(1)
                    other = logits.masked_fill(one_hot, -1e9).max(1).values
                    displacement = (tmp_adv_data - ori_data).norm(dim=1)
                    high = fixed_geometry['high_mask'].float(); low = fixed_geometry['low_mask'].float()
                    base_values = {
                        'margin': other - real,
                        'success_now': (pred != target).float(),
                        'cd': cd_candidate_batch, 'hd': hd_batch,
                        'transform': transform_val_batch,
                        'latent': latent_space_loss_batch,
                        'scale_const': scale_const,
                        'sigma_lower_fraction': (gauss_delta <= self.min_sigm + 1e-6).float().mean(1),
                        'sigma_upper_fraction': (gauss_delta >= self.max_sigm - 1e-6).float().mean(1),
                        'high_complexity_displacement_energy':
                            (displacement.square() * high).sum(1) / high.sum(1).clamp_min(1),
                        'low_complexity_displacement_energy':
                            (displacement.square() * low).sum(1) / low.sum(1).clamp_min(1),
                    }
                    for prefix, grad_x in [('cd', cd_x), ('hd', hd_x),
                                           ('sphere', latent_x), ('point_geometry', point_geom_x)]:
                        for key, value in conflict_metrics(adv_x, grad_x, fixed_geometry).items():
                            base_values[f'{prefix}_{key}'] = value
                    for key, value in corr.items():
                        if torch.is_tensor(value) and value.ndim == 1:
                            base_values[key] = value
                    ops = operator_losses(tmp_adv_data, ori_data, temperature=self.soft_temperature)
                    for key, value in ops.items():
                        if torch.is_tensor(value) and value.ndim == 1:
                            base_values[f'operator_{key}'] = value
                    conflict_rows = tensor_dict_to_records(base_values)
                    for sample_idx, row in enumerate(conflict_rows):
                        row.update({'binary_step': binary_step, 'iteration': iteration,
                                    'sample_in_batch': sample_idx,
                                    'center_selection_mode': self.center_selection_mode,
                                    'intervention_mode': self.intervention_mode,
                                    'intervention_gate': self.intervention_gate})
                        self.diagnostics['conflict_iterations'].append(row)
                    if self.record_pointwise:
                        limit = min(B, int(self.pointwise_sample_limit))
                        for sample_idx in range(limit):
                            self.diagnostics['pointwise'].append({
                                'binary_step': binary_step, 'iteration': iteration,
                                'sample_in_batch': sample_idx,
                                'adv_grad': adv_x[sample_idx].detach().to(torch.float16).cpu(),
                                'geometry_grad': point_geom_x[sample_idx].detach().to(torch.float16).cpu(),
                                'displacement': (tmp_adv_data[sample_idx] - ori_data[sample_idx]).detach().to(torch.float16).cpu(),
                                'complexity': fixed_geometry['complexity'][sample_idx].detach().to(torch.float16).cpu(),
                                'surface_variation': fixed_geometry['surface_variation'][sample_idx].detach().to(torch.float16).cpu(),
                                'normal_dispersion': fixed_geometry['normal_dispersion'][sample_idx].detach().to(torch.float16).cpu(),
                            })
                    previous_nn = corr['a2c'].detach().clone()
                    previous_hd_witness = corr['hd_witness'].detach().clone()
                elif take_diagnostic:
                    previous_nn = current_nn.detach().clone()

                if adaptive:
                    gradient, = torch.autograd.grad(loss, perturb_mat)
                    if iteration in (0, self.num_iter - 1) and not gradient.isfinite().all():
                        raise RuntimeError('Non-finite adaptive gradient.')
                    with torch.no_grad():
                        # AL-Adv distributes the step across axes by absolute gradient magnitude.
                        direction = gradient / gradient.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        perturb_mat.add_(direction, alpha=-self.attack_lr * 5)
                        perturb_mat.clamp_(-self.budget, self.budget)
                    continue

                opt.zero_grad()
                pre_perturb = perturb_mat.detach().clone() if conflict_rows is not None else None
                pre_sigma = gauss_delta.detach().clone() if conflict_rows is not None else None
                if self.intervention_mode == 'none':
                    loss.backward()
                else:
                    params = (perturb_mat, gauss_delta)
                    total_grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    adv_param = torch.autograd.grad(adv_loss_batch.mean(), params,
                                                    retain_graph=True, allow_unused=True)
                    weighted_geom = (scale_const * dist_loss_batch).mean()
                    geom_param = torch.autograd.grad(weighted_geom, params,
                                                     retain_graph=True, allow_unused=True)
                    gate = torch.ones(B, dtype=torch.bool, device=self.device)
                    if self.intervention_gate == 'pre_success': gate = pred == target
                    elif self.intervention_gate == 'post_success': gate = pred != target
                    elif self.intervention_gate == 'complexity_dominant':
                        disp2 = (tmp_adv_data - ori_data).square().sum(1)
                        gate = ((disp2 * fixed_geometry['high_mask']).sum(1) /
                                fixed_geometry['high_mask'].sum(1).clamp_min(1) >
                                (disp2 * fixed_geometry['low_mask']).sum(1) /
                                fixed_geometry['low_mask'].sum(1).clamp_min(1))
                    elif self.intervention_gate == 'switch_event':
                        gate = (torch.zeros(B, dtype=torch.bool, device=self.device)
                                if nn_switch_gate is None else nn_switch_gate)
                    elif self.intervention_gate != 'always':
                        raise ValueError(f'Unknown intervention_gate: {self.intervention_gate}')

                    corrected = list(total_grads)
                    if self.intervention_mode in ('all', 'cd', 'hd', 'translation', 'scale'):
                        if self.intervention_mode == 'cd':
                            component_loss = (scale_const * cd_candidate_batch * cd_weight_v).mean()
                            component = torch.autograd.grad(component_loss, params, retain_graph=True, allow_unused=True)
                        elif self.intervention_mode == 'hd':
                            component_loss = (scale_const * hd_batch * haus_weight_v).mean()
                            component = torch.autograd.grad(component_loss, params, retain_graph=True, allow_unused=True)
                        else:
                            component = geom_param
                        selected = (0,) if self.intervention_mode == 'translation' else ((1,) if self.intervention_mode == 'scale' else (0, 1))
                        for pi in selected:
                            if component[pi] is None or adv_param[pi] is None: continue
                            old = component[pi]
                            dot = (old * adv_param[pi]).flatten(1).sum(1)
                            coeff = dot.clamp_max(0) / (adv_param[pi].square().flatten(1).sum(1) + 1e-12)
                            coeff_view = coeff.view(B, *([1] * (old.ndim - 1)))
                            mod = old - coeff_view * adv_param[pi]
                            if self.restore_intervention_norm:
                                mod = mod * (old.flatten(1).norm(dim=1) /
                                             (mod.flatten(1).norm(dim=1) + 1e-12)).view(B, *([1] * (old.ndim - 1)))
                            sample_mask = gate.view(B, *([1] * (old.ndim - 1)))
                            mod = torch.where(sample_mask, mod, old)
                            corrected[pi] = total_grads[pi] + mod - old
                    else:
                        adv_x = torch.autograd.grad(adv_loss_batch.mean(), tmp_adv_data, retain_graph=True)[0]
                        point_loss = (scale_const * (cd_weight_v * cd_candidate_batch +
                                     haus_weight_v * hd_batch +
                                     self.latent_loss_weight * latent_space_loss_batch)).mean()
                        geom_x = torch.autograd.grad(point_loss, tmp_adv_data, retain_graph=True)[0]
                        mod_x = remove_negative_component(adv_x, geom_x, fixed_geometry,
                                                          self.intervention_mode,
                                                          self.restore_intervention_norm)
                        mod_x = torch.where(gate[:, None, None], mod_x, geom_x)
                        delta_param = torch.autograd.grad(tmp_adv_data, params,
                                                         grad_outputs=mod_x - geom_x,
                                                         retain_graph=True, allow_unused=True)
                        corrected = [g if d is None else g + d for g, d in zip(total_grads, delta_param)]
                    for parameter, gradient in zip(params, corrected):
                        parameter.grad = gradient
                opt.step()
                if conflict_rows is not None:
                    dperturb = perturb_mat.detach() - pre_perturb
                    dsigma = gauss_delta.detach() - pre_sigma
                    for sample_idx, row in enumerate(conflict_rows):
                        row['adam_translation_update_norm'] = float(dperturb[sample_idx].norm().cpu())
                        row['adam_scale_update_norm'] = float(dsigma[sample_idx].norm().cpu())

                # 日志记录均值
                self.losses['adv_loss'].append(adv_loss_batch.mean().item())
                self.losses['dist_loss'].append(dist_loss_batch.mean().item())
                self.losses['latent_space_loss'].append(latent_space_loss_batch.mean().item())

            # 二分查找：自动调节边界
            if self.fixed_scale_const is None:
                for e, label in enumerate(target):
                    if bestscore[e] != label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                        lower_bound[e] = max(lower_bound[e], scale_const[e])
                        scale_const[e] = (lower_bound[e] + upper_bound[e]) / 2.0
                    else:
                        upper_bound[e] = min(upper_bound[e], scale_const[e])
                        scale_const[e] = (lower_bound[e] + upper_bound[e]) / 2.0
            else:
                lower_bound[bestscore != -1] = float(self.fixed_scale_const)

        # [NEW] 生成迭代对比图与折线图
        if self.generate_comparison and self._snapshots:
            self._visualize_iteration_comparison(
                ori_data=ori_data,
                U_orig=U_orig,
                snapshots=self._snapshots,
                vis_dir=self.sphere_vis_dir,
            )
        if self.generate_comparison and self._metrics_history:
            self._plot_metrics_correlation(vis_dir=self.sphere_vis_dir)

        # 处理完全失败的特例
        fail_idx = (lower_bound == 0.0)
        o_bestattack[fail_idx] = ori_data.detach()[fail_idx]
        if (adaptive or dfs_gaussian) and torch.any((o_bestattack != ori_data) & ~support[:, None, :]):
            raise RuntimeError('Perturbation moved a point outside the selected support.')

        success_num = (lower_bound > 0.0).sum().item()
        print("Successfully attack {}/{}".format(success_num, B))

        return o_bestattack.transpose(1, 2).cpu().numpy(), success_num

    # [FIXED] 增加对Batch的支持：Chamfer 和 Hausdorff 计算
    def distance_terms_batch(self, points1, points2):
        """Compute Chamfer and Hausdorff from one shared pairwise matrix."""
        if points1.shape[1] == 3:
            points1 = points1.transpose(1, 2)
            points2 = points2.transpose(1, 2)
        dist = torch.cdist(points1, points2, p=2)
        if self.distance_mode == 'smooth':
            tau = self.soft_temperature
            near1 = -tau * torch.logsumexp(-dist / tau, dim=2) + tau * np.log(dist.shape[2])
            near2 = -tau * torch.logsumexp(-dist.transpose(1, 2) / tau, dim=2) + tau * np.log(dist.shape[1])
            cd = near1.mean(1) + near2.mean(1)
            max1 = tau * torch.logsumexp(near1 / tau, dim=1) - tau * np.log(near1.shape[1])
            max2 = tau * torch.logsumexp(near2 / tau, dim=1) - tau * np.log(near2.shape[1])
            hd = tau * torch.logsumexp(torch.stack([max1, max2], 1) / tau, dim=1) - tau * np.log(2.0)
        else:
            near1 = dist.min(dim=2)[0]
            near2 = dist.min(dim=1)[0]
            cd = near1.mean(1) + near2.mean(1)
            hd = torch.maximum(near1.max(1)[0], near2.max(1)[0])
        return cd, hd

    def chamfer_distance_batch(self, points1, points2):
        """返回 [B] 形状的 Chamfer 距离"""
        if points1.shape[1] == 3:  # [B, 3, N] -> [B, N, 3]
            points1 = points1.transpose(1, 2)
            points2 = points2.transpose(1, 2)
        dist1 = torch.cdist(points1, points2, p=2)
        if self.distance_mode == 'smooth':
            tau = self.soft_temperature
            min_dist1 = -tau * torch.logsumexp(-dist1 / tau, dim=2) + tau * np.log(dist1.shape[2])
            min_dist2 = -tau * torch.logsumexp(-dist1.transpose(1, 2) / tau, dim=2) + tau * np.log(dist1.shape[1])
        else:
            min_dist1 = torch.min(dist1, dim=2)[0]
            min_dist2 = torch.min(dist1.transpose(1, 2), dim=2)[0]
        return torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)

    def hausdorff_distance_batch(self, points1, points2):
        """返回 [B] 形状的 Hausdorff 距离"""
        if points1.shape[1] == 3:
            points1 = points1.transpose(1, 2)
            points2 = points2.transpose(1, 2)
        dist1 = torch.cdist(points1, points2, p=2)
        if self.distance_mode == 'smooth':
            tau = self.soft_temperature
            near1 = -tau * torch.logsumexp(-dist1 / tau, dim=2) + tau * np.log(dist1.shape[2])
            near2 = -tau * torch.logsumexp(-dist1.transpose(1, 2) / tau, dim=2) + tau * np.log(dist1.shape[1])
            max1 = tau * torch.logsumexp(near1 / tau, dim=1) - tau * np.log(near1.shape[1])
            max2 = tau * torch.logsumexp(near2 / tau, dim=1) - tau * np.log(near2.shape[1])
            return tau * torch.logsumexp(torch.stack([max1, max2], 1) / tau, dim=1) - tau * np.log(2.0)
        min_dist1 = torch.max(torch.min(dist1, dim=2)[0], dim=1)[0]
        min_dist2 = torch.max(torch.min(dist1.transpose(1, 2), dim=2)[0], dim=1)[0]
        return torch.max(min_dist1, min_dist2)

    def kernel_density(self, central_points, pc, delta):
        # Broadcasting avoids materializing two large repeated tensors.
        norm = torch.linalg.vector_norm(
            pc.unsqueeze(3) - central_points.unsqueeze(2), dim=1)  # [B, K, central_num]
        scale = (2 * delta.square()).unsqueeze(1)
        if self.perturbation_kernel == 'wendland_c2':
            # Match the half-height distance of the original exponential kernel.
            radius = 2.2088104396151227 * scale
            r = norm / radius
            density = (1 - r).clamp_min(0).pow(4) * (1 + 4 * r)
        else:
            density = torch.exp(-norm / scale)
        return density.transpose(1, 2).contiguous()

    def transformation_loss(self, adv_data, perturb_mat, gauss_delta, batch_avg=True):
        B, _, N = adv_data.shape
        if batch_avg:
            transformation_loss = torch.tensor(0.0).to(self.device)
            transformation_loss += torch.norm(perturb_mat)
            transformation_loss += 1 * (torch.norm(1 - gauss_delta))
            return transformation_loss / self.central_num
        else:
            transformation_loss = torch.zeros((B)).to(self.device)
            transformation_loss += torch.norm(perturb_mat, dim=(1, 2))
            transformation_loss += 1 * (torch.norm(1 - gauss_delta, dim=1))
            return transformation_loss / self.central_num

    def _get_kappa_ori(self, pc, normal, k=2):
        b, _, n = pc.size()
        inter_KNN = knn_points(pc.permute(0, 2, 1), pc.permute(0, 2, 1), K=k + 1)
        nn_pts = knn_gather(pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :, 1:].contiguous()
        vectors = nn_pts - pc.unsqueeze(3)
        vectors = self._normalize(vectors)
        return torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)

    def fixed_normal_structure_loss(self, adv_pc, clean_pc, clean_normal, clean_knn_idx):
        """Per-sample preservation of local edge/normal angles.

        The k-NN graph and normals come only from the clean cloud.  This avoids
        moving-neighbour confounding and keeps the loss differentiable with
        respect to the attacked coordinates.  Absolute cosines make the term
        invariant to the arbitrary sign of estimated normals.
        """
        adv = adv_pc.transpose(1, 2).contiguous()
        clean = clean_pc.transpose(1, 2).contiguous()
        normal = F.normalize(clean_normal.transpose(1, 2).contiguous(), dim=2)
        adv_nb = knn_gather(adv, clean_knn_idx)
        clean_nb = knn_gather(clean, clean_knn_idx)
        adv_edge = F.normalize(adv_nb - adv[:, :, None, :], dim=3)
        clean_edge = F.normalize(clean_nb - clean[:, :, None, :], dim=3)
        adv_cos = (adv_edge * normal[:, :, None, :]).sum(3).abs()
        clean_cos = (clean_edge * normal[:, :, None, :]).sum(3).abs()
        return (adv_cos - clean_cos).abs().mean((1, 2))

    def _get_kappa_std_ori(self, pc, normal, k=10):
        b, _, n = pc.size()
        inter_KNN = knn_points(pc.permute(0, 2, 1), pc.permute(0, 2, 1), K=k + 1)
        nn_pts = knn_gather(pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :, 1:].contiguous()
        vectors = nn_pts - pc.unsqueeze(3)
        vectors = self._normalize(vectors)
        kappa_ori = torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)
        nn_kappa = knn_gather(kappa_ori.unsqueeze(2), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :, 1:].contiguous()
        std_kappa = torch.std(nn_kappa.squeeze(1), dim=2)
        return std_kappa

    def curv_std_loss(self, gauss_delta, central_kappa_std, max_delta, min_delta):
        std = central_kappa_std.squeeze(-1)
        std_min = std.min(dim=1, keepdim=True)[0]
        std_max = std.max(dim=1, keepdim=True)[0]
        norm_std = (std - std_min) / (std_max - std_min + 1e-7)
        norm_gauss_delta = (gauss_delta - min_delta) / (max_delta - min_delta + 1e-7)
        cos_sim = F.cosine_similarity(norm_std, norm_gauss_delta)
        return cos_sim

    def _normalize(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

    def index_points(self, points, idx):
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long, device=self.device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def farthest_point_sample(self, xyz, npoint):
        return sample_farthest_points(xyz, K=npoint)[1]

    def get_gradient(self, data, target):
        input_data = data.clone().detach().float().to(self.device)
        input_data.requires_grad_()
        target = target.long().to(self.device)
        logits = self.model.forward(input_data.permute([0, 2, 1]))["logit"]
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        with torch.no_grad():
            grad = input_data.grad.detach()
            pred = torch.argmax(logits, dim=-1)
            num = (pred != target).sum().detach().cpu().item()
        return grad, num
#retry
