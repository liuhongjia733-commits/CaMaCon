"""Diagnostics and causal interventions for attack/geometry conflicts.

All functions operate on Bx3xN tensors and are side-effect free.  They are
imported by ``backbone`` only when the phase-7--12 diagnostics are enabled.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_gather, knn_points


EPS = 1e-12


def _zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)


@torch.no_grad()
def clean_geometry(points: torch.Tensor, normals: torch.Tensor, k: int = 16) -> Dict[str, torch.Tensor]:
    """Compute a fixed clean-cloud frame and complexity fields.

    Returns PCA surface variation, normal dispersion, curvature statistics,
    density and within-sample high/low-complexity masks.  No attacked geometry
    is used, so region labels cannot leak information from the result.
    """
    p = points.transpose(1, 2).contiguous()
    n = F.normalize(normals.transpose(1, 2).contiguous(), dim=2)
    query = knn_points(p, p, K=k + 1)
    idx = query.idx[:, :, 1:]
    nb = knn_gather(p, idx)
    rel = nb - p[:, :, None, :]
    covariance = rel.transpose(-1, -2) @ rel / float(k)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    pca_normal = eigenvectors[..., 0]
    sign = torch.sign((pca_normal * n).sum(2, keepdim=True)).clamp(min=-1.0, max=1.0)
    sign[sign == 0] = 1.0
    pca_normal = pca_normal * sign

    # Stable tangent frame derived from the local normal.
    axis_x = torch.zeros_like(pca_normal); axis_x[..., 0] = 1.0
    axis_y = torch.zeros_like(pca_normal); axis_y[..., 1] = 1.0
    use_y = pca_normal[..., 0].abs() > 0.9
    axis = torch.where(use_y[..., None], axis_y, axis_x)
    tangent1 = F.normalize(torch.cross(pca_normal, axis, dim=2), dim=2)
    tangent2 = F.normalize(torch.cross(pca_normal, tangent1, dim=2), dim=2)

    n_nb = knn_gather(n, idx)
    normal_dispersion = (1.0 - (n_nb * n[:, :, None, :]).sum(3).abs()).mean(2)
    surface_variation = eigenvalues[..., 0] / (eigenvalues.sum(2) + EPS)
    unit_rel = F.normalize(rel, dim=3)
    curvature = (unit_rel * pca_normal[:, :, None, :]).sum(3).abs()
    curvature_mean = curvature.mean(2)
    curvature_std = curvature.std(2)
    density = rel.norm(dim=3).mean(2)
    complexity = 0.5 * (_zscore(surface_variation) + _zscore(normal_dispersion))
    high_q = torch.quantile(complexity, 0.75, dim=1, keepdim=True)
    low_q = torch.quantile(complexity, 0.25, dim=1, keepdim=True)
    high = complexity >= high_q
    low = complexity <= low_q
    return {
        'idx': idx, 'normal': pca_normal.transpose(1, 2).contiguous(),
        'tangent1': tangent1.transpose(1, 2).contiguous(),
        'tangent2': tangent2.transpose(1, 2).contiguous(),
        'surface_variation': surface_variation, 'normal_dispersion': normal_dispersion,
        'curvature_mean': curvature_mean, 'curvature_std': curvature_std,
        'density': density, 'complexity': complexity,
        'high_mask': high, 'low_mask': low,
    }


def neighbor_average(field: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    f = field.transpose(1, 2).contiguous()
    return knn_gather(f, idx).mean(2).transpose(1, 2).contiguous()


def project_field(field: torch.Tensor, geometry: Dict[str, torch.Tensor], component: str) -> torch.Tensor:
    """Return one named component of a Bx3xN vector field."""
    if component in ('all', 'none'):
        return field
    normal = geometry['normal']
    if component == 'normal':
        return (field * normal).sum(1, keepdim=True) * normal
    if component == 'tangent':
        return field - (field * normal).sum(1, keepdim=True) * normal
    if component == 'lowfreq':
        return neighbor_average(field, geometry['idx'])
    if component == 'highfreq':
        return field - neighbor_average(field, geometry['idx'])
    if component in ('high_complexity', 'low_complexity'):
        mask = geometry['high_mask'] if component == 'high_complexity' else geometry['low_mask']
        return field * mask[:, None, :]
    raise ValueError(f'Unknown field component: {component}')


def _flat_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).flatten(1).sum(1)


def conflict_metrics(adv_grad: torch.Tensor, geom_grad: torch.Tensor,
                     geometry: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Per-sample conflict measures in physical and regional subspaces."""
    out = {}
    for name in ('all', 'normal', 'tangent', 'lowfreq', 'highfreq',
                 'high_complexity', 'low_complexity'):
        ga = project_field(adv_grad, geometry, name)
        gg = project_field(geom_grad, geometry, name)
        dot = _flat_dot(ga, gg)
        na = ga.flatten(1).norm(dim=1)
        ng = gg.flatten(1).norm(dim=1)
        conflict = (-dot).clamp_min(0)
        # Cost incurred by moving along the attack-only descent direction.
        denom = ga.square().flatten(1).sum(1).clamp_min(EPS)
        ratio = conflict / denom
        out[f'{name}_dot'] = dot
        out[f'{name}_cosine'] = dot / (na * ng + EPS)
        out[f'{name}_conflict'] = conflict
        out[f'{name}_cost_ratio'] = ratio
        out[f'{name}_adv_energy'] = denom
    return out


def remove_negative_component(adv_grad: torch.Tensor, geom_grad: torch.Tensor,
                              geometry: Optional[Dict[str, torch.Tensor]] = None,
                              component: str = 'all', restore_norm: bool = True) -> torch.Tensor:
    """PCGrad-style removal in a selected physical subspace, per sample."""
    target_a = project_field(adv_grad, geometry, component) if geometry is not None else adv_grad
    target_g = project_field(geom_grad, geometry, component) if geometry is not None else geom_grad
    dot = _flat_dot(target_g, target_a)
    coeff = dot.clamp_max(0) / (target_a.square().flatten(1).sum(1) + EPS)
    corrected_target = target_g - coeff[:, None, None] * target_a
    corrected = geom_grad + corrected_target - target_g
    if restore_norm:
        old = geom_grad.flatten(1).norm(dim=1)
        new = corrected.flatten(1).norm(dim=1)
        corrected = corrected * (old / (new + EPS))[:, None, None]
    return corrected


@torch.no_grad()
def correspondence_metrics(adv: torch.Tensor, clean: torch.Tensor,
                           geometry: Dict[str, torch.Tensor],
                           previous_nn: Optional[torch.Tensor] = None,
                           previous_hd_witness: Optional[torch.Tensor] = None):
    """Nearest-neighbor and Hausdorff active-set diagnostics."""
    a = adv.transpose(1, 2).contiguous(); c = clean.transpose(1, 2).contiguous()
    dist = torch.cdist(a, c)
    a2c_dist, a2c = dist.min(2)
    c2a_dist, c2a = dist.min(1)
    b, n = a2c.shape
    collision = []
    mutual = []
    uncovered = []
    for j in range(b):
        collision.append(1.0 - a2c[j].unique().numel() / float(n))
        covered = torch.zeros(n, dtype=torch.bool, device=adv.device); covered[a2c[j].unique()] = True
        uncovered.append(1.0 - covered.float().mean().item())
        ar = torch.arange(n, device=adv.device)
        mutual.append((c2a[j, a2c[j]] == ar).float().mean().item())
    amax, ai = a2c_dist.max(1); cmax, ci = c2a_dist.max(1)
    side = cmax > amax
    witness = torch.where(side, ci + n, ai)
    high = geometry['high_mask']
    clean_index = torch.where(side, ci, a2c.gather(1, ai[:, None]).squeeze(1))
    high_witness = high.gather(1, clean_index[:, None]).squeeze(1).float()
    result = {
        'nn_switch_rate': (torch.full((b,), float('nan'), device=adv.device) if previous_nn is None
                           else (a2c != previous_nn).float().mean(1)),
        'collision_rate': torch.tensor(collision, device=adv.device),
        'uncovered_rate': torch.tensor(uncovered, device=adv.device),
        'mutual_nn_rate': torch.tensor(mutual, device=adv.device),
        'hd_witness_switch': (torch.full((b,), float('nan'), device=adv.device)
                              if previous_hd_witness is None else (witness != previous_hd_witness).float()),
        'hd_witness_in_high_complexity': high_witness,
        'a2c': a2c, 'hd_witness': witness,
    }
    return result


def operator_losses(adv: torch.Tensor, clean: torch.Tensor, temperature: float = 0.01,
                    frozen_a2c: Optional[torch.Tensor] = None,
                    frozen_c2a: Optional[torch.Tensor] = None,
                    fixed_witness: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Counterfactual CD/HD operators evaluated at one shared state."""
    a = adv.transpose(1, 2).contiguous(); c = clean.transpose(1, 2).contiguous()
    dist = torch.cdist(a, c)
    a_min, a_idx = dist.min(2); c_min, c_idx = dist.min(1)
    dynamic_cd = a_min.mean(1) + c_min.mean(1)
    dynamic_hd = torch.maximum(a_min.max(1).values, c_min.max(1).values)
    identity = (a - c).norm(dim=2)
    if frozen_a2c is None: frozen_a2c = a_idx.detach()
    if frozen_c2a is None: frozen_c2a = c_idx.detach()
    fa = dist.gather(2, frozen_a2c[:, :, None]).squeeze(2)
    fc = dist.transpose(1, 2).gather(2, frozen_c2a[:, :, None]).squeeze(2)
    frozen_cd = fa.mean(1) + fc.mean(1)
    stacked = torch.cat([a_min, c_min], 1)
    if fixed_witness is None: fixed_witness = stacked.detach().argmax(1)
    fixed_hd = stacked.gather(1, fixed_witness[:, None]).squeeze(1)
    soft_hd = temperature * torch.logsumexp(stacked / temperature, dim=1) - temperature * torch.log(
        torch.tensor(stacked.shape[1], dtype=stacked.dtype, device=stacked.device))
    return {'dynamic_cd': dynamic_cd, 'dynamic_hd': dynamic_hd,
            'identity_mean': identity.mean(1), 'identity_max': identity.max(1).values,
            'frozen_cd': frozen_cd, 'fixed_hd': fixed_hd, 'soft_hd': soft_hd,
            'a2c': a_idx, 'c2a': c_idx, 'witness': stacked.detach().argmax(1)}


def tensor_dict_to_records(values: Dict[str, torch.Tensor]):
    """Convert a dictionary of B-vectors to JSON/PT friendly per-sample rows."""
    keys = [k for k, v in values.items() if torch.is_tensor(v) and v.ndim == 1]
    if not keys: return []
    b = values[keys[0]].shape[0]
    return [{k: float(values[k][i].detach().cpu()) for k in keys} for i in range(b)]
