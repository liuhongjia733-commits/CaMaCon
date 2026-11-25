import os
import torch
from loguru import logger
from matplotlib import pyplot as plt
from pytorch3d.structures import Pointclouds
from tqdm import tqdm

from src.attackers.camacon.backbone import CaMaCon
from src.attackers.camacon.fold_inv_model.inv_model import InvertibleAutoEncoder  # <-- 假设 inv_model.py 在您的项目路径中
from src.utils.model_eval import get_num_classes_by_dataset_name
from src.utils.save_utils import pre_check_path


def start_camacon_attack(model, device, loader_train, loader_test, num_batches_to_attack, args,attack_params=None):
    print("[STARTER] start_camacon_attack called", flush=True)
    logger.info(
        f"Start CaMaCon attack on {args.dataset} dataset, with {args.model} model")
    num_classes = get_num_classes_by_dataset_name(args.dataset)
    print(f"[STARTER] num_classes={num_classes}", flush=True)

    weights_dir = os.path.join(os.path.dirname(__file__), 'fold_inv_model')
    pretrained_weights_map={
        'ModelNet40': os.path.join(weights_dir, 'best_model.pth'),
        'ShapeNetPart': os.path.join(weights_dir, 'best_model_shapenet.pth')
    }
    dataset_name = args.dataset
    if dataset_name not in pretrained_weights_map:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {list(pretrained_weights_map.keys())}")
    AE_MODEL_PATH = pretrained_weights_map[dataset_name]
    NUM_POINTS = 1024  # <--- 确保点数与您的模型和数据集匹配

    logger.info(f"Loading Invertible Autoencoder from {AE_MODEL_PATH}")
    print(f"[STARTER] loading AE model from {AE_MODEL_PATH}...", flush=True)
    try:
        # 初始化自编码器模型结构
        ae_model = InvertibleAutoEncoder(num_points=NUM_POINTS,pretrained_path=AE_MODEL_PATH)
        ae_model.to(device)
        ae_model.eval()  # 设置为评估模式
        for parameter in ae_model.parameters():
            parameter.requires_grad_(False)
        print("[STARTER] AE model loaded and moved to device", flush=True)
        logger.info("Invertible Autoencoder loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Invertible Autoencoder: {e}")
        raise e

    all_attacked_pc = []
    # [NEW] 总共需要处理的样本数（不同点云），超出则停止攻击并截断最后一批
    total_samples = (attack_params.get('total_samples', None) if attack_params else None)
    processed_samples = 0
    print(f"[STARTER] starting attack loop, {num_batches_to_attack} batches, total_samples={total_samples}...", flush=True)
    for batch_idx, data_item in enumerate(tqdm(loader_test)):
        if batch_idx >= num_batches_to_attack:
            break
        if total_samples is not None and processed_samples >= total_samples:
            break
        print(f"[STARTER] batch {batch_idx}...", flush=True)
        pc = data_item["pc"]
        label = data_item["label"]
        # [NEW] 截断最后一个 batch，保证总样本数正好等于 total_samples
        if total_samples is not None and pc.shape[0] > (total_samples - processed_samples):
            keep = total_samples - processed_samples
            pc = pc[:keep]
            label = label[:keep]
            print(f"[STARTER] truncating batch {batch_idx} to {keep} samples", flush=True)
        pc = pc.to(device).float()
        label = label.to(device)
        target_label = (label + 1) % num_classes
        if attack_params is None:
            attack_params = {}

        print(f"[STARTER] creating CaMaCon instance...", flush=True)
            # [核心] 使用 **attack_params 将字典解包成关键字参数
        camacon = CaMaCon(
            model=model,
            ae_model=ae_model,
            device=device,
            **attack_params  # 这一行会将 Optuna 的参数正确地传递给 CaMaCon 类
        )
        camacon._batch_idx = batch_idx  # 用于对比图文件名区分不同batch
        print(f"[STARTER] CaMaCon created (batch {batch_idx})", flush=True)
        # --- [结束修改 2] ---

        print(f"[STARTER] estimating normals...", flush=True)
        pc_pyt3d = Pointclouds(points=pc)
        normals = pc_pyt3d.estimate_normals()
        if normals is None:
            # 如果法线计算失败（例如点云退化），则跳过此批次
            logger.warning(f"Could not estimate normals for batch {batch_idx}, skipping.")
            continue
        print(f"[STARTER] normals estimated, shape={normals.shape}", flush=True)

        # 拼接坐标和法线
        pc_with_normals = torch.cat([pc, normals], dim=-1)

        # 调用攻击方法 (这里的调用方式保持不变)
        # CaMaCon 的 attack 方法接收 [B, N, 6] 的输入
        print(f"[STARTER] running attack on batch {batch_idx}...", flush=True)
        adv_pc, success_num = camacon.attack(pc_with_normals.float(), label.long())
        print(f"[STARTER] attack done, success_num={success_num}", flush=True)
        processed_samples += pc.shape[0]

        losses = camacon.losses
        fig, axs = plt.subplots(6, 1, dpi=300)
        ii = 0
        for k, v in losses.items():
            axs[ii].plot(v, label=k)
            axs[ii].set_title(k)
            ii += 1
        fig.legend()
        plt.savefig(pre_check_path(f"figures/camacon_{batch_idx}.png"))

        #if batch_idx > 10:
           # 1 / 0
        plt.close(fig)  # 增加关闭图形操作，防止内存泄漏
        all_attacked_pc.append({
            # pc 的形状是 [B, N, 3]
            "real_data": pc.cpu(),
            # adv_pc_transposed 的形状也是 [B, N, 3]
            "attacked_data": torch.from_numpy(adv_pc).float().cpu(),
            "real_label": label.cpu(),
            "target_label": target_label.cpu(),
            "success_num": success_num,
            "attack_diagnostics": getattr(camacon, 'diagnostics', None),
            "ae_load_report": getattr(ae_model, 'pretrained_load_report', None),
        })
        print(f"[STARTER] batch {batch_idx} appended to results", flush=True)

    print(f"[STARTER] attack loop done, returning {len(all_attacked_pc)} batches", flush=True)
    return all_attacked_pc
