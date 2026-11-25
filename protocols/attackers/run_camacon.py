"""Run CaMaCon on a pretrained point-cloud classifier."""
import argparse
import itertools
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


DEFAULT_PARAMS = {'attack_lr': 0.14001500596318536,
 'binary_step': 13,
 'budget': 8.841486365518142,
 'cd_weight': 1.8420756304322634,
 'curv_weight': 4.939875604446555e-05,
 'haus_weight': 0.05,
 'hide_weight': 4.4328623901948285,
 'init_weight': 7.295157181269302,
 'ker_weight': 15.206542574436652,
 'latent_loss_weight': 0.0003828036109506491,
 'max_weight': 93.84312004534112,
 'num_iter': 200,
 'candidate_rule': 'cd_hd',
 'candidate_hd_weight': 2.0}


def evaluate(all_attacked_pc, model, device):
    import torch
    from src.attackers.camacon.dist_utils import ChamferDist, HausdorffDist
    cd_dist = ChamferDist()
    hd_dist = HausdorffDist(method="both")
    total = success = 0
    total_cd = total_hd = 0.0
    centroid_shifts = []
    model.eval()
    for batch in all_attacked_pc:
        real = batch["real_data"].to(device).float()
        adv = batch["attacked_data"].to(device).float()
        label = batch["real_label"].to(device)
        with torch.no_grad():
            pred = model(adv)["logit"].argmax(dim=1)
            mask = pred != label
        # centroid displacement (translation shortcut) per sample
        shift = (adv.mean(dim=1) - real.mean(dim=1)).norm(dim=-1)
        centroid_shifts.extend(shift.cpu().tolist())
        num = int(mask.sum().item())
        if num > 0:
            total_cd += cd_dist(adv[mask], real[mask], batch_avg=False).sum().item()
            total_hd += hd_dist(adv[mask], real[mask], batch_avg=False).sum().item()
        total += real.shape[0]
        success += num
    asr = success / total if total else 0.0
    cd = total_cd / success if success else float("inf")
    hd = total_hd / success if success else float("inf")
    return dict(asr=asr, cd=cd, hd=hd, total=total, success=success,
                centroid_shifts=centroid_shifts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="pct", choices=["pointnet", "dgcnn", "pct", "curvenet", "pointnet2"])
    p.add_argument("--dataset", default="ShapeNetPart", choices=["ModelNet40", "ShapeNetPart"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_batches_to_attack", type=int, default=17)
    p.add_argument("--start_batch", type=int, default=0, help="skip this many test batches")
    p.add_argument("--centroid_align", action="store_true", help="forbid translation shortcut")
    p.add_argument("--no_centroid_align", dest="centroid_align", action="store_false")
    p.set_defaults(centroid_align=True)
    p.add_argument("--num_iter", type=int, default=None, help="override num_iter (for smoke tests)")
    p.add_argument("--binary_step", type=int, default=None, help="override binary_step (for smoke tests)")
    p.add_argument("--total_samples", type=int, default=None, help="cap total samples")
    p.add_argument("--params_json", help="optional JSON file overriding the built-in experiment configuration")
    p.add_argument("--tag", default=None, help="output filename tag (default: aligned/plain)")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    import torch
    import matplotlib
    matplotlib.use("Agg")
    from loguru import logger
    from src.models.trainer import load_model, load_dataset
    from src.attackers.camacon.starter import start_camacon_attack
    from src.utils.save_utils import pre_check_path

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20261123)
    data_root = os.path.join(PROJECT_ROOT, "data")
    ckpt_root = os.path.join(PROJECT_ROOT, "ckpt")

    params = DEFAULT_PARAMS.copy()
    if args.params_json:
        with open(args.params_json, encoding="utf-8") as f:
            params.update(json.load(f))
    if args.num_iter is not None:
        params["num_iter"] = args.num_iter
    if args.binary_step is not None:
        params["binary_step"] = args.binary_step
    if args.total_samples is not None:
        params["total_samples"] = args.total_samples
    params["centroid_align"] = bool(args.centroid_align)
    print(f"[CONFIG] centroid_align={args.centroid_align} params={json.dumps(params, sort_keys=True)}", flush=True)

    model = load_model(device, args.model, args.dataset, ckpt_root)
    _, loader_test = load_dataset(args.batch_size, args.dataset, data_root)
    if args.start_batch:
        loader_test = itertools.islice(loader_test, args.start_batch, None)
    t0 = time.monotonic()
    all_attacked_pc = start_camacon_attack(
        model=model, device=device, loader_train=None, loader_test=loader_test,
        num_batches_to_attack=args.num_batches_to_attack, args=args,
        attack_params=params,
    )
    attack_sec = time.monotonic() - t0

    res = evaluate(all_attacked_pc, model, device)
    shifts = res.pop("centroid_shifts")
    res.update(
        dataset=args.dataset, model=args.model, method="CaMaCon",
        centroid_align=bool(args.centroid_align),
        params=params, attack_seconds=attack_sec,
        centroid_shift_mean=sum(shifts) / len(shifts) if shifts else None,
        centroid_shift_max=max(shifts) if shifts else None,
    )
    print("[RESULT] " + json.dumps({k: v for k, v in res.items() if k != "params"}), flush=True)

    out_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, "attacked_data", "camacon_centroid")
    tag = args.tag or ("aligned" if args.centroid_align else "plain")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(all_attacked_pc, pre_check_path(
        os.path.join(out_dir, f"{args.dataset.lower()}_{args.model.lower()}_{tag}_res.pt")))
    with open(os.path.join(out_dir, f"{args.dataset.lower()}_{args.model.lower()}_{tag}_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
