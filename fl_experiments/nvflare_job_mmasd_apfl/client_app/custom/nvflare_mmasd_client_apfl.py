#!/usr/bin/env python
"""
NVFLARE client implementing APFL (Adaptive Personalized Federated Learning)
for FreqMixFormer on MMASD+.

We maintain a shared model (global) and a personalized model per client.
The server still aggregates only the shared model (FedAvg-style).
Evaluation uses the personalized model.
"""

import argparse
import copy
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import nvflare.client as flare
from nvflare.client import FLModel
from nvflare.client.tracking import SummaryWriter

# Ensure project root (FreqMixFormer) is on sys.path for model.* imports.
_proj_root = os.environ.get("FLARE_PROJECT_ROOT")
if not _proj_root:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_dir))))
if _proj_root and _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from model.skefreqmixformer import Model as FreqMixFormer
from feeders.feeder_ntu import Feeder


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(theme: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Construct train/test dataloaders for a given MMASD+ theme."""
    proj = os.environ.get("FLARE_PROJECT_ROOT") or _proj_root
    npz_path = (
        os.path.join(proj, "data", "MMASD+", f"MMASD+_{theme}.npz") if proj else f"data/MMASD+/MMASD+_{theme}.npz"
    )

    train_ds = Feeder(
        data_path=npz_path,
        split="train",
        window_size=64,
        p_interval=[0.5, 1.0],
        random_rot=True,
        random_choose=False,
        random_shift=False,
        random_move=False,
        vel=False,
        bone=False,
        debug=False,
    )
    test_ds = Feeder(
        data_path=npz_path,
        split="test",
        window_size=64,
        p_interval=[0.95],
        vel=False,
        bone=False,
        debug=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, test_loader


def set_model_params(model: nn.Module, params: Dict) -> None:
    """Load parameter dict into model (keys should match state_dict()). Accepts numpy or tensor."""
    if params is None:
        return
    state_dict = model.state_dict()
    for k in state_dict.keys():
        if k in params:
            v = params[k]
            if isinstance(v, torch.Tensor):
                pass
            elif isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            else:
                v = torch.tensor(v, dtype=state_dict[k].dtype)
            state_dict[k] = v.to(state_dict[k].device)
    model.load_state_dict(state_dict, strict=False)


def get_model_params(model: nn.Module) -> Dict:
    """Return a CPU copy of model parameters as numpy arrays (for NUMPY exchange format)."""
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def alpha_update(
    model: nn.Module,
    model_per: nn.Module,
    alpha: float,
    lr: float,
    reg: float = 0.02,
) -> float:
    """Update alpha as in APFL."""
    grad_alpha = 0.0
    for l_params, p_params in zip(model.parameters(), model_per.parameters()):
        dif = p_params.data - l_params.data
        if p_params.grad is None or l_params.grad is None:
            continue
        grad = alpha * p_params.grad.data + (1.0 - alpha) * l_params.grad.data
        grad_alpha += float(dif.view(-1).dot(grad.view(-1)))
    grad_alpha += reg * alpha
    alpha_new = alpha - lr * grad_alpha
    alpha_new = float(np.clip(alpha_new, 0.0, 1.0))
    return alpha_new


def local_train_apfl(
    model: nn.Module,
    model_per: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    base_lr: float,
    weight_decay: float,
    alpha: float,
    global_round: int = 0,
    warmup_rounds: int = 0,
) -> float:
    """APFL local training: update shared and personalized models and alpha."""
    model.train()
    model_per.train()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=weight_decay,
    )
    optimizer_per = optim.SGD(
        model_per.parameters(),
        lr=base_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=weight_decay,
    )

    # Round-level warmup: adjust lr at the start of local training
    if warmup_rounds > 0 and global_round < warmup_rounds:
        lr = base_lr * float(global_round + 1) / float(warmup_rounds)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer_per.param_groups:
            param_group["lr"] = lr

    for _ in range(epochs):
        for data, label, *_ in train_loader:
            data = data.to(device)
            label = label.to(device)

            # Update shared model
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # Update personalized model
            optimizer_per.zero_grad()
            output_per = model_per(data)
            loss_per = criterion(output_per, label)
            loss_per.backward()
            optimizer_per.step()

            # Update alpha using current grads (as in APFL)
            alpha = alpha_update(model, model_per, alpha, lr=base_lr)

    # Couple personalized model with shared model at the end of local training
    with torch.no_grad():
        for lp, p in zip(model_per.parameters(), model.parameters()):
            lp.data = (1.0 - alpha) * p.data + alpha * lp.data

    return alpha


def local_eval(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Compute top-1 accuracy on the local test split."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label, *_ in data_loader:
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.numel()
    if total == 0:
        return 0.0
    return correct / total


def local_eval_per_class(
    model: nn.Module, data_loader: DataLoader, device: torch.device, num_class: int = 11
) -> Tuple[float, List[float]]:
    """Compute overall and per-class accuracy; return (overall_acc, list of per-class acc)."""
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    with torch.no_grad():
        for data, label, *_ in data_loader:
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            all_labels.extend(label.cpu().tolist())
            all_preds.extend(pred.cpu().tolist())
    if not all_labels:
        return 0.0, [0.0] * num_class
    from sklearn.metrics import confusion_matrix
    labels_arr = np.array(all_labels, dtype=np.int64)
    preds_arr = np.array(all_preds, dtype=np.int64)
    confusion = confusion_matrix(labels_arr, preds_arr, labels=list(range(num_class)))
    list_raw_sum = np.sum(confusion, axis=1)
    list_raw_sum = np.where(list_raw_sum == 0, 1, list_raw_sum)
    each_acc = (np.diag(confusion) / list_raw_sum).tolist()
    overall = (labels_arr == preds_arr).mean().item()
    return overall, each_acc


def save_per_class_csv(per_class_acc: List[float], round_num: int, theme: str) -> None:
    """Write per-class accuracy to per_class_acc.csv in cwd (for Streamlit to load)."""
    out_path = Path.cwd() / "per_class_acc.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "theme"] + [f"class_{i}" for i in range(len(per_class_acc))])
        writer.writerow([round_num, theme] + per_class_acc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--theme",
        type=str,
        required=True,
        help="theme name (e.g., theme1, theme2, theme3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device string, e.g., cuda:0 or cpu; use 'auto' to map theme1/2/3 to different GPUs.",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="number of local epochs per FL round",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="local batch size",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=0.1,
        help="local learning rate (match centralized config for fair comparison)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="weight decay",
    )
    parser.add_argument(
        "--apfl_alpha",
        type=float,
        default=0.01,
        help="initial alpha for APFL convex combination",
    )
    parser.add_argument(
        "--warmup_rounds",
        type=int,
        default=5,
        help="number of warmup rounds for FL (lr linearly scales from 0 to base_lr).",
    )
    args = parser.parse_args()

    set_seed(42)

    # Determine device, optionally mapping different clients to different GPUs when device=='auto'
    if torch.cuda.is_available():
        if args.device == "auto":
            try:
                site_name = flare.get_site_name()
            except Exception:
                site_name = args.theme
            theme_to_idx = {"theme1": 0, "theme2": 1, "theme3": 2}
            local_idx = theme_to_idx.get(site_name, 0)
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible:
                gpus = [g.strip() for g in visible.split(",") if g.strip()]
                if gpus:
                    local_idx = min(local_idx, len(gpus) - 1)
            device_str = f"cuda:{local_idx}"
        else:
            device_str = args.device
    else:
        device_str = "cpu"

    device = torch.device(device_str)

    # Build shared model and personalized copy
    model = FreqMixFormer(
        num_class=11,
        num_point=25,
        num_person=2,
        graph="graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
    ).to(device)
    model_per = copy.deepcopy(model)

    train_loader, test_loader = build_dataloaders(args.theme, args.batch_size)

    flare.init()
    writer = SummaryWriter()

    alpha = float(args.apfl_alpha)
    personal_initialized = False
    round_num = 0

    while flare.is_running():
        input_model: FLModel = flare.receive()

        if input_model is None or input_model.params is None:
            break

        # Load global shared parameters into shared model each round
        set_model_params(model, input_model.params)

        # Initialize personalized model only at the first round
        if not personal_initialized:
            model_per = copy.deepcopy(model)
            personal_initialized = True

        # APFL local training updates shared model, personalized model, and alpha
        alpha = local_train_apfl(
            model=model,
            model_per=model_per,
            train_loader=train_loader,
            device=device,
            epochs=args.local_epochs,
            base_lr=args.base_lr,
            weight_decay=args.weight_decay,
            alpha=alpha,
            global_round=round_num,
            warmup_rounds=args.warmup_rounds,
        )

        # Evaluate using personalized model
        acc = local_eval(model_per, test_loader, device)
        _, per_class_acc = local_eval_per_class(model_per, test_loader, device, num_class=11)
        try:
            save_per_class_csv(per_class_acc, round_num, args.theme)
        except Exception:
            pass
        writer.add_scalar(f"accuracy/{args.theme}", acc, round_num)
        writer.add_scalar(f"alpha/{args.theme}", alpha, round_num)
        round_num += 1

        out_params = get_model_params(model)
        num_steps = args.local_epochs * len(train_loader)
        output_model = FLModel(
            params=out_params,
            metrics={"accuracy": acc},
            meta={
                "theme": args.theme,
                "NUM_STEPS_CURRENT_ROUND": num_steps,
                "alpha": alpha,
            },
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()

in()

        "NUM_STEPS_CURRENT_ROUND": num_steps,
                "alpha": alpha,
            },
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()

