#!/usr/bin/env python
"""
NVFLARE client for FreqMixFormer on MMASD+.

Each FL client corresponds to one theme (theme1, theme2, theme3) and
uses the existing FreqMixFormer model and Feeder implementation.

This script is meant to be launched by NVIDIA FLARE (or the FL simulator)
as the client-side training loop.
"""

import argparse
import random
from typing import Dict, Optional, Tuple
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
    # custom/ -> app -> run_X -> workspace; use env or assume PYTHONPATH set
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
    npz_path = os.path.join(proj, "data", "MMASD+", f"MMASD+_{theme}.npz") if proj else f"data/MMASD+/MMASD+_{theme}.npz"

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


def set_model_params(model: nn.Module, params: Dict, personalization: str = "none") -> None:
    """Load parameter dict into model (keys should match state_dict()). Accepts numpy or tensor.

    personalization:
      - "none": load all params
      - "fedbn": keep local BN params (skip BN-related keys from global)
      - "fedper": keep local classifier head (skip fc.* from global)
      - "local": ignore global params (pure local training under FL framework)
    """
    if params is None:
        return
    if personalization == "local":
        # Do not load any external parameters: keep local weights only.
        return
    state_dict = model.state_dict()
    for k in state_dict.keys():
        if personalization == "fedbn":
            # Skip BN statistics/affine params so each client keeps its own BN.
            if (
                "bn" in k
                or "data_bn" in k
                or "running_mean" in k
                or "running_var" in k
                or "num_batches_tracked" in k
            ):
                continue
        if personalization == "fedper":
            # Skip classifier head so each client has its own fc.
            if k.startswith("fc."):
                continue
        if k in params:
            v = params[k]
            if isinstance(v, torch.Tensor):
                pass
            elif isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            else:
                # Scalars (e.g. numpy.float64) or other types
                v = torch.tensor(v, dtype=state_dict[k].dtype)
            state_dict[k] = v.to(state_dict[k].device)
    model.load_state_dict(state_dict, strict=False)


def get_model_params(model: nn.Module) -> Dict:
    """Return a CPU copy of model parameters as numpy arrays (for NUMPY exchange format)."""
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def local_train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    base_lr: float,
    weight_decay: float,
    fl_method: str = "fedavg",
    prox_mu: float = 0.0,
    global_params: Optional[Dict[str, torch.Tensor]] = None,
    global_round: int = 0,
    warmup_rounds: int = 0,
) -> None:
    """Run local training for a few epochs on the client's data."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=weight_decay,
    )

    # Simple warmup across FL rounds: linearly scale lr from 0 -> base_lr
    if warmup_rounds > 0 and global_round < warmup_rounds:
        lr = base_lr * float(global_round + 1) / float(warmup_rounds)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    for _ in range(epochs):
        for data, label, *_ in train_loader:
            data = data.to(device)  # expected shape: [N, C, T, V, M]
            label = label.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            if fl_method.lower() == "fedprox" and prox_mu > 0.0 and global_params is not None:
                prox = 0.0
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    p0 = global_params.get(name)
                    if p0 is None:
                        continue
                    prox = prox + (p - p0).pow(2).sum()
                loss = loss + 0.5 * prox_mu * prox
            loss.backward()
            optimizer.step()


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
        help="CUDA device string, e.g., cuda:0 or cpu",
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
        "--fl_method",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox"],
        help="FL method implemented on client side (server still does FedAvg aggregation)",
    )
    parser.add_argument(
        "--prox_mu",
        type=float,
        default=float(os.environ.get("FEDPROX_MU", "0.0")),
        help="FedProx mu (only used when --fl_method fedprox). "
        "Default can be set via FEDPROX_MU env var.",
    )
    parser.add_argument(
        "--warmup_rounds",
        type=int,
        default=5,
        help="number of warmup rounds for FL (lr linearly scales from 0 to base_lr).",
    )
    parser.add_argument(
        "--personalization",
        type=str,
        default="none",
        choices=["none", "fedbn", "fedper", "local"],
        help="personalized FL mode: keep local BN (fedbn), local head fc (fedper), or fully local (local)",
    )
    args = parser.parse_args()

    set_seed(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model to mirror config/mmasd/joint_themex.yaml
    model = FreqMixFormer(
        num_class=11,
        num_point=25,
        num_person=2,
        graph="graph.ntu_rgb_d.Graph",
        graph_args={"labeling_mode": "spatial"},
    ).to(device)

    train_loader, test_loader = build_dataloaders(args.theme, args.batch_size)

    # Initialize NVFLARE client environment
    flare.init()

    # Experiment tracking: streams metrics to server for TensorBoard
    writer = SummaryWriter()

    # FL loop: receive global model -> local train -> send updated model back
    round_num = 0
    while flare.is_running():
        input_model: FLModel = flare.receive()

        if input_model is None or input_model.params is None:
            # No more work from server
            break

        # Load global parameters into local model (respect personalization mode)
        set_model_params(model, input_model.params, personalization=args.personalization)

        global_params = None
        if args.fl_method.lower() == "fedprox" and args.prox_mu > 0.0:
            global_params = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

        # Local training (can be tuned per FL method)
        local_train(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.local_epochs,
            base_lr=args.base_lr,
            weight_decay=args.weight_decay,
            fl_method=args.fl_method,
            prox_mu=args.prox_mu,
            global_params=global_params,
            global_round=round_num,
            warmup_rounds=args.warmup_rounds,
        )

        # Local evaluation for monitoring
        acc = local_eval(model, test_loader, device)

        # Stream accuracy to server for experiment tracking (TensorBoard)
        writer.add_scalar(f"accuracy/{args.theme}", acc, round_num)
        round_num += 1

        # Collect parameters to send back.
        # For personalized FL, you could exclude some layers here (e.g., last FC).
        out_params = get_model_params(model)

        num_steps = args.local_epochs * len(train_loader)
        output_model = FLModel(
            params=out_params,
            metrics={"accuracy": acc},
            meta={
                "theme": args.theme,
                "NUM_STEPS_CURRENT_ROUND": num_steps,
            },
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()
