#!/usr/bin/env python
"""
NVFLARE client for FreqMixFormer on MMASD+.

Each FL client corresponds to one theme (theme1, theme2, theme3) and
uses the existing FreqMixFormer model and Feeder implementation.

This script is meant to be launched by NVIDIA FLARE (or the FL simulator)
as the client-side training loop.
"""

import argparse
from typing import Dict, Tuple
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import nvflare.client as flare
from nvflare.client import FLModel

# Ensure project root (FreqMixFormer) is on sys.path so that `model.*` imports work
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_this_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from model.skefreqmixformer import Model as FreqMixFormer
from feeders.feeder_ntu import Feeder


def build_dataloaders(theme: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Construct train/test dataloaders for a given MMASD+ theme."""
    data_root = "data/MMASD+"
    npz_path = f"{data_root}/MMASD+_{theme}.npz"

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


def set_model_params(model: nn.Module, params: Dict[str, torch.Tensor]) -> None:
    """Load parameter dict into model (keys should match state_dict())."""
    if params is None:
        return
    state_dict = model.state_dict()
    for k in state_dict.keys():
        if k in params:
            state_dict[k] = params[k].to(state_dict[k].device)
    model.load_state_dict(state_dict, strict=False)


def get_model_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a CPU copy of model parameters."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def local_train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    base_lr: float,
    weight_decay: float,
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

    for _ in range(epochs):
        for data, label, *_ in train_loader:
            data = data.to(device)  # expected shape: [N, C, T, V, M]
            label = label.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
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
    args = parser.parse_args()

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

    # FL loop: receive global model -> local train -> send updated model back
    while flare.is_running():
        input_model: FLModel = flare.receive()

        if input_model is None or input_model.params is None:
            # No more work from server
            break

        # Load global parameters into local model
        set_model_params(model, input_model.params)

        # Local training (can be tuned per FL method)
        local_train(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.local_epochs,
            base_lr=args.base_lr,
            weight_decay=args.weight_decay,
        )

        # Local evaluation for monitoring
        acc = local_eval(model, test_loader, device)

        # Collect parameters to send back.
        # For personalized FL, you could exclude some layers here (e.g., last FC).
        out_params = get_model_params(model)

        output_model = FLModel(
            params=out_params,
            metrics={"accuracy": acc},
            meta={"theme": args.theme},
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()

