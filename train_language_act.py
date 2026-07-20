import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from language_act_dataset import (
    AlohaNPZActionChunkDataset,
    compute_normalization_stats,
    discover_episode_paths,
    save_normalization_stats,
)
ACT_ROOT = Path(__file__).parent / "third_party" / "official_act"
if not ACT_ROOT.exists():
    raise FileNotFoundError(
        f"Vendored official ACT source not found at {ACT_ROOT}. "
        "See LANGUAGE_ACT.md for setup details."
    )
sys.path.insert(0, str(ACT_ROOT))

from policy import ACTPolicy


def move_batch(batch, device):
    return {
        "images": batch["images"].to(device, non_blocking=True),
        "state": batch["state"].to(device, non_blocking=True),
        "actions": batch["actions"].to(device, non_blocking=True),
        "is_pad": batch["is_pad"].to(device, non_blocking=True),
        "instruction": batch["instruction"],
    }


def run_epoch(model, loader, device, kl_weight, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "l1": 0.0, "kl": 0.0, "prior_l1": 0.0}
    batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch(batch, device)
            loss_dict = model(
                batch["state"],
                batch["images"],
                batch["instruction"],
                actions=batch["actions"],
                is_pad=batch["is_pad"],
            )
            loss = loss_dict["loss"]
            l1 = loss_dict["l1"]
            kl = loss_dict["kl"]
            prior_l1 = torch.zeros((), device=device)
            if not training:
                prior_prediction = model(
                    batch["state"],
                    batch["images"],
                    batch["instruction"],
                )
                valid = (~batch["is_pad"]).unsqueeze(-1).expand_as(
                    batch["actions"]
                )
                prior_l1 = torch.abs(
                    prior_prediction - batch["actions"]
                )[valid].mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["l1"] += float(l1.detach())
            totals["kl"] += float(kl.detach())
            totals["prior_l1"] += float(prior_l1.detach())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    return {key: value / max(batches, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", action="append", required=True)
    parser.add_argument("--val-dir", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("checkpoints/language_act"))
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--text-model-name", default="distilbert-base-uncased")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=6)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=3200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches-per-epoch", type=int)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume model and optimizer state; --epochs is the final epoch.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    train_paths = discover_episode_paths(args.train_dir, args.max_episodes)
    val_paths = discover_episode_paths(args.val_dir) if args.val_dir else train_paths
    stats = compute_normalization_stats(train_paths)
    save_normalization_stats(stats, args.output / "normalization_stats.json")

    train_dataset = AlohaNPZActionChunkDataset(
        train_paths, args.chunk_size, stats, cache_size=min(len(train_paths), 8)
    )
    val_dataset = AlohaNPZActionChunkDataset(
        val_paths, args.chunk_size, stats, cache_size=min(len(val_paths), 8)
    )
    policy_config = {
        "lr": args.learning_rate,
        "lr_backbone": args.lr_backbone,
        "backbone": args.backbone,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "camera_names": [
            "overhead_cam", "wrist_cam_left", "wrist_cam_right"
        ],
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "text_model_name": args.text_model_name,
        "device": str(device),
    }
    model = ACTPolicy(policy_config).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = model.configure_optimizers()

    start_epoch = 1
    best_val = float("inf")
    best_prior = float("inf")
    if args.resume is not None:
        resume_path = args.resume.resolve()
        resume_checkpoint = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if resume_checkpoint["model_config"]["num_queries"] != args.chunk_size:
            raise ValueError("Resume checkpoint chunk size does not match --chunk-size")
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_val = float(
            resume_checkpoint.get(
                "best_val", resume_checkpoint["val_metrics"]["loss"]
            )
        )
        best_prior = float(resume_checkpoint.get("best_prior", float("inf")))
        print(
            f"resumed={resume_path}, start_epoch={start_epoch}, "
            f"previous_val={resume_checkpoint['val_metrics']['loss']:.5f}",
            flush=True,
        )

    manifest = {
        "train": [str(path) for path in train_paths],
        "val": [str(path) for path in val_paths],
        "instructions": sorted(set(train_dataset.instructions)),
        "args": vars(args)
        | {
            "output": str(args.output),
            "resume": str(args.resume) if args.resume is not None else None,
        },
        "model_config": policy_config,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(
        f"device={device}, train_episodes={len(train_paths)}, "
        f"train_steps={len(train_dataset)}, val_episodes={len(val_paths)}"
    )
    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}; "
            f"set --epochs to at least {start_epoch}."
        )
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.kl_weight,
            optimizer=optimizer,
            max_batches=args.max_batches_per_epoch,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            args.kl_weight,
            max_batches=args.max_batches_per_epoch,
        )
        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_l1={train_metrics['l1']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_l1={val_metrics['l1']:.5f} "
            f"val_prior_l1={val_metrics['prior_l1']:.5f}",
            flush=True,
        )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": policy_config,
            "val_metrics": val_metrics,
            "best_val": min(best_val, val_metrics["loss"]),
            "best_prior": min(best_prior, val_metrics["prior_l1"]),
        }
        torch.save(checkpoint, args.output / "latest.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, args.output / "best.pt")
        if val_metrics["prior_l1"] < best_prior:
            best_prior = val_metrics["prior_l1"]
            torch.save(checkpoint, args.output / "best_prior.pt")


if __name__ == "__main__":
    main()
