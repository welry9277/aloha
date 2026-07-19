# Official ACT + language conditioning

The baseline is built on the official `tonyzhaozh/act` repository at commit
`742c753c0d4a5d87076c8f69e5628c79a8cc5488`. The original CVAE, ResNet
backbone, Transformer, action queries, and L1+KL loss are retained. A frozen
DistilBERT instruction embedding is projected into the ACT hidden dimension
and inserted as one additional Transformer memory token. Existing compressed
NPZ demonstrations are consumed directly and are never modified.

## Inputs and targets

- Three RGB cameras in this order: overhead, left wrist, right wrist
- 14D normalized joint/gripper state
- One canonical English instruction per episode
- A padded `(chunk_size, 14)` normalized action target

Install the official ACT and language dependencies once:

```bash
./.venv/Scripts/python.exe -m pip install -r requirements-language-act.txt
./.venv/Scripts/python.exe -m pip install -e third_party/official_act/detr
```

## 1. Validate NPZ files

```bash
./.venv/Scripts/python.exe -u validate_npz_dataset.py demonstrations_lr_train --max-episodes 5
```

## 2. Five-episode overfit smoke test

```bash
./.venv/Scripts/python.exe -u train_language_act.py --train-dir demonstrations_lr_train --max-episodes 5 --output checkpoints/language_act_overfit5 --chunk-size 10 --batch-size 4 --epochs 30 --max-batches-per-epoch 100
```

The run writes `normalization_stats.json`, `manifest.json`, `latest.pt`, and
`best.pt`. The manifest records the exact episode list and canonical
instructions.

DistilBERT is always frozen; only its projection into ACT and the original ACT
vision/action policy are trainable.

## Important boundaries

- `unseen_rl` is evaluation-only and must never be included in a train folder.
- Normalization statistics are computed from train episodes only.
- ACT reads NPZ directly; RLDS conversion remains specific to OpenVLA-OFT.
- Offline overfitting is only a pipeline check. Closed-loop MuJoCo evaluation
  is still required before the full data-scale experiments.
