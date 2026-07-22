# Official ACT + language conditioning

The baseline is built on the official `tonyzhaozh/act` repository at commit
`742c753c0d4a5d87076c8f69e5628c79a8cc5488`. The original CVAE, ResNet
backbone, Transformer, action queries, and L1+KL loss are retained. A frozen
DistilBERT instruction embedding is projected into the ACT hidden dimension
and inserted as one additional Transformer memory token. Existing compressed
NPZ demonstrations are consumed directly and are never modified.

## Repository layout

- `aloha/`: shared MuJoCo environment, controller, recording, and instructions
- `experts/`: current scripted experts; old diagnostics are under `experts/legacy/`
- `training/`: ACT model, dataset, and training entry point
- `evaluation/`: single-episode and suite evaluators plus shell launchers
- `tools/`: replay, dataset validation, and simulator diagnostics
- `collect_demonstrations.py`: primary demonstration collection entry point

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
./.venv/Scripts/python.exe -u tools/validate_npz_dataset.py demonstrations_lr_train --max-episodes 5
```

## 2. Five-episode overfit smoke test

```bash
./.venv/Scripts/python.exe -u training/train_language_act.py --train-dir demonstrations_lr_train --max-episodes 5 --output checkpoints/language_act_overfit5 --chunk-size 10 --batch-size 4 --epochs 30 --max-batches-per-epoch 100
```

The run writes `normalization_stats.json`, `manifest.json`, `latest.pt`, and
`best.pt`. The manifest records the exact episode list and canonical
instructions.

## 3. Closed-loop overfit rollout

Restore one of the five training demonstrations exactly, query ACT every full
chunk, and execute each predicted 10 Hz action target in MuJoCo:

```bash
./.venv/Scripts/python.exe -u evaluation/evaluate_language_act.py --checkpoint checkpoints/language_act_overfit5/best.pt --episode demonstrations_lr_train_resume1/episode_0000.npz
```

Use `--no-viewer` for a fast headless check. This is a memorization test, not a
generalization measurement. A later evaluation should use held-out episode
initial conditions and report success over multiple seeds.

## 4. Render the ACT architecture

After installing the Graphviz application and the Python dependencies, render
the tensor-only ACT core as an SVG:

```bash
./.venv/Scripts/python.exe visualize_act_model.py
```

The default output is `model_visualizations/language_act_core.svg`. DistilBERT
is collapsed into a `(1, hidden_dim)` language-embedding input so the ACT graph
remains readable. Use `--expand-nested --depth 5` only when a much larger graph
is needed for debugging.

DistilBERT is always frozen; only its projection into ACT and the original ACT
vision/action policy are trainable.

## 5. Balanced closed-loop evaluation

Run all three report checkpoints against both main tasks and all primitives:

```bash
bash evaluation/evaluate_act_balanced.sh
```

Pass `seen`, `composition`, or `composition-worst` to evaluate one checkpoint.
PowerShell users can use `evaluation/run_act_evaluations.ps1` with the matching
`-Model` value.

## Important boundaries

- `unseen_rl` is evaluation-only and must never be included in a train folder.
- Normalization statistics are computed from train episodes only.
- ACT reads NPZ directly; RLDS conversion remains specific to OpenVLA-OFT.
- Offline overfitting is only a pipeline check. Closed-loop MuJoCo evaluation
  is still required before the full data-scale experiments.
