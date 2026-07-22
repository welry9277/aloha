# Boundary-state adaptation experiment

This experiment tests whether ACT's unseen-role failure comes from the state
distribution at the transition between the two primitives.

The two boundary tasks are:

1. `left_pick_place_after_right_push`
   - scripted Right tray push and retreat
   - start recording
   - scripted Left pick-and-place
2. `right_pick_place_after_left_push`
   - scripted Left tray push and retreat
   - start recording
   - scripted Right pick-and-place

The push primitive is deliberately not recorded. Therefore each saved policy
target contains only PnP actions while its initial state matches the phase
boundary produced by the opposite arm's push.

## Files

- `collect_boundary_dataset.py`: generates one split of boundary-state NPZs.
- `collect_boundary_splits.ps1`: generates train/val/test splits.
- `train_boundary_act.ps1`: trains the symmetric five-task boundary model.
- `evaluate_boundary_act.ps1`: runs main, primitive, boundary, and hybrid tests.

Run all PowerShell commands from the repository root on the Windows GPU host.
The scripts resolve their own paths, so activating the virtual environment is
optional as long as `.venv` exists.

## 1. Generate data

Optional two-episode smoke collection:

```powershell
.\experiments\boundary_adaptation\collect_boundary_splits.ps1 `
  -TrainEpisodes 2 -ValEpisodes 2 -TestEpisodes 2
```

After that smoke test, extend the same directories to the final totals:

```powershell
.\experiments\boundary_adaptation\collect_boundary_splits.ps1 -Resume
```

If the split directories are empty, collect the final dataset directly:

```powershell
.\experiments\boundary_adaptation\collect_boundary_splits.ps1
```

Final expected counts for each of the two boundary tasks are:

- train: 50 successful episodes
- val: 10 successful episodes
- primitive test: 20 successful episodes

Only trajectories for which both push and PnP succeed are retained.
`--episodes` means the desired total file count when resuming, not the number
to append.

To extend only the two training splits from 50 to 200 episodes, use the
dedicated script. It uses fresh seed ranges and preserves existing files:

```powershell
.\experiments\boundary_adaptation\extend_boundary_train_to_200.ps1
```

For a background run:

```powershell
.\experiments\boundary_adaptation\extend_boundary_train_to_200.ps1 -Background
```

With exactly 50 existing files, this adds 150 files named
`episode_0050.npz` through `episode_0199.npz` for each boundary direction.

## 2. Train ACT

```powershell
.\experiments\boundary_adaptation\train_boundary_act.ps1
```

To keep training alive after closing the terminal, launch it in the
background:

```powershell
.\experiments\boundary_adaptation\train_boundary_act.ps1 -Background
```

The command prints the background process ID. Follow its training log with:

```powershell
Get-Content .\logs\language_act_symmetric_boundary_250.log -Wait
```

The fresh model uses five tasks, with 50 train and 10 validation episodes per
task. It replaces both ordinary PnP datasets with their post-opposite-push
boundary versions. This keeps the comparison against the original composition
model fixed at 250 training episodes. Task-balanced sampling and batch size 10
place two samples from every task in each balanced batch.

The five training tasks are:

```text
seen_lr
left_tray_push
right_tray_push
left_pick_place_after_right_push
right_pick_place_after_left_push
```

Checkpoints are written to:

```text
checkpoints/language_act_symmetric_boundary_250
```

Do not resume a five-task checkpoint: the task mixture and normalization
statistics have changed.

## 3. Evaluate

Quick two-episode test of every suite:

```powershell
.\experiments\boundary_adaptation\evaluate_boundary_act.ps1 `
  -Suite all -Checkpoint best-worst -Limit 2
```

Full evaluation:

```powershell
.\experiments\boundary_adaptation\evaluate_boundary_act.ps1 `
  -Suite all -Checkpoint best-worst
```

Individual suites can be selected with `main`, `primitives`, `boundary`, or
`hybrid`. To compare checkpoint criteria, replace `best-worst` with
`best-prior` or `latest`.

Results are saved under:

```text
results/act_symmetric_boundary/<checkpoint>/
```

The decisive comparisons are:

- each boundary PnP versus its held-out ordinary PnP
- unseen RL before versus after boundary-state training
- Expert Right push -> ACT Left PnP in the hybrid suite

Improvement only on the boundary and hybrid tests supports the
phase-boundary distribution-shift explanation. Improvement on full unseen RL
also shows that ACT can use the added transition data to compose the roles.
