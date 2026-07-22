# Boundary-state adaptation experiment

This experiment tests whether ACT's unseen-role failure comes from the state
distribution at the transition between the two primitives.

The added task is `left_pick_place_after_right_push`:

1. A scripted expert completes `right_tray_push` and retreats.
2. Recording starts at that post-push boundary state.
3. The scripted expert completes `left_pick_place`.

The first primitive is deliberately not recorded. Therefore the saved policy
target is still only Left pick-and-place, but its initial state matches the
state encountered after Right tray push.

## Files

- `collect_boundary_dataset.py`: generates one split of boundary-state NPZs.
- `collect_boundary_splits.ps1`: generates train/val/test splits.
- `train_boundary_act.ps1`: trains a fresh six-task, task-balanced ACT model.
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

Final expected counts are:

- train: 50 successful episodes
- val: 10 successful episodes
- primitive test: 20 successful episodes

Only successful Right-push -> Left-PnP trajectories are retained. `--episodes`
means the desired total file count when resuming, not the number to append.

## 2. Train ACT

```powershell
.\experiments\boundary_adaptation\train_boundary_act.ps1
```

The fresh model uses six tasks, with 50 train and 10 validation episodes per
task. Task-balanced sampling and batch size 12 place two samples from every
task in each idealized balanced batch. Checkpoints are written to:

```text
checkpoints/language_act_boundary_balanced_300
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
results/act_boundary/<checkpoint>/
```

The decisive comparisons are:

- boundary Left PnP versus ordinary Left PnP
- unseen RL before versus after boundary-state training
- Expert Right push -> ACT Left PnP in the hybrid suite

Improvement only on the boundary and hybrid tests supports the
phase-boundary distribution-shift explanation. Improvement on full unseen RL
also shows that ACT can use the added transition data to compose the roles.
