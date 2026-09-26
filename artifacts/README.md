# Artifact directory

Checkpoints and replays are not committed. The Section 5.4 configuration uses:

```text
artifacts/source_run/               # config, manifest, replay_train.npz
artifacts/profile_run/              # config and profile.pt
artifacts/checkpoints/checkpoint_0150.pt
```

These files must come from the same declared training run. The audit verifies
transition partitions and records hashes in its output manifest.

