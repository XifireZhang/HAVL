# Data directory

Large datasets are intentionally excluded from Git. The default configuration
expects KuaiRand-27K under:

```text
third_party/kuaisim/dataset/kuairand/Kuairand-27K/data/
```

Use `third_party/kuaisim/scripts/download_kuairand_27k.py` and
`prepare_kuairand_27k_features.py`, or point the TOML configuration at an
existing prepared copy.

