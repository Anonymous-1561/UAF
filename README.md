# UAF

User-specific Adaptive Fine-tuning for Cross-domain Recommendations

## Requirements

- Python 2.7
- TensorFlow 1.15.0
- NumPy 1.16.6
- Pandas 0.24.2
- Scikit-Learn 0.20.3

## Quick Starts

Please put the datasets in `./data` or use `--data_folder` to specify.

- Pre-train

```bash
sh run-pre.sh
```

- UAF-Hard & UAF-Soft

```bash
sh run-policy-rec.sh
```

- UAF-RL

```bash
sh run-policy-rl-rec.sh
```
