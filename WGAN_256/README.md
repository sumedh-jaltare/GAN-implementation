# WGAN_256

## Experiment card

| Field | Value |
|-------|-------|
| Model | WGAN-GP |
| Resolution | 256×256 |
| Dataset | `../dataset/train` |
| Epochs | 100 |
| Optimizer | Adam |
| Learning Rate | 0.0001 (G and Critic) |
| Adam betas | (0.0, 0.9) |
| Batch Size | 16 (fallback 8) |
| Latent Dimension | 100 |
| n_critic | 5 |
| λ (gradient penalty) | 10 |
| Final Result | Not started |
| Notes | Critic has no Sigmoid; BCE removed; GP enforced |

## Run

```bash
cd WGAN_256
python train.py
python generate.py --checkpoint checkpoints/generator_latest.pth --num-images 1
```

Resume:

```bash
python train.py --resume checkpoints/generator_latest.pth
```
