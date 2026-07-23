# GAN_256

## Experiment card

| Field | Value |
|-------|-------|
| Model | DCGAN |
| Resolution | 256×256 |
| Dataset | `../dataset/train` |
| Epochs | 100 |
| Optimizer | Adam |
| Learning Rate | G: 0.0002 / D: 0.0001 |
| Batch Size | 16 (fallback 8) |
| Latent Dimension | 100 |
| Final Result | Completed |
| Notes | Separate generator/discriminator learning rates |

## Run

```bash
cd GAN_256
python train.py
python generate.py --checkpoint checkpoints/generator_latest.pth --num-images 1
```
