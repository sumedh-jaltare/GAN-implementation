# GAN_128

## Experiment card

| Field | Value |
|-------|-------|
| Model | DCGAN |
| Resolution | 128×128 |
| Dataset | `../dataset/train` |
| Epochs | 100 |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Batch Size | 32 |
| Latent Dimension | 100 |
| Final Result | Completed successfully |
| Notes | Baseline DCGAN run |

## Run

```bash
cd GAN_128
python train.py
python generate.py --checkpoint checkpoints/generator_latest.pth --num-images 1
```
