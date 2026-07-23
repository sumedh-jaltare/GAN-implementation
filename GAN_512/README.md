# GAN_512

## Experiment card

| Field | Value |
|-------|-------|
| Model | DCGAN |
| Resolution | 512×512 |
| Dataset | `../dataset/train` |
| Epochs | 100 |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Batch Size | 8 |
| Latent Dimension | 100 |
| Final Result | Completed |
| Notes | Higher memory / smaller batch size |

## Run

```bash
cd GAN_512
python train.py
python generate.py --checkpoint checkpoints/generator_latest.pth --num-images 1
```
