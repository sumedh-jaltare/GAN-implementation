# GAN Research Experiments

Reproducible DCGAN / WGAN experiments sharing one dataset.

## Layout

```
experiments/
├── README.md
├── .gitignore
├── dataset/train/          # shared training images
├── GAN_128/                # DCGAN 128×128 (completed)
├── GAN_256/                # DCGAN 256×256 (completed)
├── GAN_512/                # DCGAN 512×512 (completed)
└── WGAN_256/               # WGAN-GP 256×256
```

Each experiment folder is self-contained (`train.py`, `models.py`, `generate.py`, `utils.py`, `config.py`) and loads images from `../dataset/train`.

## Experiment summary

| Experiment | Model | Resolution | Status | Batch | LR | Epochs |
|------------|-------|------------|--------|-------|-----|--------|
| `GAN_128` | DCGAN | 128×128 | Completed | 32 | 0.0002 | 100 |
| `GAN_256` | DCGAN | 256×256 | Completed | 16 | G 0.0002 / D 0.0001 | 100 |
| `GAN_512` | DCGAN | 512×512 | Completed | 8 | 0.0002 | 100 |
| `WGAN_256` | WGAN-GP | 256×256 | Ready to train | 16 | 0.0001 (β=(0,0.9)), n_critic=5, λ=10 | 100 |

## Setup

```bash
cd GAN_128   # or GAN_256 / GAN_512 / WGAN_256
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place training images in `dataset/train/` (already shared across experiments).

## Train / generate

```bash
cd GAN_128
python train.py
python generate.py --checkpoint checkpoints/generator_latest.pth --num-images 1
```

Resume:

```bash
python train.py --resume checkpoints/generator_latest.pth
```

Hyperparameters and paths live in each experiment’s `config.py`.
