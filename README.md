# GAN Implementation

DCGAN implementations for generating images at three resolutions: **128×128**, **256×256**, and **512×512**.

## Project layout

```
GAN_128/   # 128×128 model, training, and generation
GAN_256/   # 256×256 model, training, and generation
GAN_512/   # 512×512 model, training, and generation
```

Each folder contains:

| File | Description |
|------|-------------|
| `models.py` | Generator and Discriminator |
| `train.py` | Training loop and CLI |
| `generate.py` | Sample generation from a checkpoint |
| `utils.py` | Data loading helpers |
| `requirements.txt` | Python dependencies |

## Setup

```bash
cd GAN_128   # or GAN_256 / GAN_512
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place training images under `data/train/` (or pass `--dataroot`).

## Train

```bash
python train.py
```

Useful flags (defaults differ by resolution):

```bash
python train.py --epochs 100 --batch-size 32 --image-size 128
```

Checkpoints are written to `checkpoints/`. Sample grids go to `training_samples/`.

## Generate

```bash
python generate.py --checkpoint checkpoints/<your_checkpoint>.pth --output-dir output --num-images 16
```

## Requirements

- Python 3.9+
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `Pillow>=9.0.0`
