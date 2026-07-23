"""Experiment configuration for WGAN_256 (WGAN-GP)."""

IMAGE_SIZE = 256
BATCH_SIZE = 16
FALLBACK_BATCH_SIZE = 8
LATENT_DIM = 100

# WGAN-GP paper defaults for Adam on both G and Critic.
LEARNING_RATE = 0.0001
BETA1 = 0.0
BETA2 = 0.9

EPOCHS = 100
NUM_WORKERS = 2
FIXED_SAMPLE_COUNT = 16

# Critic is updated n_critic times per generator update.
N_CRITIC = 5
# Gradient penalty coefficient λ from WGAN-GP.
LAMBDA_GP = 10.0

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "output"
SAMPLE_DIR = "training_samples"
TRAINING_LOG_CSV = "training_log.csv"
DATASET_PATH = "../dataset/train"
