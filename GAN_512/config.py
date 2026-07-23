"""Experiment configuration for GAN_512. Values match prior defaults."""

IMAGE_SIZE = 512
BATCH_SIZE = 8
LATENT_DIM = 100
LEARNING_RATE = 0.0002
EPOCHS = 100
NUM_WORKERS = 2
BETA1 = 0.5
BETA2 = 0.999
FIXED_SAMPLE_COUNT = 16

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "output"
SAMPLE_DIR = "training_samples"
TRAINING_LOG_CSV = "training_log.csv"
DATASET_PATH = "../dataset/train"
