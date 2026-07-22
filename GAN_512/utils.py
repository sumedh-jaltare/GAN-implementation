import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FlatImageDataset(Dataset):
    """Load images from a flat directory (no class subfolders required)."""

    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = sorted(
            os.path.join(root, name)
            for name in os.listdir(root)
            if name.lower().endswith(self.IMAGE_EXTENSIONS)
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def get_dataloader(dataroot, image_size, batch_size, workers):
    # Native 512x512 pipeline when train.py passes image_size=512 (default).
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = FlatImageDataset(root=dataroot, transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
    )


def save_generated_images(images, output_dir, start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(output_dir, f"generated_{start_index + i:04d}.png")
        array = image.detach().cpu().numpy()
        array = (array * 0.5 + 0.5).clip(0.0, 1.0)
        array = (array * 255).astype("uint8")
        array = array.transpose(1, 2, 0)
        Image.fromarray(array).save(path)
