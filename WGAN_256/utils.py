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


def compute_gradient_penalty(critic, real_samples, fake_samples, device, lambda_gp=10.0):
    """
    WGAN-GP gradient penalty (Gulrajani et al., 2017).

    1. Sample epsilon ~ U(0, 1) and form interpolates between real and fake.
    2. Compute critic scores on interpolates.
    3. Take gradients of those scores w.r.t. the interpolates.
    4. Penalize deviation of the gradient L2-norm from 1:
           GP = λ * mean((||∇̂x C(̂x)||₂ - 1)²)
    """
    batch_size = real_samples.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)

    interpolates = epsilon * real_samples + (1.0 - epsilon) * fake_samples
    interpolates = interpolates.detach().requires_grad_(True)

    critic_interpolates = critic(interpolates)
    grad_outputs = torch.ones_like(critic_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1.0) ** 2).mean()
    return penalty
