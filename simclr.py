"""SimCLR pieces: encoder, projection head, augmentations, NT-Xent loss."""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode

PX = 96
AUG_SEED = 123


def build_encoder():
    """torchvision resnet18(weights=None) with fc = Identity (512-d output); no pretrained weights."""
    enc = resnet18(weights=None)
    enc.fc = nn.Identity()
    return enc


class ProjectionHead(nn.Module):
    """2-layer MLP 512 -> 512 -> 128, used for pretraining only; never evaluated."""

    def __init__(self, in_dim=512, hidden=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim))

    def forward(self, h):
        """Project encoder features into the contrastive space."""
        return self.net(h)


class SimCLRAugment:
    """Defect-preserving augment; every random draw comes from its own torch.Generator."""

    def __init__(self, px=PX, aug_seed=AUG_SEED, scale=(0.9, 1.0), brightness=0.3, contrast=0.3):
        self.px, self.scale = px, scale
        self.brightness, self.contrast = brightness, contrast
        self.gen = torch.Generator().manual_seed(aug_seed)   # separate from the model seed

    def _uniform(self, lo, hi):
        """One float in [lo, hi) from the augmentation generator."""
        return lo + (hi - lo) * torch.rand(1, generator=self.gen).item()

    def _coin(self, p):
        """True with probability p, from the augmentation generator."""
        return torch.rand(1, generator=self.gen).item() < p

    def _crop_box(self, h, w):
        """Square crop box (top, left, height, width) with area scale in self.scale; aspect 1.0 keeps castings round."""
        side = min(round(math.sqrt(h * w * self._uniform(*self.scale))), h, w)
        top = int(torch.randint(0, h - side + 1, (1,), generator=self.gen))
        left = int(torch.randint(0, w - side + 1, (1,), generator=self.gen))
        return top, left, side, side

    @staticmethod
    def border_median(img):
        """Median of the outermost pixel ring; used to fill rotated corners instead of black."""
        ring = torch.cat([img[..., 0, :], img[..., -1, :], img[..., 1:-1, 0], img[..., 1:-1, -1]], dim=-1)
        return ring.median().item()

    def __call__(self, img):
        """Augment one float image (1, H, W) in [0, 1]; returns (1, px, px) in [0, 1]."""
        fill = self.border_median(img)
        top, left, ch, cw = self._crop_box(*img.shape[-2:])
        img = TF.resized_crop(img, top, left, ch, cw, [self.px, self.px], antialias=True)
        img = TF.rotate(img, self._uniform(0.0, 360.0), interpolation=InterpolationMode.BILINEAR, fill=[fill])
        if self._coin(0.5):
            img = TF.hflip(img)
        if self._coin(0.5):
            img = TF.vflip(img)
        img = TF.adjust_brightness(img, self._uniform(1 - self.brightness, 1 + self.brightness))
        img = TF.adjust_contrast(img, self._uniform(1 - self.contrast, 1 + self.contrast))
        return img.clamp(0.0, 1.0)   # no blur: it can erase fine cracks and pits


def to_float(images_u8):
    """uint8 (n, H, W) -> float tensor (n, 1, H, W) in [0, 1]."""
    return torch.from_numpy(np.asarray(images_u8)).float().div(255.0).unsqueeze(1)


def to_model_input(x):
    """Fixed normalization (mean 0.5, std 0.5) and replicate grayscale to 3 channels."""
    return ((x - 0.5) / 0.5).repeat(1, 3, 1, 1)


def two_views(batch, augment):
    """Return two independently augmented views of a (n, 1, H, W) batch."""
    v1 = torch.stack([augment(x) for x in batch])
    v2 = torch.stack([augment(x) for x in batch])
    return v1, v2


def negatives_per_anchor(batch_size):
    """Each of the 2N views has 1 positive and 2N - 2 negatives."""
    return 2 * batch_size - 2


def nt_xent(z1, z2, tau):
    """NT-Xent on L2-normalized projections with cosine similarity; 2N-2 negatives per anchor."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)          # (2N, d), unit length
    sim = z @ z.T / tau                                    # cosine similarity / tau
    sim.fill_diagonal_(float("-inf"))                      # a view is not its own negative
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])   # view i pairs with i +/- N
    return F.cross_entropy(sim, targets)


def sanity_check(batch=64, tau=0.2, seed=0):
    """Print shapes, negatives per anchor, and NT-Xent on random vs identical views."""
    torch.manual_seed(seed)
    enc, head = build_encoder().eval(), ProjectionHead().eval()
    with torch.no_grad():
        h = enc(torch.randn(batch, 3, PX, PX))
        z = head(h)
        z_a, z_b = torch.randn(batch, 128), torch.randn(batch, 128)
        rows = [("encoder output shape", tuple(h.shape)),
                ("projection shape", tuple(z.shape)),
                ("negatives per anchor (2N - 2)", negatives_per_anchor(batch)),
                ("loss, random embeddings", round(nt_xent(z_a, z_b, tau).item(), 3)),
                ("  reference log(2N - 1)", round(math.log(2 * batch - 1), 3)),
                ("loss, identical views", round(nt_xent(z_a, z_a.clone(), tau).item(), 3))]
    print(f"NT-Xent sanity check (batch {batch}, tau {tau})")
    for name, value in rows:
        print(f"  {name:<32} {value}")


def save_aug_check(images_u8, augment, n_views=6, path="results/aug_check.png"):
    """Grid: one row per defect image, the original then n_views augmented views."""
    x = to_float(images_u8)
    fig, axes = plt.subplots(len(x), n_views + 1, figsize=(2.1 * (n_views + 1), 2.3 * len(x)))
    for r, img in enumerate(x):
        tiles = [img] + [augment(img) for _ in range(n_views)]
        for c, tile in enumerate(tiles):
            ax = axes[r, c]
            ax.imshow(tile[0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title("original" if c == 0 else f"view {c}", fontsize=14)
    fig.suptitle("SimCLR augmentations on defect images (train pool, 96 px)", fontsize=16)
    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if "--sanity" in sys.argv:
        sanity_check()
        sys.exit(0)
    import data
    imgs, labels = data.load_images("train")
    defects = imgs[labels == 1][:4]   # first 4 defect images by path order, train pool only
    save_aug_check(defects, SimCLRAugment())
    print(f"train images cached: {imgs.shape}, wrote results/aug_check.png")
