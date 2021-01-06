from torch.utils.data import Dataset
from torchvision.transforms import Compose

from pytorch_unet import data_generator


class CustomDataset(Dataset):
    def __init__(
        self,
        height: int,
        width: int,
        n_samples: int,
        transform: Compose = None,
    ):
        (
            self.input_images,
            self.target_masks,
        ) = data_generator.generate_random_data(
            height=height,
            width=width,
            n_samples=n_samples,
        )
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform is not None:
            image = self.transform(image)
        return [image, mask]
