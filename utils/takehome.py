import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

# from the DeepWeeds dataset
CLASS_NAMES = ['Chinee Apple',
               'Lantana',
               'Parkinsonia',
               'Parthenium',
               'Prickly Acacia',
               'Rubber Vine',
               'Siam Weed',
               'Snake Weed',
               'Negatives']



class DWDataset(Dataset):
    def __init__(self, ds_root_folder, labels_key, transforms=None):
        super(DWDataset).__init__()
        self.labels_path = ds_root_folder / f"labels/{labels_key}.csv"
        self.images_folder = ds_root_folder / "images"
        # Filename, Label
        self.labels = pd.read_csv(self.labels_path)
        self.transforms = transforms
        self.classes = {i: n for i, n in enumerate(CLASS_NAMES)}

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img = default_loader(self.images_folder / row.Filename)
        if self.transforms is None:
            return img, row.Label
        return self.transforms(img), row.Label


def display_random_deepweed_images(dataset, num_images=4):
    """
    Display random images from DeepWeed dataset with labels from CSV.
    
    Args:
        data_dir: Path to datasets/dw-deepweed-data directory
        num_images: Number of random images to display
    """
    random_idx = random.sample(range(len(dataset)), num_images)
    
    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
    if num_images == 1:
        axes = [axes]
    
    for ax, idx in zip(axes, random_idx):
        img, label = dataset[idx]
        ax.imshow(img)
        ax.set_title(f"Label: {dataset.classes[label]}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()