import argparse

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as f
import torchvision
from tqdm import tqdm
from itertools import product

from simclr.modules.transformations import TransformsSimCLR
from simclr import SimCLR
from simclr.modules.resnet import get_resnet
from simclr.modules import LogisticRegression
from utils.takehome import DWDataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# from DeepWeeds
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
CLASS_NAMES = ['Chinee Apple',
               'Lantana',
               'Parkinsonia',
               'Parthenium',
               'Prickly Acacia',
               'Rubber Vine',
               'Siam Weed',
               'Snake Weed',
               'Negatives']
classes = dict(zip(CLASSES, CLASS_NAMES))

def inference(loader, simclr_model, device):
    input_vector, labels_vector, h_vector, z_vector = [], [], [], []

    for step, (x, y) in enumerate(loader):
        x = x.to(next(simclr_model.parameters()).device)
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)
            input_vector.append(x.cpu().numpy())
            labels_vector.append(y.cpu().numpy())
            h_vector.append(h.cpu().numpy())
            z_vector.append(z.cpu().numpy())

    return [
        np.concatenate(input_vector, axis=0), 
        np.concatenate(h_vector, axis=0), 
        np.concatenate(z_vector, axis=0), 
        np.concatenate(labels_vector, axis=0)
    ]


def cache_features(loader, feature_dir, feature_suff, device):
    def _cache(paths, arrays):

        if all(os.path.exists(p) for p in paths):
            print(f"skipping since {paths} exist")
            return

        print(f"Saving {len(paths)} features to {feature_dir}...")
        for path, array in zip(paths, arrays):
            print(f"\t {path}: {array.shape}, {array.dtype}")
            np.save(path, array)

    
    feature_dir.mkdir(parents=False, exist_ok=True)

    encoder = get_resnet("resnet50", pretrained=False)
    simclr_model = SimCLR(encoder, 64, encoder.fc.in_features)
    simclr_model.load_state_dict(torch.load("checkpoint_100.tar", map_location=device, weights_only=True))
    simclr_model = simclr_model.to(device)
    simclr_model = simclr_model.eval()
    # remove the last projection layer
    simclr_model.projector.pop(2)
    X, H, Z1, labels = inference(loader, simclr_model, device)
    _cache([feature_dir / f"X_{feature_suff}.npy", 
            feature_dir / f"H_{feature_suff}.npy", 
            feature_dir / f"Z1_{feature_suff}.npy", 
            feature_dir / f"y_{feature_suff}.npy"], 
           [X, H, Z1, labels])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser.add_argument("model_checkpoint", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    

    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    label_csv_paths = (list((args.dataset_dir / "labels").glob("test*csv")) + 
                       list((args.dataset_dir / "labels").glob("val*csv")) + 
                       list((args.dataset_dir / "labels").glob("train*csv")))
    for csv_path in label_csv_paths:
        dataset = DWDataset(args.dataset_dir, csv_path.stem, TransformsSimCLR(size=224).train_transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=8,)
        cache_features(loader, args.output_dir, csv_path.stem, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))



