import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


"""
CSV will contain -- 

folder name, label
10_12_walking_happy_1, 0
10_12_walking_happy_2, 0
10_12_walking_happy_3, 0

"""

class DeepFakeSmallDataset(Dataset):
    """Real Fake images small sample dataset"""

    def __init__(self, csv_file, root_dir, frames, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with binary labels (real: 1, fake: 0).
            root_dir (string): Directory with all the real and fake images derived from sequences of mouth movement.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.data)

    def read_sequence(self, path, selected_folder, use_transform, idx):
        X = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i))) # TODO?

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)
        return X

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = self.data.iloc[idx, 1]
        images = self.read_sequence(self.root_dir, self.data.iloc[idx, 0], self.transform, idx)
        sequence_name = self.data.iloc[idx, 0]
        return images, label, sequence_name
