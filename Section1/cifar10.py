import tarfile
import pickle
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

### DOWNLOAD CIFAR10 DATASET FROM THE UofT LINK BELOW ###
### https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

class CIFAR10Dataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        self.root = Path(root)
        archive_path = self.root / "cifar-10-python.tar.gz"
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing {archive_path}. Download it from the UofT link above.")
        ### unzip the zip file if not already unzipped
        if not (self.root / "cifar-10-batches-py").exists():
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(self.root)
        extract_dir = self.root / "cifar-10-batches-py"
        batch_names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        images, labels = [], []
        for name in batch_names:
            with (extract_dir / name).open("rb") as handle:
                entry = pickle.load(handle, encoding="latin1")
            data = torch.tensor(entry["data"], dtype=torch.float32).view(-1, 3, 32, 32) / 255.0
            targets = torch.tensor(entry["labels"], dtype=torch.long)
            images.append(data)
            labels.append(targets)
        self.images = torch.cat(images, dim=0)
        self.labels = torch.cat(labels, dim=0)

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, index):
        return {"images": self.images[index], "labels": self.labels[index]}


# --- Added for KID calculation ---\n")
class ArrayDataset(Dataset):
    def __init__(self, array):
        self.array = array
    def __len__(self):
        return self.array.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.array[idx]).permute(2, 0,1)

def load_cifar10(
    root: str,
    batch_size: int = 64,
    train: bool = True,
    shuffle: bool = None,
    num_workers: int = 0,
) -> DataLoader:
    dataset = CIFAR10Dataset(root=root, train=train)
    shuffle = train if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

