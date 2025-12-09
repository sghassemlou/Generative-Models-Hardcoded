

# --- Added for KID calculation ---
import numpy as np
class ArrayDataset(Dataset):
    def __init__(self, array):
        self.array = array
    def __len__(self):
        return self.array.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.array[idx]).permute(2, 0, 1)
