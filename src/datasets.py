import os, glob
from PIL import Image
from torch.utils.data import Dataset

class SSLDataset(Dataset):
    def __init__(self, root, transform):
        self.paths = []
        for cls in ["fractured", "not_fractured"]:
            d = os.path.join(root, "training", cls)
            self.paths += glob.glob(os.path.join(d, "*"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)
