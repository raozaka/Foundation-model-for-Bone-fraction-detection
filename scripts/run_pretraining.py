import torch
from src.datasets import SSLDataset
from src.models import SimCLR
from src.losses import nt_xent
import torchvision.transforms as T
from torch.utils.data import DataLoader

DATA_ROOT = "path_to_dataset"

transform = T.Compose([...])  # define augmentations

dataset = SSLDataset(DATA_ROOT, transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

model = SimCLR().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(10):
    for v1, v2 in loader:
        v1, v2 = v1.cuda(), v2.cuda()
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss = nt_xent(z1, z2)
        opt.zero_grad()
        loss.backward()
        opt.step()
