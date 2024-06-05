import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CounterFactKnownDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        size: typing.Optional[int] = None,
        *args,
        **kwargs
    ):
        data_dir = Path(data_dir)

        with open(data_dir, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]