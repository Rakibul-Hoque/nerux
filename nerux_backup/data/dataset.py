import numpy as np

from ..tensor import Tensor
from ..tensor import functional as F
from ..tensor import factory as init


class BaseDataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

    def subset(self, indices):
        return Subset(self, indices)

    def split(self, ratio=0.8, shuffle=True, seed=None):
        n = len(self)
        indices = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        cut = int(n * ratio)
        return Subset(self, indices[:cut]), Subset(self, indices[cut:])


class Dataset(BaseDataset):
    def __init__(self, x, y=None, transform=None, target_transform=None):
        self.x = x if isinstance(x, Tensor) else init.tensor(x)
        self.y = None
        if y is not None:
            self.y = y if isinstance(y, Tensor) else init.tensor(y)
            if len(self.x) != len(self.y):
                raise ValueError(
                    f"x and y are in different size. len(x)={len(self.x)} != len(y)={len(self.y)}"
                )
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx] if self.y is not None else None

        if self.transform is not None:

            x = self.transform(x.unsqueeze(axis=0))[0]
        if y is not None and self.target_transform is not None:

            y = self.target_transform(y.unsqueeze(axis=0))[0]

        return (x, y) if y is not None else x

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        return (
            f"TensorDataset(samples={len(self)}, "
            f"x_shape={self.x.shape}, "
            f"y_shape={self.y.shape if self.y is not None else None})"
        )

    def save(self, path: str):
        """
        Save raw (pre-transform) tensors to a .npz file.
        Load back with Dataset.load(path).

        Args:
            path: file path; .npz extension added automatically if missing.
        """
        arrays = {"x": self.x.data}
        if self.y is not None:
            arrays["y"] = self.y.data
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str, transform=None, target_transform=None):
        """
        Load a dataset saved with .save().

        Args:
            path:             .npz file path (with or without extension)
            transform:        optional transform to attach
            target_transform: optional target transform to attach
        """
        if not path.endswith(".npz"):
            path = path + ".npz"
        data = np.load(path)
        x = data["x"]
        y = data["y"] if "y" in data else None
        return cls(x, y, transform=transform, target_transform=target_transform)


class Subset(BaseDataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.array(indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ConcatDataset(BaseDataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self._cumulative = np.cumsum([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        dataset_idx = np.searchsorted(self._cumulative, idx, side="right")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self._cumulative[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return int(self._cumulative[-1])
