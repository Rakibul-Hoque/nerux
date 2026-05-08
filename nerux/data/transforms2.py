import numpy as np
from ..tensor import Tensor


class Custom:
    def __call__(self, x):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        lines = "\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {lines}\n])"


class ToTensor:
    def __call__(self, x):
        if isinstance(x, Tensor):
            return x
        return Tensor(np.array(x))

    def __repr__(self):
        return "ToTensor()"


class ToFloat:
    def __call__(self, x):
        return x.float()

    def __repr__(self):
        return "ToFloat()"


class Normalize:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class StandardScaler:
    """
    Fit mean/std from data then normalize.
    Usage:
        scaler = StandardScaler()
        scaler.fit(x_train)           # x_train: np.ndarray or Tensor
        transform = scaler.transform  # use as a transform callable
    """

    def __init__(self, eps=1e-8):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        return self

    def transform(self, x):
        return (x - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def __repr__(self):
        return f"StandardScaler(fitted={self.mean_ is not None})"


class MinMaxScaler:
    """Scale features to [0, 1] (or custom range)."""

    def __init__(self, feature_range=(0, 1), eps=1e-8):
        self.min_ = None
        self.max_ = None
        self.range = feature_range
        self.eps = eps

    def fit(self, x):
        self.min_ = x.min(axis=0)
        self.max_ = x.max(axis=0)
        return self

    def transform(self, x):
        lo, hi = self.range
        scale = (x - self.min_) / (self.max_ - self.min_ + self.eps)
        return scale * (hi - lo) + lo

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def __repr__(self):
        return f"MinMaxScaler(range={self.range}, fitted={self.min_ is not None})"


class Flatten:
    def __call__(self, x):
        return x.flatten()

    def __repr__(self):
        return "Flatten()"


class Reshape:
    def __init__(self, *shape):
        self.shape = shape

    def __call__(self, x):
        return x.reshape(*self.shape)

    def __repr__(self):
        return f"Reshape{self.shape}"


# ── Augmentation transforms (operate on raw numpy / Tensor alike) ──────────
class OneHotEncoder:
    def __init__(self, num_classes, dimantion=2):
        self.num_classes = num_classes
        self.dimantion = dimantion

    def __call__(self, x):
        data = x.data if hasattr(x, "data") else x
        if self.dimantion == 2:
            one_hot_arr = Tensor.zeros((self.num_classes))
            one_hot_arr[int(data)] = 1
            return one_hot_arr
        if self.dimantion == 3:
            sequence = data.size
            one_hot_arr = np.zeros((sequence, self.num_classes))
            one_hot_arr[np.arange(sequence), data] = 1

            return Tensor(one_hot_arr)

    def __repr__(self):
        return f"OneHotEncoder(num_classes={self.num_classes})"


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        data = x.data if hasattr(x, "data") else np.array(x)
        if np.random.rand() < self.p:
            data = np.flip(data, axis=-1).copy()
        return Tensor(data)

    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        data = x.data if hasattr(x, "data") else np.array(x)
        if np.random.rand() < self.p:
            data = np.flip(data, axis=-2).copy()
        return Tensor(data)

    def __repr__(self):
        return f"RandomVerticalFlip(p={self.p})"


class RandomNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        return x + (Tensor.randn(x.shape) * self.std)

    def __repr__(self):
        return f"RandomNoise(std={self.std})"


class RandomCrop:
    """
    Randomly crop a 2D/3D array to (crop_h, crop_w).
    Expects shape (..., H, W) or (H, W).
    """

    def __init__(self, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, x):
        data = x.data if hasattr(x, "data") else np.array(x)
        h, w = data.shape[-2], data.shape[-1]
        top = np.random.randint(0, h - self.crop_h + 1)
        left = np.random.randint(0, w - self.crop_w + 1)
        cropped = data[..., top : top + self.crop_h, left : left + self.crop_w]
        return Tensor(cropped.copy())

    def __repr__(self):
        return f"RandomCrop({self.crop_h}, {self.crop_w})"


class Lambda:
    """Wrap any callable as a transform."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)

    def __repr__(self):
        return f"Lambda({self.fn.__name__})"
