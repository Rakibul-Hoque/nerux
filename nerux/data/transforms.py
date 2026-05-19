import numpy as np
from ..tensor import Tensor
from ..tensor import functional as F
from ..tensor import factory as init

# ── Marker mixin ──────────────────────────────────────────────────────────────


class RandomTransform:
    """
    Inherit this to mark a transform as stochastic.
    The dataset cache layer uses this to skip caching random transforms.
    """

    pass


# ── Base / Compose ────────────────────────────────────────────────────────────


class Custom:
    """Subclass this to write a custom transform."""

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Compose:
    """
    Chain transforms sequentially.
    Works on batched tensors — every child transform must also accept batches.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    @property
    def is_random(self):
        return any(isinstance(t, RandomTransform) for t in self.transforms)

    def __call__(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        lines = "\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {lines}\n])"


# ── Type / dtype ──────────────────────────────────────────────────────────────


class ToTensor:
    """
    Convert numpy array or nested list to Tensor.
    Input may already be a Tensor — returned unchanged.
    Accepts (N, ...) batches or single samples.
    """

    def __call__(self, x) -> Tensor:
        if isinstance(x, Tensor):
            return x
        return init.tensor(x)

    def __repr__(self):
        return "ToTensor()"


class ToFloat:
    """Cast batch to float32."""

    def __call__(self, x: Tensor) -> Tensor:
        return x.float()

    def __repr__(self):
        return "ToFloat()"


class ToDtype:
    """Cast batch to an arbitrary numpy dtype string, e.g. 'float64', 'int32'."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x: Tensor) -> Tensor:
        return x.astype(self.dtype)

    def __repr__(self):
        return f"ToDtype({self.dtype})"


# ── Normalisation ─────────────────────────────────────────────────────────────


class Normalize:
    """
    Normalise a batch: (x - mean) / (std + eps).

    mean / std can be:
      - scalars          → applied to every element
      - 1-D arrays       → per-channel, broadcast over (N, C, ...)
      - Tensors          → used directly

    Batch dim is axis 0 and is never touched.
    """

    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        mean = self.mean
        std = self.std

        return (x - mean) / (std + self.eps)

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class StandardScaler:
    """
    Fit mean / std on a full dataset array then use as a transform.

    fit() expects the entire dataset tensor (N, ...).
    transform() accepts the same shape or any prefix-compatible batch.
    """

    def __init__(self, eps=1e-8):
        self.mean_ = None
        self.std_ = None
        self.eps = eps

    def fit(self, x: Tensor) -> "StandardScaler":
        # axis=0 → statistics over the batch dimension
        self.mean_ = x.mean()
        self.std_ = x.std()
        return self

    def transform(self, x: Tensor) -> Tensor:
        if self.mean_ is None:
            raise RuntimeError("StandardScaler.fit() must be called before transform()")
        return (x - self.mean_) / (self.std_ + self.eps)

    def fit_transform(self, x: Tensor) -> Tensor:
        return self.fit(x).transform(x)

    def __repr__(self):
        return f"StandardScaler(fitted={self.mean_ is not None})"


class MinMaxScaler:
    """Scale batch features to [lo, hi] (default [0, 1])."""

    def __init__(self, feature_range=(0, 1), eps=1e-8):
        self.min_ = None
        self.max_ = None
        self.range = feature_range
        self.eps = eps

    def fit(self, x: Tensor) -> "MinMaxScaler":
        self.min_ = x.min()
        self.max_ = x.max()
        return self

    def transform(self, x: Tensor) -> Tensor:
        if self.min_ is None:
            raise RuntimeError("MinMaxScaler.fit() must be called before transform()")
        lo, hi = self.range
        scale = (x - self.min_) / (self.max_ - self.min_ + self.eps)
        return scale * (hi - lo) + lo

    def fit_transform(self, x: Tensor) -> Tensor:
        return self.fit(x).transform(x)

    def __repr__(self):
        return f"MinMaxScaler(range={self.range}, fitted={self.min_ is not None})"


# ── Shape ─────────────────────────────────────────────────────────────────────


class Flatten:
    """
    Flatten every sample in a batch independently.
    (N, d1, d2, ...) → (N, d1*d2*...)
    """

    def __call__(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        return x.reshape(N, -1)

    def __repr__(self):
        return "Flatten()"


class Reshape:
    """
    Reshape each sample in the batch.
    Pass the per-sample shape — the batch dim (N) is prepended automatically.
    e.g. Reshape(3, 32, 32) on a batch turns (N, 3072) → (N, 3, 32, 32)
    """

    def __init__(self, *shape):
        self.shape = shape

    def __call__(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        return x.reshape(N, *self.shape)

    def __repr__(self):
        return f"Reshape{self.shape}"


class Squeeze:
    """Remove a size-1 dimension. Operates on the full batch tensor."""

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, x: Tensor) -> Tensor:
        return x.squeeze(self.axis)

    def __repr__(self):
        return f"Squeeze(axis={self.axis})"


class Unsqueeze:
    """Insert a size-1 dimension. Operates on the full batch tensor."""

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, x: Tensor) -> Tensor:
        return x.unsqueeze(self.axis)

    def __repr__(self):
        return f"Unsqueeze(axis={self.axis})"


class Permute:
    """
    Permute axes of the full batch tensor.
    Include the batch axis (0) in the permutation.
    e.g. Permute(0, 3, 1, 2) converts (N, H, W, C) → (N, C, H, W)
    """

    def __init__(self, *axes):
        self.axes = axes

    def __call__(self, x: Tensor) -> Tensor:
        return x.transpose(self.axes)

    def __repr__(self):
        return f"Permute{self.axes}"


# ── Encoding ──────────────────────────────────────────────────────────────────


class OneHotEncoder:
    """
    Convert integer label batch to one-hot.

    dimension=2 : (N,)   or (N, 1) → (N, num_classes)
    dimension=3 : (N, L)           → (N, L, num_classes)   token sequences
    """

    def __init__(self, num_classes, dimension=2):
        self.num_classes = num_classes
        self.dimension = dimension

    def __call__(self, x: Tensor) -> Tensor:
        data = x.data.astype(int)

        if self.dimension == 2:
            # (N,) or (N, 1) → (N, C)
            idx = data.flatten()  # (N,)
            out = np.zeros((len(idx), self.num_classes), dtype=np.float32)
            out[np.arange(len(idx)), idx] = 1.0
            return init.tensor(out)

        if self.dimension == 3:
            # (N, L) → (N, L, C)
            N, L = data.shape
            out = np.zeros((N, L, self.num_classes), dtype=np.float32)
            out[np.arange(N)[:, None], np.arange(L)[None, :], data] = 1.0
            return init.tensor(out)

        raise ValueError(f"OneHotEncoder: unsupported dimension={self.dimension}")

    def __repr__(self):
        return (
            f"OneHotEncoder(num_classes={self.num_classes}, dimension={self.dimension})"
        )


# ── Numeric augmentations ─────────────────────────────────────────────────────


class RandomNoise(RandomTransform):
    """
    Add i.i.d. Gaussian noise to every sample in the batch.
    (N, ...) → (N, ...)
    """

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x: Tensor) -> Tensor:
        # Tensor.randn accepts a shape tuple — same shape as the whole batch
        return x + init.randn(x.shape) * self.std

    def __repr__(self):
        return f"RandomNoise(std={self.std})"


class RandomErasing(RandomTransform):
    """
    Randomly zero out a rectangular patch in each image in the batch.
    Expects (N, C, H, W) or (N, H, W).
    Each sample in the batch is erased independently.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, x: Tensor) -> Tensor:
        data = x.data.copy()  # numpy needed for indexing
        N = data.shape[0]
        H = data.shape[-2]
        W = data.shape[-1]

        for n in range(N):
            if np.random.rand() >= self.p:
                continue
            area = H * W
            for _ in range(10):  # 10 attempts to find valid patch
                erase_area = np.random.uniform(*self.scale) * area
                aspect = np.random.uniform(*self.ratio)
                eh = int(round((erase_area * aspect) ** 0.5))
                ew = int(round((erase_area / aspect) ** 0.5))
                if eh < H and ew < W:
                    top = np.random.randint(0, H - eh)
                    left = np.random.randint(0, W - ew)
                    data[n, ..., top : top + eh, left : left + ew] = self.value
                    break
        return init.tensor(data)

    def __repr__(self):
        return f"RandomErasing(p={self.p}, scale={self.scale})"


class RandomMixup(RandomTransform):
    """
    Mixup augmentation on a batch.
    Blends pairs of samples using Beta(alpha, alpha) weights.
    Returns mixed_x — call with (x, y) via Lambda if you need mixed labels too.

    Usage:
        mixup = RandomMixup(alpha=0.4)
        x_mixed, lam = mixup(x)          # returns (Tensor, float)
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, x: Tensor):
        lam = float(np.random.beta(self.alpha, self.alpha))
        idx = np.random.permutation(x.shape[0])
        perm = init.tensor(x.data[idx])  # shuffled copy
        return x * lam + perm * (1 - lam), lam

    def __repr__(self):
        return f"RandomMixup(alpha={self.alpha})"


class RandomCutout(RandomTransform):
    """
    Cut out n_holes square patches of size length×length from each image.
    Expects (N, C, H, W) or (N, H, W). Fills with fill_value (default 0).
    """

    def __init__(self, n_holes=1, length=16, fill_value=0.0):
        self.n_holes = n_holes
        self.length = length
        self.fill_value = fill_value

    def __call__(self, x: Tensor) -> Tensor:
        data = x.data.copy()
        N, H, W = data.shape[0], data.shape[-2], data.shape[-1]

        for n in range(N):
            for _ in range(self.n_holes):
                cy = np.random.randint(H)
                cx = np.random.randint(W)
                top = max(0, cy - self.length // 2)
                bot = min(H, cy + self.length // 2)
                left = max(0, cx - self.length // 2)
                rght = min(W, cx + self.length // 2)
                data[n, ..., top:bot, left:rght] = self.fill_value

        return init.tensor(data)

    def __repr__(self):
        return f"RandomCutout(n_holes={self.n_holes}, length={self.length})"


# ── Spatial augmentations (image batches) ─────────────────────────────────────


class RandomHorizontalFlip(RandomTransform):
    """
    Flip each image in the batch horizontally with probability p.
    (N, ..., H, W) — each sample flipped independently.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        data = x.data.copy()
        mask = np.random.rand(data.shape[0]) < self.p  # (N,) boolean
        data[mask] = data[mask, ..., ::-1].copy()
        return init.tensor(data)

    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip(RandomTransform):
    """
    Flip each image in the batch vertically with probability p.
    (N, ..., H, W) — each sample flipped independently.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        data = x.data.copy()
        mask = np.random.rand(data.shape[0]) < self.p
        data[mask] = data[mask, ..., ::-1, :].copy()
        return init.tensor(data)

    def __repr__(self):
        return f"RandomVerticalFlip(p={self.p})"


class RandomCrop(RandomTransform):
    """
    Crop each image in the batch to (crop_h, crop_w) at an independently
    sampled random position.
    Expects (N, ..., H, W).
    """

    def __init__(self, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, x: Tensor) -> Tensor:
        data = x.data
        N = data.shape[0]
        H, W = data.shape[-2], data.shape[-1]

        tops = np.random.randint(0, H - self.crop_h + 1, size=N)
        lefts = np.random.randint(0, W - self.crop_w + 1, size=N)

        out = np.stack(
            [
                data[
                    n,
                    ...,
                    tops[n] : tops[n] + self.crop_h,
                    lefts[n] : lefts[n] + self.crop_w,
                ]
                for n in range(N)
            ]
        )
        return init.tensor(out)

    def __repr__(self):
        return f"RandomCrop({self.crop_h}, {self.crop_w})"


class CenterCrop:
    """
    Crop the centre (crop_h, crop_w) region from every image in the batch.
    Expects (N, ..., H, W).
    """

    def __init__(self, crop_h, crop_w):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2], x.shape[-1]
        top = (H - self.crop_h) // 2
        left = (W - self.crop_w) // 2
        # pure Tensor slice — no numpy needed
        return x[..., top : top + self.crop_h, left : left + self.crop_w]

    def __repr__(self):
        return f"CenterCrop({self.crop_h}, {self.crop_w})"


class Pad:
    """
    Zero-pad spatial dims of a batch by pad_h / pad_w pixels on each side.
    Expects (N, C, H, W) or (N, H, W).
    Uses numpy only because your Tensor class has no pad op.
    """

    def __init__(self, pad_h, pad_w=None, value=0.0):
        self.pad_h = pad_h
        self.pad_w = pad_w if pad_w is not None else pad_h
        self.value = value

    def __call__(self, x: Tensor) -> Tensor:
        data = x.data
        ph, pw = self.pad_h, self.pad_w

        if data.ndim == 4:  # (N, C, H, W)
            pad_width = ((0, 0), (0, 0), (ph, ph), (pw, pw))
        elif data.ndim == 3:  # (N, H, W)
            pad_width = ((0, 0), (ph, ph), (pw, pw))
        else:
            raise ValueError(f"Pad expects 3D or 4D batch, got {data.ndim}D")

        return init.tensor(np.pad(data, pad_width, constant_values=self.value))

    def __repr__(self):
        return f"Pad(pad_h={self.pad_h}, pad_w={self.pad_w})"


class RandomRotation90(RandomTransform):
    """
    Rotate each image in the batch by a random multiple of 90°.
    Expects (N, ..., H, W). Uses numpy.rot90.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        data = x.data.copy()
        for n in range(data.shape[0]):
            if np.random.rand() < self.p:
                k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
                data[n] = np.rot90(data[n], k=k, axes=(-2, -1)).copy()
        return init.tensor(data)

    def __repr__(self):
        return f"RandomRotation90(p={self.p})"


# ── Sequence / NLP ────────────────────────────────────────────────────────────


class TokenDropout(RandomTransform):
    """
    Randomly replace tokens in a sequence batch with a pad/mask token id.
    Expects integer token batch (N, L).
    Each token is dropped independently with probability p.
    """

    def __init__(self, p=0.1, mask_token=0):
        self.p = p
        self.mask_token = mask_token

    def __call__(self, x: Tensor) -> Tensor:
        data = x.data.copy()
        mask = np.random.rand(*data.shape) < self.p
        data[mask] = self.mask_token
        return init.tensor(data)

    def __repr__(self):
        return f"TokenDropout(p={self.p}, mask_token={self.mask_token})"


class TruncateOrPad:
    """
    Ensure all sequences in the batch have exactly max_len tokens.
    Truncates from the right or pads on the right with pad_token.
    Expects (N, L) integer tensor.
    """

    def __init__(self, max_len, pad_token=0):
        self.max_len = max_len
        self.pad_token = pad_token

    def __call__(self, x: Tensor) -> Tensor:
        N, L = x.shape[0], x.shape[1]
        data = x.data

        if L >= self.max_len:
            return init.tensor(data[:, : self.max_len].copy())

        pad = np.full((N, self.max_len - L), self.pad_token, dtype=data.dtype)
        return init.tensor(np.concatenate([data, pad], axis=1))

    def __repr__(self):
        return f"TruncateOrPad(max_len={self.max_len}, pad_token={self.pad_token})"


# ── Tabular / general ─────────────────────────────────────────────────────────


class DropFeatures:
    """
    Zero out specific feature columns across the whole batch.
    Expects (N, F) tensor. indices: list of column indices to drop.
    Pure Tensor operation — no numpy needed.
    """

    def __init__(self, indices):
        self.indices = list(indices)

    def __call__(self, x: Tensor) -> Tensor:
        # build a mask Tensor of ones, zero out selected columns
        mask = init.ones(x.shape, dtype=np.float32)
        mask.data[:, self.indices] = 0.0
        return x * mask

    def __repr__(self):
        return f"DropFeatures(indices={self.indices})"


class ClipValues:
    """
    Clamp all values in the batch to [min_val, max_val].
    Uses your Tensor.clip method directly.
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: Tensor) -> Tensor:
        return x.clip(self.min_val, self.max_val)

    def __repr__(self):
        return f"ClipValues({self.min_val}, {self.max_val})"


class RandomFeatureDrop(RandomTransform):
    """
    Randomly zero entire feature columns in a tabular batch (N, F).
    Each column is independently dropped with probability p.
    Different columns may be dropped for different batches,
    but within one batch all samples see the same dropped columns.
    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        f = x.shape[1]
        mask = (np.random.rand(f) >= self.p).astype(np.float32)  # (F,)
        # broadcast (F,) over (N, F)
        return x * init.tensor(mask)

    def __repr__(self):
        return f"RandomFeatureDrop(p={self.p})"


# ── Utility ───────────────────────────────────────────────────────────────────


class Lambda:
    """
    Wrap any callable as a transform.
    Set is_random=True if the function is stochastic so the cache layer
    knows not to cache its output.
    """

    def __init__(self, fn, is_random=False):
        self.fn = fn
        self._is_random = is_random

    def __call__(self, x: Tensor) -> Tensor:
        return self.fn(x)

    def __repr__(self):
        return f"Lambda({self.fn.__name__})"
