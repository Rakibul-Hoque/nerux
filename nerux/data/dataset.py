import numpy as np

from ..tensor import Tensor
from ..tensor import functional as F
from ..tensor import factory as init


def _is_tensor(value):
    return isinstance(value, Tensor)


def _to_tensor(value):
    return value if _is_tensor(value) else init.tensor(value)


def _length(value):
    try:
        return len(value)
    except TypeError as exc:
        raise TypeError(
            f"Expected an indexable collection, got {type(value).__name__}"
        ) from exc


def _apply_transform(transform, value):
    if transform is None:
        return value

    if isinstance(transform, (list, tuple)):
        for fn in transform:
            value = fn(value)
        return value

    return transform(value)


# BaseDataset


class BaseDataset:
    """
    Base interface for map-style datasets.

    Subclasses implement:
        __getitem__(index)
        __len__()
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

    def subset(self, indices):
        return Subset(self, indices)

    def split(self, ratio=0.8, shuffle=True, seed=None):
        """
        Split a map-style dataset into two subsets.

        Returns:
            (first_subset, second_subset)
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be between 0 and 1")

        indices = np.arange(len(self))

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        cut = int(len(self) * ratio)

        return (
            Subset(self, indices[:cut]),
            Subset(self, indices[cut:]),
        )


# Dataset


class Dataset(BaseDataset):
    """
    General-purpose in-memory dataset.

    Supported formats:

    1. Dataset(x, y)
    2. Dataset({"image": images, "label": labels})
    3. Dataset(tensors=(images, features, labels))
    4. Dataset(x, transform=...)
    5. Dataset(data, transforms={field: transform})

    For named datasets, every field must have the same length.

    Field transforms receive individual samples, not batches.
    """

    def __init__(
        self,
        data=None,
        y=None,
        *,
        tensors=None,
        transform=None,
        target_transform=None,
        transforms=None,
        sample_transform=None,
    ):
        if data is not None and tensors is not None:
            raise ValueError("Provide either data or tensors, not both")

        if data is None and tensors is None:
            raise ValueError("Either data or tensors must be provided")

        if tensors is not None:
            if isinstance(tensors, dict):
                data = tensors
            else:
                data = tuple(tensors)

        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms or {}
        self.sample_transform = sample_transform

        self._mode = None

        # Named fields

        if isinstance(data, dict):
            if not data:
                raise ValueError("Dataset cannot be empty")

            self._mode = "dict"
            self.data = {key: self._prepare_field(value) for key, value in data.items()}

            lengths = {key: _length(value) for key, value in self.data.items()}

            unique_lengths = set(lengths.values())

            if len(unique_lengths) != 1:
                raise ValueError(
                    f"All dataset fields must have equal lengths: {lengths}"
                )

            if y is not None:
                raise ValueError(
                    "Do not provide y when data is a dictionary. "
                    "Include the target as another field."
                )

            self._length = next(iter(unique_lengths))

        # Multiple positional fields

        elif isinstance(data, tuple):
            if not data:
                raise ValueError("Dataset cannot be empty")

            self._mode = "tuple"
            self.data = tuple(self._prepare_field(value) for value in data)

            lengths = [_length(value) for value in self.data]

            if len(set(lengths)) != 1:
                raise ValueError(
                    f"All dataset fields must have equal lengths: {lengths}"
                )

            if y is not None:
                raise ValueError(
                    "Do not provide y when data is a tuple. "
                    "Include the target in the tuple."
                )

            self._length = lengths[0]

        # Traditional x, y API

        else:
            self._mode = "xy"
            self.x = self._prepare_field(data)
            self.y = self._prepare_field(y) if y is not None else None

            self._length = _length(self.x)

            if self.y is not None:
                if _length(self.y) != self._length:
                    raise ValueError(
                        f"x and y have different lengths: "
                        f"{self._length} != {_length(self.y)}"
                    )

    @staticmethod
    def _prepare_field(value):
        """
        Convert arrays and numerical data into Nerux tensors.

        Existing Nerux tensors are preserved.
        """
        if _is_tensor(value):
            return value

        if isinstance(value, np.ndarray):
            return init.tensor(value)

        # Lists and tuples can represent collections of samples.
        # Preserve their outer sample structure.
        if isinstance(value, (list, tuple)):
            return value

        return init.tensor(value)

    def __len__(self):
        return self._length

    def _get_raw_sample(self, index):
        if self._mode == "dict":
            return {key: value[index] for key, value in self.data.items()}

        if self._mode == "tuple":
            return tuple(value[index] for value in self.data)

        x = self.x[index]
        y = self.y[index] if self.y is not None else None

        if self.y is None:
            return x

        return x, y

    def _apply_field_transforms(self, sample):
        if self._mode == "dict":
            result = dict(sample)

            for key, transform in self.transforms.items():
                if key not in result:
                    raise KeyError(f"Transform references unknown field: {key}")

                result[key] = _apply_transform(
                    transform,
                    result[key],
                )

            if self.transform is not None:
                result = self.transform(result)

            return result

        if self._mode == "tuple":
            if self.transforms:
                raise ValueError(
                    "Use positional transforms through a sequence "
                    "of callables for tuple datasets."
                )

            result = sample

            if self.transform is not None:
                result = self.transform(result)

            return result

        # Legacy x, y dataset.
        x, y = sample if self.y is not None else (sample, None)

        x = _apply_transform(self.transform, x)

        if y is not None:
            y = _apply_transform(self.target_transform, y)

        return (x, y) if y is not None else x

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Dataset indices must be integers")

        index = int(index)

        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range")

        sample = self._get_raw_sample(index)

        sample = self._apply_field_transforms(sample)

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)

        return sample

    def __repr__(self):
        return f"Dataset(length={len(self)}, mode={self._mode!r})"

    # --------------------------------------------------------
    # Serialization
    # --------------------------------------------------------

    def save(self, path):
        """
        Save supported in-memory tensor datasets to NPZ.

        Only raw tensor fields are supported. Transforms are not
        serialized.
        """
        if self._mode == "xy":
            arrays = {"x": self.x}

            if self.y is not None:
                arrays["y"] = self.y

        elif self._mode == "dict":
            arrays = self.data

        else:
            arrays = {f"field_{i}": value for i, value in enumerate(self.data)}

        output = {}

        for key, value in arrays.items():
            if not _is_tensor(value):
                raise TypeError(
                    f"Cannot save field {key!r}: only tensor fields are supported"
                )

            output[key] = value.data

        np.savez(path, **output)

    @classmethod
    def load(
        cls,
        path,
        *,
        transform=None,
        target_transform=None,
        transforms=None,
        sample_transform=None,
    ):
        """
        Load a dataset saved by Dataset.save().

        Named datasets remain named datasets.
        Positional datasets remain positional datasets.
        """
        if not path.endswith(".npz"):
            path += ".npz"

        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)

            if keys == ["x"]:
                data = archive["x"]
                y = None

            elif "x" in keys and "y" in keys:
                data = archive["x"]
                y = archive["y"]

            elif all(key.startswith("field_") and key[6:].isdigit() for key in keys):
                keys.sort(key=lambda key: int(key[6:]))
                data = tuple(archive[key].copy() for key in keys)
                y = None

            else:
                data = {key: archive[key].copy() for key in keys}
                y = None

        return cls(
            data,
            y,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            sample_transform=sample_transform,
        )


# Subset


class Subset(BaseDataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")

        n = len(dataset)

        if np.any(self.indices < -n) or np.any(self.indices >= n):
            raise IndexError("Subset index out of range")

        self.indices = np.where(
            self.indices < 0,
            self.indices + n,
            self.indices,
        )

    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError("Subset index out of range")

        return self.dataset[int(self.indices[index])]

    def __len__(self):
        return len(self.indices)


# ConcatDataset


class ConcatDataset(BaseDataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)

        if not self.datasets:
            raise ValueError("At least one dataset is required")

        self._cumulative = np.cumsum([len(dataset) for dataset in self.datasets])

    def __len__(self):
        return int(self._cumulative[-1])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range")

        dataset_idx = int(np.searchsorted(self._cumulative, index, side="right"))

        previous = int(self._cumulative[dataset_idx - 1]) if dataset_idx > 0 else 0

        return self.datasets[dataset_idx][index - previous]


class IterableDataset:
    """
    Base class for streaming datasets.

    Implement __iter__ to yield individual samples.

    Random access and dataset length are optional.
    """

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise TypeError("IterableDataset length is unknown")

    def __add__(self, other):
        return ChainIterableDataset([self, other])


class ChainIterableDataset(IterableDataset):
    """Concatenate iterable datasets without loading them into memory."""

    def __init__(self, datasets):
        self.datasets = list(datasets)

    def __iter__(self):
        for dataset in self.datasets:
            yield from dataset

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
