import numpy as np

from ..tensor import Tensor
from ..tensor import functional as F
from ..tensor import factory as init

from .dataset import BaseDataset, IterableDataset
from .sampler import (
    SequentialSampler,
    RandomSampler,
    BatchSampler,
)


# ============================================================
# Collation
# ============================================================


def default_collate(batch):
    """
    Recursively combine individual samples into a batch.

    Supported:
        Nerux Tensor
        NumPy arrays
        Numeric scalars
        Dictionaries
        Tuples
        Lists
        Strings and arbitrary Python objects
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch")

    first = batch[0]

    # Tensor
    if isinstance(first, Tensor):
        if not all(isinstance(x, Tensor) for x in batch):
            raise TypeError("Cannot mix tensors and non-tensors in a batch field")
        shapes = [x.shape for x in batch]
        if any(shape != shapes[0] for shape in shapes):
            raise ValueError(f"Tensor shapes must match for stacking: {shapes}")
        return F.stack(batch, axis=0)

    # NumPy arrays
    if isinstance(first, np.ndarray):
        arrays = [np.asarray(x) for x in batch]
        shapes = [x.shape for x in arrays]
        if any(shape != shapes[0] for shape in shapes):
            raise ValueError(f"Array shapes must match for stacking: {shapes}")

        return init.tensor(np.stack(arrays, axis=0))

    # Dictionaries
    if isinstance(first, dict):
        keys = tuple(first.keys())
        for sample in batch:
            if not isinstance(sample, dict):
                raise TypeError("All samples must have the same structure")

            if tuple(sample.keys()) != keys:
                raise ValueError(
                    "Dictionary keys and their order must match across samples"
                )
        return {key: default_collate([sample[key] for sample in batch]) for key in keys}

    # Tuples
    if isinstance(first, tuple):
        if not all(
            isinstance(sample, tuple) and len(sample) == len(first) for sample in batch
        ):
            raise ValueError("Tuple structures must match across samples")

        return tuple(default_collate(list(values)) for values in zip(*batch))

    # Lists
    if isinstance(first, list):
        if not all(
            isinstance(sample, list) and len(sample) == len(first) for sample in batch
        ):
            # Preserve variable-length lists as sample data.
            return batch
        return [default_collate(list(values)) for values in zip(*batch)]

    # Numeric scalars
    if isinstance(first, (bool, int, float, np.number)):
        if not all(isinstance(value, (bool, int, float, np.number)) for value in batch):
            raise TypeError("Cannot mix numeric and non-numeric scalar values")
        return init.tensor(np.asarray(batch))

    return list(batch)


# ============================================================
# DataLoader
# ============================================================


class DataLoader:
    """
    DataLoader for Nerux map-style and iterable datasets.

    Args:
        dataset:
            Dataset or IterableDataset.

        batch_size:
            Number of samples per batch.

        shuffle:
            Shuffle map-style datasets.

        sampler:
            Custom index sampler. Incompatible with shuffle=True.

        batch_sampler:
            Custom sampler yielding lists of indices.

        drop_last:
            Drop an incomplete final batch.

        collate_fn:
            Convert a list of samples into a batch.

        batch_transform:
            Transform the completed batch.

    Notes:
        - IterableDataset cannot use index samplers.
        - len(loader) is unavailable for unknown-length streams.
        - This implementation does not yet use worker processes.
    """

    def __init__(
        self,
        dataset,
        batch_size=32,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        collate_fn=None,
        batch_transform=None,
    ):
        if batch_sampler is not None:
            if sampler is not None or shuffle or drop_last:
                raise ValueError(
                    "batch_sampler cannot be combined with sampler, "
                    "shuffle=True, or drop_last=True"
                )

            if batch_size != 32:
                raise ValueError("When using batch_sampler, do not specify batch_size")

        elif not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if sampler is not None and shuffle:
            raise ValueError("sampler and shuffle=True cannot be combined")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate
        self.batch_transform = batch_transform

        self.is_iterable = isinstance(dataset, IterableDataset)

        if self.is_iterable:
            if sampler is not None or batch_sampler is not None:
                raise ValueError("IterableDataset cannot use index samplers")

            if shuffle:
                raise ValueError(
                    "IterableDataset requires its own streaming "
                    "shuffling implementation"
                )

            self.sampler = None
            self.batch_sampler = None

        else:
            if batch_sampler is not None:
                self.batch_sampler = batch_sampler
                self.sampler = None

            else:
                if sampler is not None:
                    self.sampler = sampler

                elif shuffle:
                    self.sampler = RandomSampler(dataset)

                else:
                    self.sampler = SequentialSampler(dataset)

                self.batch_sampler = BatchSampler(
                    self.sampler,
                    batch_size,
                    drop_last=drop_last,
                )

    # --------------------------------------------------------
    # Internal batch iterator
    # --------------------------------------------------------

    def _iter_batches(self, samples):
        """
        Consume samples incrementally.

        Unlike collecting all samples first, this keeps streaming
        memory usage bounded by the current batch.
        """
        batch = []

        for sample in samples:
            batch.append(sample)

            if len(batch) == self.batch_size:
                yield self._collate_batch(batch)
                batch = []

        if batch and not self.drop_last:
            yield self._collate_batch(batch)

    def _collate_batch(self, samples):
        batch = self.collate_fn(samples)

        if self.batch_transform is not None:
            batch = self.batch_transform(batch)

        return batch

    # --------------------------------------------------------
    # Iteration
    # --------------------------------------------------------

    def __iter__(self):
        if self.is_iterable:
            yield from self._iter_batches(iter(self.dataset))
            return

        for indices in self.batch_sampler:
            samples = [self.dataset[index] for index in indices]

            if not samples:
                continue

            yield self._collate_batch(samples)

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    def __len__(self):
        if self.is_iterable:
            try:
                n = len(self.dataset)
            except (TypeError, NotImplementedError):
                raise TypeError(
                    "DataLoader length is unknown for this IterableDataset"
                ) from None

            if self.drop_last:
                return n // self.batch_size

            return (n + self.batch_size - 1) // self.batch_size

        return len(self.batch_sampler)
