import numpy as np


# BaseSampler


class BaseSampler:
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


# SequentialSampler


class SequentialSampler(BaseSampler):
    def __init__(self, dataset):
        self.n = len(dataset)

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


# RandomSampler


class RandomSampler(BaseSampler):
    def __init__(
        self,
        dataset,
        replacement=False,
        num_samples=None,
        seed=None,
    ):
        self.n = len(dataset)
        self.replacement = replacement

        if num_samples is None:
            num_samples = self.n

        if not isinstance(num_samples, (int, np.integer)):
            raise TypeError("num_samples must be an integer")

        if num_samples < 0:
            raise ValueError("num_samples cannot be negative")

        if not replacement and num_samples > self.n:
            raise ValueError(
                "Cannot sample more elements than the dataset without replacement"
            )

        self.num_samples = int(num_samples)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        if self.replacement:
            indices = self._rng.integers(
                0,
                self.n,
                size=self.num_samples,
            )
        else:
            indices = self._rng.permutation(self.n)[: self.num_samples]

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


# WeightedRandomSampler


class WeightedRandomSampler(BaseSampler):
    def __init__(
        self,
        weights,
        num_samples,
        replacement=True,
        seed=None,
    ):
        self.weights = np.asarray(weights, dtype=np.float64)

        if self.weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")

        if len(self.weights) == 0:
            raise ValueError("weights cannot be empty")

        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be finite")

        if np.any(self.weights < 0):
            raise ValueError("weights cannot be negative")

        total = self.weights.sum()

        if total <= 0:
            raise ValueError("At least one weight must be positive")

        if not isinstance(num_samples, (int, np.integer)):
            raise TypeError("num_samples must be an integer")

        if num_samples < 0:
            raise ValueError("num_samples cannot be negative")

        if not replacement and num_samples > len(self.weights):
            raise ValueError(
                "Cannot sample more elements than available without replacement"
            )

        self.weights = self.weights / total
        self.num_samples = int(num_samples)
        self.replacement = replacement
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        indices = self._rng.choice(
            len(self.weights),
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights,
        )

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


# BatchSampler


class BatchSampler(BaseSampler):
    """Group indices from another sampler into batches."""

    def __init__(self, sampler, batch_size, drop_last=False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = drop_last

    def __iter__(self):
        batch = []

        for index in self.sampler:
            batch.append(index)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.sampler)

        if self.drop_last:
            return n // self.batch_size

        return (n + self.batch_size - 1) // self.batch_size
