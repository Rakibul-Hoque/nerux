import numpy as np
from ..tensor import Tensor


class SequentialSampler:
    def __init__(self, dataset):
        self.n = len(dataset)

    def __iter__(self):
        return iter(np.arange(self.n))

    def __len__(self):
        return self.n


class RandomSampler:
    def __init__(self, dataset, replacement=False, num_samples=None):
        self.n = len(dataset)
        self.replacement = replacement
        self.num_samples = num_samples or self.n

    def __iter__(self):
        if self.replacement:
            return iter(np.random.randint(0, self.n, size=self.num_samples))
        return iter(np.random.permutation(self.n))

    def __len__(self):
        return self.num_samples


class WeightedRandomSampler:
    """
    Sample indices with given per-sample weights (useful for class imbalance).

    Example:
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(labels))
    """

    def __init__(self, weights, num_samples, replacement=True):
        self.weights = np.array(weights, dtype=float)
        self.weights /= self.weights.sum()
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(
            np.random.choice(
                len(self.weights),
                size=self.num_samples,
                replace=self.replacement,
                p=self.weights,
            )
        )

    def __len__(self):
        return self.num_samples


# ── Default collate ──────────────────────────────────────────────────────────


def default_collate(batch):

    if isinstance(batch[0], tuple):
        xs, ys = zip(*batch)
        x_stack = Tensor(np.stack([x.data for x in xs], axis=0))
        if ys[0] is None:
            return x_stack, None
        y_stack = Tensor(np.stack([y.data for y in ys], axis=0))
        return x_stack, y_stack
    else:
        return Tensor(np.stack([x.data for x in batch], axis=0))


# ── DataLoader ───────────────────────────────────────────────────────────────


class DataLoader:
    """
    Args:
        dataset:     any Dataset subclass
        batch_size:  samples per batch
        shuffle:     random order each epoch (ignored if sampler given)
        sampler:     custom index sampler; overrides shuffle
        drop_last:   drop the last incomplete batch
        collate_fn:  fn(list of samples) -> batch; defaults to default_collate
    """

    def __init__(
        self,
        dataset,
        batch_size=32,
        shuffle=False,
        sampler=None,
        drop_last=False,
        collate_fn=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

    def __iter__(self):
        indices = list(iter(self.sampler))
        # Chunk into batches
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        for batch_indices in batches:
            batch = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(batch)

    def __len__(self):
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
