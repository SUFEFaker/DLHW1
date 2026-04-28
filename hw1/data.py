from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass(frozen=True)
class Sample:
    path: str
    label: int


class EuroSATDataModule:
    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 64,
        image_size: tuple[int, int] = (64, 64),
        seed: int = 42,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_definitions: dict[str, list[dict]] | None = None,
        class_names: list[str] | None = None,
        mean: list[float] | np.ndarray | None = None,
        std: list[float] | np.ndarray | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
    ) -> None:
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must sum to 1.0.")

        self.data_root = Path(data_root)
        self.batch_size = int(batch_size)
        self.image_size = tuple(int(v) for v in image_size)
        self.seed = int(seed)
        self.split_ratios = split_ratios
        self.predefined_splits = split_definitions
        self.class_names = class_names or []
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples

        self.splits: dict[str, list[Sample]] = {}
        self.input_shape = (self.image_size[0], self.image_size[1], 3)

    @property
    def input_dim(self) -> int:
        return int(np.prod(self.input_shape))

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def prepare(self) -> None:
        if self.predefined_splits is not None:
            # 评估和分析阶段复用训练时保存的划分，避免重新随机划分导致结果不一致。
            if not self.class_names:
                self.class_names = self._discover_classes()
            self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
            self.splits = {
                split_name: [
                    Sample(path=item["path"], label=int(item["label"])) for item in items
                ]
                for split_name, items in self.predefined_splits.items()
            }
        else:
            self.class_names = self._discover_classes()
            self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
            samples = self._discover_samples()
            self.splits = self._stratified_split(samples)

        self.splits["train"] = self._limit(self.splits["train"], self.max_train_samples)
        self.splits["val"] = self._limit(self.splits["val"], self.max_val_samples)
        self.splits["test"] = self._limit(self.splits["test"], self.max_test_samples)

        if self.mean is None or self.std is None:
            # 只用训练集统计归一化参数，验证集和测试集不能参与统计。
            self.mean, self.std = self._compute_normalization_stats(self.splits["train"])

    def _discover_classes(self) -> list[str]:
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        return sorted(path.name for path in self.data_root.iterdir() if path.is_dir())

    def _discover_samples(self) -> list[Sample]:
        samples: list[Sample] = []
        for class_name in self.class_names:
            class_dir = self.data_root / class_name
            label = self.class_to_idx[class_name]
            for image_path in sorted(class_dir.glob("*")):
                if image_path.is_file():
                    relative = image_path.relative_to(self.data_root).as_posix()
                    samples.append(Sample(path=relative, label=label))
        if not samples:
            raise RuntimeError(f"No image files found under {self.data_root}")
        return samples

    def _stratified_split(self, samples: list[Sample]) -> dict[str, list[Sample]]:
        # 按类别分别划分，保证 train/val/test 中类别比例基本一致。
        buckets: dict[int, list[Sample]] = {index: [] for index in range(len(self.class_names))}
        for sample in samples:
            buckets[sample.label].append(sample)

        train_ratio, val_ratio, _ = self.split_ratios
        rng = np.random.default_rng(self.seed)
        splits = {"train": [], "val": [], "test": []}

        for label in range(len(self.class_names)):
            bucket = list(buckets[label])
            rng.shuffle(bucket)
            count = len(bucket)
            train_count = int(count * train_ratio)
            val_count = int(count * val_ratio)
            test_count = count - train_count - val_count

            if train_count <= 0 or val_count <= 0 or test_count <= 0:
                raise ValueError(
                    "Each class must have enough samples to create non-empty train/val/test splits."
                )

            splits["train"].extend(bucket[:train_count])
            splits["val"].extend(bucket[train_count : train_count + val_count])
            splits["test"].extend(bucket[train_count + val_count :])

        return splits

    @staticmethod
    def _limit(samples: list[Sample], max_samples: int | None) -> list[Sample]:
        if max_samples is None:
            return samples
        return samples[: max(0, int(max_samples))]

    def _load_image(self, relative_path: str) -> np.ndarray:
        if Image is None:
            raise ImportError(
                "Pillow is required to load EuroSAT images. Install it with: "
                "python -m pip install Pillow"
            )

        absolute_path = self.data_root / relative_path
        with Image.open(absolute_path) as image:
            image = image.convert("RGB")
            expected_size = (self.image_size[1], self.image_size[0])
            if image.size != expected_size:
                image = image.resize(expected_size)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return array

    def load_display_image(self, relative_path: str) -> np.ndarray:
        return self._load_image(relative_path)

    def _compute_normalization_stats(self, samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sq_sum = np.zeros(3, dtype=np.float64)
        pixel_count = 0

        for sample in samples:
            image = self._load_image(sample.path)
            pixels = image.reshape(-1, 3)
            channel_sum += pixels.sum(axis=0)
            channel_sq_sum += np.square(pixels, dtype=np.float64).sum(axis=0)
            pixel_count += pixels.shape[0]

        mean = channel_sum / pixel_count
        variance = channel_sq_sum / pixel_count - mean ** 2
        std = np.sqrt(np.maximum(variance, 1e-12))
        return mean.astype(np.float32), std.astype(np.float32)

    def iter_batches(
        self,
        split: str,
        shuffle: bool = False,
        with_paths: bool = False,
        epoch: int = 0,
    ):
        if split not in self.splits:
            raise KeyError(f"Unknown split: {split}")
        samples = self.splits[split]
        indices = np.arange(len(samples))
        if shuffle:
            # 每个 epoch 使用不同但可复现的随机顺序。
            rng = np.random.default_rng(self.seed + epoch)
            rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch_samples = [samples[index] for index in batch_indices]
            images = np.empty((len(batch_samples), *self.input_shape), dtype=np.float32)
            labels = np.empty((len(batch_samples),), dtype=np.int64)
            paths: list[str] = []

            for i, sample in enumerate(batch_samples):
                image = self._load_image(sample.path)
                # 使用训练集均值和标准差标准化，再展平成 MLP 输入向量。
                image = (image - self.mean) / self.std
                images[i] = image
                labels[i] = sample.label
                paths.append(sample.path)

            features = images.reshape(len(batch_samples), -1)
            if with_paths:
                yield features, labels, paths
            else:
                yield features, labels

    def split_sizes(self) -> dict[str, int]:
        return {name: len(items) for name, items in self.splits.items()}

    def serialize_splits(self) -> dict[str, list[dict[str, int | str]]]:
        return {
            split_name: [{"path": sample.path, "label": sample.label} for sample in samples]
            for split_name, samples in self.splits.items()
        }
