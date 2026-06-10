"""CelebA helpers for the CLAQ research workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


_ATTRIBUTE_NAME_OVERRIDES = {
    "5_o_Clock_Shadow": "five o clock shadow",
}


def humanize_celeba_attribute(name: str) -> str:
    if name in _ATTRIBUTE_NAME_OVERRIDES:
        return _ATTRIBUTE_NAME_OVERRIDES[name]
    return name.replace("_", " ").lower()


@dataclass(frozen=True)
class CelebAAttributeSpec:
    target_attribute: str
    target_index: int
    query_attribute_names: list[str]
    query_attribute_indices: list[int]
    concept_names: list[str]
    sensitive_attribute_names: list[str]
    sensitive_indices: torch.Tensor
    class_names: list[str]

    @property
    def num_queries(self) -> int:
        return len(self.query_attribute_names)


def build_celeba_attribute_spec(
    attr_names: list[str],
    target_attribute: str,
    sensitive_attributes: list[str] | tuple[str, ...],
) -> CelebAAttributeSpec:
    if target_attribute not in attr_names:
        raise ValueError(f"Unknown target attribute: {target_attribute}")

    target_index = attr_names.index(target_attribute)
    query_attribute_names = [name for name in attr_names if name != target_attribute]
    query_attribute_indices = [attr_names.index(name) for name in query_attribute_names]
    concept_names = [humanize_celeba_attribute(name) for name in query_attribute_names]

    matched_sensitive_names = [name for name in sensitive_attributes if name in query_attribute_names]
    missing_sensitive_names = [name for name in sensitive_attributes if name not in query_attribute_names]
    if missing_sensitive_names:
        raise ValueError(
            "Sensitive attributes missing from the query vocabulary: "
            + ", ".join(missing_sensitive_names)
        )

    sensitive_indices = torch.tensor(
        [query_attribute_names.index(name) for name in matched_sensitive_names],
        dtype=torch.long,
    )
    return CelebAAttributeSpec(
        target_attribute=target_attribute,
        target_index=target_index,
        query_attribute_names=query_attribute_names,
        query_attribute_indices=query_attribute_indices,
        concept_names=concept_names,
        sensitive_attribute_names=matched_sensitive_names,
        sensitive_indices=sensitive_indices,
        class_names=[f"not {humanize_celeba_attribute(target_attribute)}", humanize_celeba_attribute(target_attribute)],
    )


def _clean_celeba_attr_names(attr_names: list[str], num_columns: int) -> list[str]:
    # Some torchvision versions return a trailing empty name from whitespace-split parsing,
    # so the name list is one longer than the attr tensor. Trust the tensor width.
    cleaned = [name for name in attr_names if name and name.strip()]
    if len(cleaned) != num_columns:
        raise ValueError(
            f"CelebA attribute name/column mismatch after cleaning: "
            f"{len(cleaned)} names vs {num_columns} columns"
        )
    return cleaned


def load_celeba_attribute_spec(
    root: str | Path,
    target_attribute: str,
    sensitive_attributes: list[str] | tuple[str, ...],
    download: bool = False,
) -> CelebAAttributeSpec:
    dataset = torchvision.datasets.CelebA(
        root=str(root),
        split="train",
        target_type="attr",
        transform=None,
        download=download,
    )
    attr_names = _clean_celeba_attr_names(list(dataset.attr_names), dataset.attr.size(1))
    return build_celeba_attribute_spec(
        attr_names=attr_names,
        target_attribute=target_attribute,
        sensitive_attributes=sensitive_attributes,
    )


class CelebAClaqDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        spec: CelebAAttributeSpec,
        transform=None,
        return_query_targets: bool = False,
        download: bool = False,
    ) -> None:
        self.base_dataset = torchvision.datasets.CelebA(
            root=str(root),
            split=split,
            target_type="attr",
            transform=transform,
            download=download,
        )
        self.spec = spec
        self.return_query_targets = return_query_targets

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, attr = self.base_dataset[index]
        target_label = attr[self.spec.target_index].long()
        if not self.return_query_targets:
            return image, target_label
        query_targets = attr[self.spec.query_attribute_indices].float()
        return image, target_label, query_targets

    @property
    def query_targets(self) -> torch.Tensor:
        return self.base_dataset.attr[:, self.spec.query_attribute_indices].float()


def get_celeba_datasets(
    transform,
    root: str | Path,
    spec: CelebAAttributeSpec,
    return_query_targets: bool = False,
    download: bool = False,
) -> tuple[CelebAClaqDataset, CelebAClaqDataset, CelebAClaqDataset]:
    train_ds = CelebAClaqDataset(
        root=root,
        split="train",
        spec=spec,
        transform=transform,
        return_query_targets=return_query_targets,
        download=download,
    )
    valid_ds = CelebAClaqDataset(
        root=root,
        split="valid",
        spec=spec,
        transform=transform,
        return_query_targets=return_query_targets,
        download=False,
    )
    test_ds = CelebAClaqDataset(
        root=root,
        split="test",
        spec=spec,
        transform=transform,
        return_query_targets=return_query_targets,
        download=False,
    )
    return train_ds, valid_ds, test_ds


def get_celeba_concept_qa_datasets(
    transform,
    root: str | Path,
    spec: CelebAAttributeSpec,
    download: bool = False,
) -> tuple[CelebAClaqDataset, CelebAClaqDataset]:
    train_ds = CelebAClaqDataset(
        root=root,
        split="train",
        spec=spec,
        transform=transform,
        return_query_targets=True,
        download=download,
    )
    valid_ds = CelebAClaqDataset(
        root=root,
        split="valid",
        spec=spec,
        transform=transform,
        return_query_targets=True,
        download=False,
    )
    return train_ds, valid_ds


def get_celeba_loaders(
    transform,
    root: str | Path,
    spec: CelebAAttributeSpec,
    batch_size: int,
    num_workers: int,
    shuffle_train: bool = True,
    return_query_targets: bool = False,
    download: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds, valid_ds, test_ds = get_celeba_datasets(
        transform=transform,
        root=root,
        spec=spec,
        return_query_targets=return_query_targets,
        download=download,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


def get_celeba_concept_qa_loaders(
    transform,
    root: str | Path,
    spec: CelebAAttributeSpec,
    batch_size: int,
    num_workers: int,
    shuffle_train: bool = True,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_ds, valid_ds = get_celeba_concept_qa_datasets(
        transform=transform,
        root=root,
        spec=spec,
        download=download,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader


def get_raw_celeba_dataset(
    root: str | Path,
    spec: CelebAAttributeSpec,
    split: str = "test",
    download: bool = False,
) -> CelebAClaqDataset:
    return CelebAClaqDataset(
        root=root,
        split=split,
        spec=spec,
        transform=None,
        return_query_targets=False,
        download=download,
    )
