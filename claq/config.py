"""Small configuration layer for the CLAQ research repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


def detect_device(allow_mps: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PathsConfig:
    repo_root: Path
    data_root: Path
    assets_root: Path
    artifacts_root: Path
    concept_file: Path
    gpt_answers_file: Path

    @property
    def models_root(self) -> Path:
        return self.artifacts_root / "models"

    @property
    def checkpoints_root(self) -> Path:
        return self.models_root

    @property
    def runs_root(self) -> Path:
        return self.artifacts_root / "runs"

    @property
    def figures_root(self) -> Path:
        return self.artifacts_root / "figures"

    @property
    def bootstrap_models_root(self) -> Path:
        return self.models_root / "bootstrap"

    @property
    def bootstrap_checkpoints_root(self) -> Path:
        return self.bootstrap_models_root

    @property
    def reference_runs_root(self) -> Path:
        return self.runs_root / "reference"

    @property
    def sensitive_labels_root(self) -> Path:
        return self.artifacts_root / "sensitive_labels" / "cifar10"

    @property
    def bootstrap_concept_qa_checkpoint(self) -> Path:
        return self.bootstrap_checkpoints_root / "concept_qa_cifar10_reference.pth"

    @property
    def bootstrap_actor_checkpoint(self) -> Path:
        return self.bootstrap_checkpoints_root / "baseline_actor_cifar10_reference.pth"

    @property
    def bootstrap_classifier_checkpoint(self) -> Path:
        return self.bootstrap_checkpoints_root / "baseline_classifier_cifar10_reference.pth"

    def ensure_artifact_dirs(self) -> None:
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.bootstrap_models_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.reference_runs_root.mkdir(parents=True, exist_ok=True)
        self.figures_root.mkdir(parents=True, exist_ok=True)
        self.sensitive_labels_root.mkdir(parents=True, exist_ok=True)
        (self.assets_root / "gpt_answers").mkdir(parents=True, exist_ok=True)


def default_paths(repo_root: str | Path | None = None) -> PathsConfig:
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    assets_root = repo_root / "assets"

    return PathsConfig(
        repo_root=repo_root,
        data_root=repo_root / "data",
        assets_root=assets_root,
        artifacts_root=repo_root / "artifacts",
        concept_file=assets_root / "concepts" / "cifar10.txt",
        gpt_answers_file=assets_root / "gpt_answers" / "cifar10_answers_gpt4.npy",
    )


@dataclass
class Cifar10ClaqConfig:
    dataset_name: str = "cifar10"
    clip_model_name: str = "ViT-B/16"
    max_queries: int = 128
    num_classes: int = 10
    threshold_for_binarization: float = 0.0
    confidence_threshold: float = 0.95
    confidence_max_steps: int = 20
    sensitive_tau: float = 0.7
    sensitive_topk: int = 3
    actor_eps: float = 1.0
    batch_size_cuda: int = 1024
    batch_size_cpu: int = 64
    num_workers_cuda: int = 0
    num_workers_cpu: int = 0
    default_train_epochs: int = 10
    learning_rate: float = 1e-4
    lambda_adv: float = 0.0
    alpha_sens: float = 0.0
    min_history: int = 1
    max_history: int = 2
    non_sensitive_history_only: bool = True
    history_mode: str = "uniform"
    random_seed: int = 0
    allow_mps: bool = False

    @property
    def device(self) -> torch.device:
        return detect_device(allow_mps=self.allow_mps)

    @property
    def batch_size(self) -> int:
        return self.batch_size_cuda if self.device.type == "cuda" else self.batch_size_cpu

    @property
    def num_workers(self) -> int:
        return self.num_workers_cuda if self.device.type == "cuda" else self.num_workers_cpu


@dataclass
class CelebAClaqConfig:
    dataset_name: str = "celeba"
    clip_model_name: str = "ViT-B/16"
    target_attribute: str = "Attractive"
    sensitive_attributes: tuple[str, ...] = (
        "Male",
        "No_Beard",
        "Mustache",
        "Goatee",
        "Sideburns",
        "5_o_Clock_Shadow",
        "Heavy_Makeup",
        "Wearing_Lipstick",
    )
    max_queries: int = 39
    num_classes: int = 2
    threshold_for_binarization: float = 0.0
    confidence_threshold: float = 0.95
    confidence_max_steps: int = 20
    sensitive_tau: float = 0.7
    sensitive_topk: int = 3
    actor_eps: float = 1.0
    batch_size_cuda: int = 1024
    batch_size_cpu: int = 32
    num_workers_cuda: int = 0
    num_workers_cpu: int = 0
    concept_qa_epochs: int = 3
    default_train_epochs: int = 5
    learning_rate: float = 1e-4
    lambda_adv: float = 0.0
    alpha_sens: float = 0.0
    min_history: int = 1
    max_history: int = 2
    non_sensitive_history_only: bool = True
    history_mode: str = "uniform"
    random_seed: int = 0
    allow_mps: bool = False

    @property
    def device(self) -> torch.device:
        return detect_device(allow_mps=self.allow_mps)

    @property
    def batch_size(self) -> int:
        return self.batch_size_cuda if self.device.type == "cuda" else self.batch_size_cpu

    @property
    def num_workers(self) -> int:
        return self.num_workers_cuda if self.device.type == "cuda" else self.num_workers_cpu
