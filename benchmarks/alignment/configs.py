from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaseModelConfig:
    name: str
    tag: str
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj",
    )
    max_length: int = 512


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_prefs"
    test_split: str = "test_prefs"
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"


@dataclass(frozen=True)
class RewardPairConfig:
    gold_model: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    proxy_model: str = "internlm/internlm2-1_8b-reward"
    gold_tag: str = "skywork_gold"
    proxy_tag: str = "internlm_proxy"


@dataclass(frozen=True)
class DPOTrainingConfig:
    lr: float = 5e-6
    beta: float = 0.1
    n_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    warmup_steps: int = 150
    keep_last: int = 50
    tail_fraction: float = 0.5
    loss_type: str = "dpo"
    label_smoothing: float = 0.0


@dataclass(frozen=True)
class EvalConfig:
    best_of_n_values: tuple[int, ...] = (1, 4, 16, 64, 256)
    temperature: float = 1.0
    max_new_tokens: int = 256
    num_eval_prompts: int = 1000
    posterior_samples: int = 16
    seed: int = 42


@dataclass(frozen=True)
class AlignmentExperimentConfig:
    base_model: BaseModelConfig
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reward_pair: RewardPairConfig = field(default_factory=RewardPairConfig)
    training: DPOTrainingConfig = field(default_factory=DPOTrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    schemes: tuple[str, ...] = (
        "map", "last_iterate", "swa", "ema",
        "softmax", "ess", "threshold", "laplace",
    )
    seed_count: int = 3
    base_seed: int = 42


LLAMA_8B = BaseModelConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    tag="llama-3.1-8b",
)

GEMMA_9B = BaseModelConfig(
    name="google/gemma-2-9b-it",
    tag="gemma-2-9b",
)

DEFAULT_EXPERIMENTS = [
    AlignmentExperimentConfig(base_model=LLAMA_8B),
    AlignmentExperimentConfig(base_model=GEMMA_9B),
]
