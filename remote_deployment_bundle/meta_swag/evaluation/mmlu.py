from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


SUBJECT_GROUPS = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics",
        "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
        "human_aging", "management", "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine", "virology",
    ],
}

CHOICES = ["A", "B", "C", "D"]


def format_mmlu_prompt(question: str, choices: list[str], few_shot_examples: list[dict] | None = None) -> str:
    parts = []
    if few_shot_examples:
        for ex in few_shot_examples:
            parts.append(_format_single(ex["question"], ex["choices"], ex["answer"]))
            parts.append("")
    parts.append(_format_single(question, choices))
    return "\n".join(parts)


def _format_single(question: str, choices: list[str], answer: str | None = None) -> str:
    lines = [question]
    for letter, choice in zip(CHOICES, choices):
        lines.append(f"{letter}. {choice}")
    if answer is not None:
        lines.append(f"Answer: {answer}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


@dataclass
class MMLUResult:
    subject: str
    subject_group: str
    accuracy: float
    n_questions: int
    correct: int


def load_mmlu_subject(subject: str, split: str = "test", num_few_shot: int = 5) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", subject)
    few_shot_data = []
    if "validation" in ds and num_few_shot > 0:
        for row in ds["validation"]:
            if len(few_shot_data) >= num_few_shot:
                break
            few_shot_data.append({
                "question": row["question"],
                "choices": row["choices"],
                "answer": CHOICES[row["answer"]],
            })

    test_data = []
    for row in ds[split]:
        test_data.append({
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
        })

    return few_shot_data, test_data


@torch.no_grad()
def evaluate_mmlu_subject(
    model: torch.nn.Module,
    tokenizer,
    subject: str,
    few_shot_examples: list[dict],
    test_examples: list[dict],
    device: torch.device,
    batch_size: int = 8,
) -> MMLUResult:
    choice_token_ids = []
    for c in CHOICES:
        token_ids = tokenizer.encode(c, add_special_tokens=False)
        choice_token_ids.append(token_ids[-1])

    correct = 0
    total = 0

    for i in range(0, len(test_examples), batch_size):
        batch = test_examples[i:i + batch_size]
        prompts = [
            format_mmlu_prompt(ex["question"], ex["choices"], few_shot_examples)
            for ex in batch
        ]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(device)
        outputs = model(**encoded)
        last_logits = outputs.logits[:, -1, :]

        for j, ex in enumerate(batch):
            choice_logits = torch.tensor([last_logits[j, tid].item() for tid in choice_token_ids])
            predicted = int(choice_logits.argmax().item())
            if predicted == ex["answer"]:
                correct += 1
            total += 1

    subject_group = "Other"
    for group, subjects in SUBJECT_GROUPS.items():
        if subject in subjects:
            subject_group = group
            break

    return MMLUResult(
        subject=subject,
        subject_group=subject_group,
        accuracy=correct / max(total, 1),
        n_questions=total,
        correct=correct,
    )


@torch.no_grad()
def evaluate_mmlu_subject_bma(
    model: torch.nn.Module,
    tokenizer,
    subject: str,
    few_shot_examples: list[dict],
    test_examples: list[dict],
    device: torch.device,
    posterior_predictive,
    batch_size: int = 8,
) -> MMLUResult:
    choice_token_ids = []
    for c in CHOICES:
        token_ids = tokenizer.encode(c, add_special_tokens=False)
        choice_token_ids.append(token_ids[-1])

    correct = 0
    total = 0

    for i in range(0, len(test_examples), batch_size):
        batch = test_examples[i:i + batch_size]
        prompts = [
            format_mmlu_prompt(ex["question"], ex["choices"], few_shot_examples)
            for ex in batch
        ]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(device)

        avg_probs = posterior_predictive.average_softmax(
            model, encoded["input_ids"], encoded["attention_mask"],
        )
        last_probs = avg_probs[:, -1, :]

        for j, ex in enumerate(batch):
            choice_probs = torch.tensor([last_probs[j, tid].item() for tid in choice_token_ids])
            predicted = int(choice_probs.argmax().item())
            if predicted == ex["answer"]:
                correct += 1
            total += 1

    subject_group = "Other"
    for group, subjects in SUBJECT_GROUPS.items():
        if subject in subjects:
            subject_group = group
            break

    return MMLUResult(
        subject=subject,
        subject_group=subject_group,
        accuracy=correct / max(total, 1),
        n_questions=total,
        correct=correct,
    )


def all_subjects() -> list[str]:
    subjects = []
    for group_subjects in SUBJECT_GROUPS.values():
        subjects.extend(group_subjects)
    return sorted(subjects)
