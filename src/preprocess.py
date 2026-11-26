"""
Preprocessing module for IndoMentalBERT dataset.
Handles text cleaning, decontractions, negative sample generation, and tokenization.
Supports both intents JSON and large CSV datasets (Context, Response).
"""

import os
import re
import json
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Clean text by lowercasing, stripping noise characters, and normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Strip punctuation and special noise symbols
    text = re.sub(r'[$)\?"’.°!;\'€%:,(/\[\]\\]', "", text)
    # Replace special unicode spaces and hyphens
    text = re.sub(r"\u200b", " ", text)
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"-", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_from_csv(csv_path: str, max_samples: int = None, random_seed: int = 42):
    """
    Load dataset from CSV with 'Context' and 'Response' columns.
    Generates negative samples by pairing Context with a different Response.
    """
    random.seed(random_seed)
    df = pd.read_csv(csv_path).dropna(subset=["Context", "Response"])
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)

    questions = df["Context"].apply(clean_text).tolist()
    answers = df["Response"].apply(clean_text).tolist()

    positive_pairs = [
        {"question": q, "answer": a, "tag": "mental_health_qa", "label": 1.0}
        for q, a in zip(questions, answers)
    ]

    # Generate random negative answers for each question
    negative_pairs = []
    n = len(answers)
    for i, q in enumerate(questions):
        # Pick random index different from i
        neg_idx = random.choice([idx for idx in range(n) if idx != i])
        negative_pairs.append({
            "question": q,
            "answer": answers[neg_idx],
            "tag": "negative_sample",
            "label": -1.0
        })

    df_combined = pd.concat([pd.DataFrame(positive_pairs), pd.DataFrame(negative_pairs)], axis=0)
    return df_combined.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def load_from_json(json_path: str, random_seed: int = 42):
    """Load intents JSON and create positive + negative contrastive pairs."""
    random.seed(random_seed)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["intents"]

    positive_pairs = []
    tags_dict = {}

    for item in data:
        tag = item["tag"]
        patterns = item.get("patterns", [])
        responses = item.get("responses", [])
        if tag not in tags_dict:
            tags_dict[tag] = responses

        for q in patterns:
            for a in responses:
                positive_pairs.append({
                    "question": clean_text(q),
                    "answer": clean_text(a),
                    "tag": tag,
                    "label": 1.0
                })

    df_pos = pd.DataFrame(positive_pairs)
    all_tags = list(tags_dict.keys())

    # Generate negative samples: pair each question with a response from a DIFFERENT tag
    negative_pairs = []
    for _, row in df_pos.iterrows():
        current_tag = row["tag"]
        other_tags = [t for t in all_tags if t != current_tag]
        if not other_tags:
            continue
        sampled_tag = random.choice(other_tags)
        sampled_response = random.choice(tags_dict[sampled_tag])

        negative_pairs.append({
            "question": row["question"],
            "answer": clean_text(sampled_response),
            "tag": current_tag,
            "label": -1.0
        })

    df_neg = pd.DataFrame(negative_pairs)
    df_combined = pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)
    return df_combined


def load_and_prepare_dataset(data_path: str, max_samples: int = None, random_seed: int = 42):
    """Unified loader supporting both JSON and CSV datasets."""
    if data_path.endswith(".csv"):
        return load_from_csv(data_path, max_samples=max_samples, random_seed=random_seed)
    elif data_path.endswith(".json"):
        return load_from_json(data_path, random_seed=random_seed)
    else:
        raise ValueError(f"Unsupported file format: {data_path}. Must be .json or .csv")


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_seed: int = 42):
    """Perform stratified train/validation split."""
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
