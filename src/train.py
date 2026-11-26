"""
Training script for IndoMentalBERT using indobenchmark/indobert-base-p1.
Resolves vocabulary mismatch and trains with proper batching and validation metrics.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure src and project root are in sys.path regardless of execution directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in (CURRENT_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Robust relative/absolute import handling
try:
    from src.preprocess import load_and_prepare_dataset, split_dataset
    from src.model import IndoMentalBERT
except ImportError:
    from preprocess import load_and_prepare_dataset, split_dataset
    from model import IndoMentalBERT


class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_length: int = 128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_enc = self.tokenizer(
            str(row["question"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        a_enc = self.tokenizer(
            str(row["answer"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        label = torch.tensor(row["label"], dtype=torch.float32)

        return {
            "q_input_ids": q_enc["input_ids"].squeeze(0),
            "q_attention_mask": q_enc["attention_mask"].squeeze(0),
            "a_input_ids": a_enc["input_ids"].squeeze(0),
            "a_attention_mask": a_enc["attention_mask"].squeeze(0),
            "label": label
        }


def compute_accuracy(preds, labels, threshold=0.0):
    pred_binary = (preds >= threshold).astype(float)
    label_binary = (labels >= threshold).astype(float)
    correct = (pred_binary == label_binary).sum()
    return float(correct / len(labels))


def resolve_path(path: str) -> str:
    """Resolve file path relative to working dir or project root."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    root_path = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_path):
        return root_path
    return path


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[*] Training on device: {device}")

    # 1. Resolve & Load Data
    resolved_data_path = resolve_path(args.data_path)
    if not os.path.exists(resolved_data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{args.data_path}' or '{resolved_data_path}'. "
            f"Please verify the file exists."
        )

    print(f"[*] Loading data from: {resolved_data_path}")
    df = load_and_prepare_dataset(resolved_data_path, max_samples=args.max_samples, random_seed=args.seed)
    train_df, val_df = split_dataset(df, test_size=args.test_size, random_seed=args.seed)
    print(f"[*] Total dataset size: {len(df)} pairs | Train: {len(train_df)} | Validation: {len(val_df)}")

    # 2. Tokenizer & Model (with automatic offline cache fallback)
    print(f"[*] Initializing native Indonesian model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)

    model = IndoMentalBERT(model_name=args.model_name, dropout_rate=args.dropout).to(device)

    # 3. DataLoaders
    train_ds = QADataset(train_df, tokenizer, max_length=args.max_length)
    val_ds = QADataset(val_df, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    criterion = nn.MSELoss()

    best_val_acc = 0.0
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Starting Training: {args.epochs} Epochs | Batch Size: {args.batch_size} | Max Len: {args.max_length}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_preds, train_labels = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)
            labels = batch["label"].to(device)

            similarity = model(q_ids, q_mask, a_ids, a_mask)
            loss = criterion(similarity, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_preds.extend(similarity.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = compute_accuracy(np.array(train_preds), np.array(train_labels))
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                q_ids = batch["q_input_ids"].to(device)
                q_mask = batch["q_attention_mask"].to(device)
                a_ids = batch["a_input_ids"].to(device)
                a_mask = batch["a_attention_mask"].to(device)
                labels = batch["label"].to(device)

                similarity = model(q_ids, q_mask, a_ids, a_mask)
                loss = criterion(similarity, labels)
                total_val_loss += loss.item()

                val_preds.extend(similarity.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = compute_accuracy(np.array(val_preds), np.array(val_labels))
        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch:02d} Summary: "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(output_dir, "indomentalbert_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": avg_val_loss,
            }, save_path)
            print(f"[*] New best model saved to: {save_path} (Val Acc: {val_acc * 100:.2f}%)")

    print(f"\n[*] Training Completed! Best Validation Accuracy: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IndoMentalBERT on Indonesian Mental Health Corpus")
    parser.add_argument("--data_path", type=str, default="data.csv", help="Path to data.csv or intents json")
    parser.add_argument("--model_name", type=str, default="indobenchmark/indobert-base-p1")
    parser.add_argument("--output_dir", type=str, default="outputs/indomentalbert")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional subset size for fast local testing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
