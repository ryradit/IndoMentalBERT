"""
Interactive Inference & Semantic Response Matching CLI for IndoMentalBERT.
Tests the trained model checkpoint against real-time user queries and candidate responses.
"""

import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.preprocess import clean_text
from src.model import IndoMentalBERT


def load_model_and_pool(
    model_checkpoint: str = "outputs/indomentalbert/indomentalbert_best.pt",
    model_name: str = "indobenchmark/indobert-base-p1",
    data_path: str = "data.csv",
    sample_pool_size: int = 100,
    device: str = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[*] Running on device: {device}")

    # 1. Load Tokenizer & Model
    print(f"[*] Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    print(f"[*] Loading model checkpoint from: {model_checkpoint}")
    model = IndoMentalBERT(model_name=model_name).to(device)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[✓] Checkpoint loaded (Epoch {checkpoint.get('epoch', '?')}, Val Acc: {checkpoint.get('val_accuracy', 0)*100:.2f}%)")
    else:
        model.load_state_dict(checkpoint)
        print("[✓] State dict loaded directly.")

    model.eval()

    # 2. Load candidate responses from data.csv
    print(f"[*] Loading candidate response pool from: {data_path}")
    df = pd.read_csv(data_path).dropna(subset=["Response"])
    unique_responses = df["Response"].drop_duplicates().tolist()
    if sample_pool_size and sample_pool_size < len(unique_responses):
        candidate_pool = unique_responses[:sample_pool_size]
    else:
        candidate_pool = unique_responses

    print(f"[✓] Candidate pool ready with {len(candidate_pool)} unique responses.\n")
    return model, tokenizer, candidate_pool, device


def query_counselor(query: str, model, tokenizer, candidate_pool, device, top_k: int = 3, max_length: int = 128):
    cleaned_q = clean_text(query)
    q_enc = tokenizer(
        cleaned_q,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        q_emb = model.encode_text(q_enc["input_ids"], q_enc["attention_mask"], is_question=True)

        scores = []
        for resp in candidate_pool:
            cleaned_a = clean_text(resp)
            a_enc = tokenizer(
                cleaned_a,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)
            a_emb = model.encode_text(a_enc["input_ids"], a_enc["attention_mask"], is_question=False)
            sim = torch.sum(q_emb * a_emb, dim=1).item()
            scores.append((resp, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IndoMentalBERT Interactive Counselor Demo")
    parser.add_argument("--checkpoint", type=str, default="outputs/indomentalbert/indomentalbert_best.pt")
    parser.add_argument("--data", type=str, default="data.csv")
    parser.add_argument("--pool_size", type=int, default=80)
    args = parser.parse_args()

    model, tokenizer, pool, device = load_model_and_pool(
        model_checkpoint=args.checkpoint,
        data_path=args.data,
        sample_pool_size=args.pool_size
    )

    test_queries = [
        "Saya merasa sangat cemas setiap hari dan tidak bisa fokus bekerja.",
        "Apakah ada gunanya saya pergi ke psikolog?",
        "Saya merasa hampa dan kehilangan motivasi untuk melakukan apa pun."
    ]

    print("=" * 70)
    print("INDOMENTALBERT DEMO - SAMPLE COUNSELING RETRIEVAL")
    print("=" * 70)

    for q in test_queries:
        print(f"\nUser Query: \"{q}\"")
        top_matches = query_counselor(q, model, tokenizer, pool, device, top_k=2)
        for i, (resp, score) in enumerate(top_matches, 1):
            print(f"  [{i}] Cosine Similarity: {score:.4f}")
            # print first 160 characters
            snippet = resp.strip().replace("\n", " ")[:180]
            print(f"      Response: \"{snippet}...\"\n")
