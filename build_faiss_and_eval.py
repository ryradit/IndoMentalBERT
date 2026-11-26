#!/usr/bin/env python3
"""
Build FAISS Vector Index and Evaluate Retrieval Metrics for IndoMentalBERT.

This script:
1. Encodes all counseling responses from data.csv into 768-dim normalized embeddings.
2. Builds an optimized Inner Product (Cosine Similarity) FAISS vector index.
3. Performs comprehensive validation on held-out / contrastive pairs:
   - Confusion Matrix (True Positives, False Positives, True Negatives, False Negatives)
   - Accuracy, Precision, Recall, F1-Score
   - Multi-scale BLEU Scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
4. Exports publication-quality visualization graphs into reports/evaluation_metrics.png.
5. Saves the index to outputs/indomentalbert/counseling_index.faiss and response catalog.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import faiss

# Set writable matplotlib cache dir
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import IndoMentalBERT

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs", "indomentalbert", "indomentalbert_best.pt")
DATA_PATH = os.path.join(PROJECT_ROOT, "data.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "indomentalbert")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(device):
    local_model_dir = os.path.join(PROJECT_ROOT, "outputs", "indobert_base_local")
    model_name = local_model_dir if os.path.exists(local_model_dir) else "indobenchmark/indobert-base-p1"
    
    print(f"[*] Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    print(f"[*] Loading model from: {CHECKPOINT_PATH}")
    model = IndoMentalBERT(model_name=model_name)
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[✓] Checkpoint loaded (Val Acc: {ckpt.get('val_accuracy', 0)*100:.2f}%)")
        else:
            model.load_state_dict(ckpt)
            print("[✓] Raw state dict loaded.")
    else:
        print("[!] Warning: Checkpoint not found, using base weights.")
        
    model.to(device)
    model.eval()
    return model, tokenizer


def encode_texts(texts, model, tokenizer, device, is_question=False, batch_size=64):
    all_embeddings = []
    total = len(texts)
    
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            # Using IndoMentalBERT's dedicated dual encoder head (q_ffn or a_ffn with L2 norm)
            emb = model.encode_text(inputs["input_ids"], inputs["attention_mask"], is_question=is_question)
            all_embeddings.append(emb.cpu().numpy())
            
            if (i // batch_size) % 15 == 0 or i + batch_size >= total:
                print(f"    Encoded {min(i + batch_size, total)}/{total} items...")
                
    return np.vstack(all_embeddings).astype("float32")


def build_and_save_faiss(response_texts, response_embeddings):
    dim = response_embeddings.shape[1]
    # Using IndexFlatIP: inner product of L2-normalized vectors is exactly cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(response_embeddings)
    
    index_path = os.path.join(OUTPUT_DIR, "counseling_index.faiss")
    catalog_path = os.path.join(OUTPUT_DIR, "response_catalog.json")
    
    faiss.write_index(index, index_path)
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(response_texts, f, ensure_ascii=False, indent=2)
        
    print(f"\n[✓] FAISS index saved to: {index_path} (Total vectors: {index.ntotal}, Dim: {dim})")
    print(f"[✓] Response catalog saved to: {catalog_path}")
    return index


def evaluate_metrics(df, model, tokenizer, device, sample_size=1000):
    print("\n--- Running Quantitative Evaluation (Classification + Retrieval + BLEU) ---")
    eval_df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    
    contexts = eval_df["Context"].tolist()
    true_responses = eval_df["Response"].tolist()
    
    # Negative pairs (Context paired with mismatched Response)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(eval_df))
    for i in range(len(eval_df)):
        if shuffled_indices[i] == i:
            swap_idx = (i + 1) % len(eval_df)
            shuffled_indices[i], shuffled_indices[swap_idx] = shuffled_indices[swap_idx], shuffled_indices[i]
            
    false_responses = [true_responses[idx] for idx in shuffled_indices]
    
    print("[*] Encoding evaluation contexts (query head)...")
    c_emb = encode_texts(contexts, model, tokenizer, device, is_question=True, batch_size=64)
    print("[*] Encoding positive responses (answer head)...")
    r_pos_emb = encode_texts(true_responses, model, tokenizer, device, is_question=False, batch_size=64)
    print("[*] Encoding negative responses (answer head)...")
    r_neg_emb = encode_texts(false_responses, model, tokenizer, device, is_question=False, batch_size=64)
    
    # Cosine similarities
    pos_scores = np.sum(c_emb * r_pos_emb, axis=1)
    neg_scores = np.sum(c_emb * r_neg_emb, axis=1)
    
    # Find decision threshold
    threshold = 0.50
    
    # Ground Truth: 1 for relevant, 0 for irrelevant
    y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    y_scores = np.concatenate([pos_scores, neg_scores])
    y_pred = (y_scores >= threshold).astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    
    print("\n" + "=" * 50)
    print("      INSPECTING CLASSIFICATION & RETRIEVAL METRICS")
    print("=" * 50)
    print(f"Total Evaluated Pairs : {len(y_true)}")
    print(f"True Positives  (TP)  : {tp}")
    print(f"False Positives (FP)  : {fp}")
    print(f"True Negatives  (TN)  : {tn}")
    print(f"False Negatives (FN)  : {fn}")
    print(f"Accuracy              : {acc * 100:.2f}%")
    print(f"Precision             : {prec * 100:.2f}%")
    print(f"Recall                : {rec * 100:.2f}%")
    print(f"F1-Score              : {f1 * 100:.2f}%")
    print("=" * 50)
    
    # BLEU Scores Evaluation via FAISS Nearest Neighbor Retrieval
    print("\n[*] Evaluating multi-scale BLEU scores with FAISS Index retrieval...")
    dim = r_pos_emb.shape[1]
    test_index = faiss.IndexFlatIP(dim)
    test_index.add(r_pos_emb)
    
    D, I = test_index.search(c_emb, k=1)
    
    smoothie = SmoothingFunction().method1
    bleu_1_list, bleu_2_list, bleu_3_list, bleu_4_list = [], [], [], []
    
    for i in range(len(eval_df)):
        retrieved_idx = I[i][0]
        ref = true_responses[i].lower().split()
        cand = true_responses[retrieved_idx].lower().split()
        
        b1 = sentence_bleu([ref], cand, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        b2 = sentence_bleu([ref], cand, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        b3 = sentence_bleu([ref], cand, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        b4 = sentence_bleu([ref], cand, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        bleu_1_list.append(b1)
        bleu_2_list.append(b2)
        bleu_3_list.append(b3)
        bleu_4_list.append(b4)
        
    avg_b1 = float(np.mean(bleu_1_list) * 100)
    avg_b2 = float(np.mean(bleu_2_list) * 100)
    avg_b3 = float(np.mean(bleu_3_list) * 100)
    avg_b4 = float(np.mean(bleu_4_list) * 100)
    
    print("\n" + "=" * 50)
    print("             RETRIEVED RESPONSE BLEU SCORES")
    print("=" * 50)
    print(f"BLEU-1 (Unigram match) : {avg_b1:.2f}%")
    print(f"BLEU-2 (Bigram match)  : {avg_b2:.2f}%")
    print(f"BLEU-3 (Trigram match) : {avg_b3:.2f}%")
    print(f"BLEU-4 (4-gram match)  : {avg_b4:.2f}%")
    print("=" * 50)
    
    metrics = {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1_score": float(f1),
        "bleu_1": avg_b1, "bleu_2": avg_b2, "bleu_3": avg_b3, "bleu_4": avg_b4,
        "pos_scores": pos_scores.tolist(), "neg_scores": neg_scores.tolist()
    }
    
    with open(os.path.join(REPORTS_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics, cm


def plot_graphs(metrics, cm):
    print("\n[*] Generating visualization charts...")
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor('#f8f9fa')
    plt.suptitle("IndoMentalBERT: FAISS Retrieval & Clinical Evaluation Report", fontsize=16, fontweight="bold", y=0.98)
    
    # 1. Confusion Matrix Heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    labels = ["Negative (Irrelevant)", "Positive (Relevant)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax1,
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax1.set_title(f"Confusion Matrix\n(TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']} | TN: {metrics['tn']})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted", fontsize=10)
    ax1.set_ylabel("Ground Truth", fontsize=10)
    
    # 2. Performance Metrics Bar Chart
    ax2 = fig.add_subplot(2, 2, 2)
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_values = [metrics["accuracy"] * 100, metrics["precision"] * 100, metrics["recall"] * 100, metrics["f1_score"] * 100]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22"]
    
    bars = ax2.bar(metric_names, metric_values, color=colors, width=0.55, edgecolor="#2c3e50", linewidth=1.2)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Percentage (%)", fontsize=10)
    ax2.set_title("Classification & Retrieval Performance", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height:.2f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    # 3. BLEU Scores Progression
    ax3 = fig.add_subplot(2, 2, 3)
    bleu_labels = ["BLEU-1\n(Unigram)", "BLEU-2\n(Bigram)", "BLEU-3\n(Trigram)", "BLEU-4\n(4-gram)"]
    bleu_vals = [metrics["bleu_1"], metrics["bleu_2"], metrics["bleu_3"], metrics["bleu_4"]]
    bleu_colors = ["#1abc9c", "#16a085", "#27ae60", "#2ecc71"]
    
    bars_bleu = ax3.bar(bleu_labels, bleu_vals, color=bleu_colors, width=0.55, edgecolor="#2c3e50", linewidth=1.2)
    ax3.set_ylim(0, max(bleu_vals) * 1.25)
    ax3.set_ylabel("BLEU Score (%)", fontsize=10)
    ax3.set_title("N-gram Semantic Precision (BLEU Scores)", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars_bleu:
        height = bar.get_height()
        ax3.annotate(f"{height:.2f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    # 4. Cosine Similarity Distribution
    ax4 = fig.add_subplot(2, 2, 4)
    sns.kdeplot(metrics["pos_scores"], label="Relevant Pairs (Positive)", color="#27ae60", fill=True, alpha=0.4, linewidth=2, ax=ax4)
    sns.kdeplot(metrics["neg_scores"], label="Mismatched Pairs (Negative)", color="#e74c3c", fill=True, alpha=0.4, linewidth=2, ax=ax4)
    ax4.axvline(0.50, color="#2c3e50", linestyle="--", linewidth=1.5, label="Decision Threshold (0.50)")
    ax4.set_title("Semantic Separation Distribution (Cosine Similarity)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Cosine Similarity", fontsize=10)
    ax4.set_ylabel("Density", fontsize=10)
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    chart_path = os.path.join(REPORTS_DIR, "evaluation_metrics.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[✓] Visualization graph saved to: {chart_path}")


def main():
    device = get_device()
    print(f"[*] Compute device: {device}")
    
    print(f"[*] Reading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[*] Total dataset records: {len(df)}")
    
    df["Context"] = df["Context"].fillna("").astype(str).str.strip()
    df["Response"] = df["Response"].fillna("").astype(str).str.strip()
    df = df[(df["Context"] != "") & (df["Response"] != "")].reset_index(drop=True)
    
    unique_responses = sorted(list(set(df["Response"].tolist())))
    print(f"[*] Total unique counseling responses to index: {len(unique_responses)}")
    
    model, tokenizer = load_model_and_tokenizer(device)
    
    print("\n--- Phase 1: Encoding Unique Responses into FAISS Vector Space ---")
    response_embeddings = encode_texts(unique_responses, model, tokenizer, device, is_question=False, batch_size=64)
    build_and_save_faiss(unique_responses, response_embeddings)
    
    print("\n--- Phase 2: Evaluation & Metrics Calculation ---")
    metrics, cm = evaluate_metrics(df, model, tokenizer, device, sample_size=1000)
    plot_graphs(metrics, cm)
    
    print("\n[✓] All pipeline stages finished successfully!")


if __name__ == "__main__":
    main()
