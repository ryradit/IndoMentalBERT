"""
IndoMentalBERT Model Architecture.
Siamese Dual-Encoder for Question-Answer Alignment built on native Indonesian IndoBERT.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

__all__ = ["FFN", "IndoMentalBERT"]


class FFN(nn.Module):
    """Projection Feed-Forward Network with ReLU, Dropout, and Residual Connection."""
    def __init__(self, hidden_size: int = 768, dropout_rate: float = 0.2):
        super(FFN, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights with glorot/xavier normal
        nn.init.xavier_normal_(self.dense.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dense(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + residual  # Skip connection
        return out


class IndoMentalBERT(nn.Module):
    """
    Multi-Task Siamese Dual-Encoder for Indonesian Mental Health.
    Hard Parameter Sharing Architecture:
    1. Base Encoder: Shared Transformer (indobenchmark/indobert-base-p1)
    2. Head 1 (Retrieval Matching): Query & Answer FFN projection for Cosine Similarity.
    3. Head 2 (Clinical Risk Triage): 5-class severity classification (Low, Mild, Moderate, Severe, Emergency).
    4. Head 3 (Cognitive Distortion Detector): Multi-label distortion classification.
    """
    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        freeze_bert: bool = False,
        dropout_rate: float = 0.2,
        num_risk_classes: int = 5,
        num_distortion_classes: int = 6
    ):
        super(IndoMentalBERT, self).__init__()
        self.model_name = model_name

        # Load model with automatic offline cache fallback
        try:
            self.bert = AutoModel.from_pretrained(model_name)
        except Exception:
            self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        
        # Head 1: Semantic Retrieval Projection Heads (Siamese Q&A)
        self.q_ffn = FFN(hidden_size=hidden_size, dropout_rate=dropout_rate)
        self.a_ffn = FFN(hidden_size=hidden_size, dropout_rate=dropout_rate)

        # Head 2: Multi-Task Clinical Severity / Risk Triaging Head
        self.risk_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_risk_classes)
        )

        # Head 3: Multi-Task Cognitive Distortion Diagnostic Head
        self.distortion_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_distortion_classes)
        )

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_question: bool = True) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use pooler output (CLS token representation)
        cls_rep = outputs.pooler_output
        if is_question:
            emb = self.q_ffn(cls_rep)
        else:
            emb = self.a_ffn(cls_rep)
        # L2 normalize embeddings
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def predict_clinical_triage(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass for multi-task clinical triaging and cognitive distortion classification."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.pooler_output
        risk_logits = self.risk_classifier(cls_rep)
        distortion_logits = self.distortion_classifier(cls_rep)
        return risk_logits, distortion_logits

    def forward(
        self,
        q_input_ids: torch.Tensor,
        q_attention_mask: torch.Tensor,
        a_input_ids: torch.Tensor,
        a_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        q_emb = self.encode_text(q_input_ids, q_attention_mask, is_question=True)
        a_emb = self.encode_text(a_input_ids, a_attention_mask, is_question=False)
        # Compute Cosine Similarity (Dot product of normalized vectors)
        cosine_sim = torch.sum(q_emb * a_emb, dim=1)
        return cosine_sim


if __name__ == "__main__":
    print("[*] Testing IndoMentalBERT architecture initialization...")
    model = IndoMentalBERT()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[✓] IndoMentalBERT initialized successfully! Total trainable parameters: {total_params:,}")
