"""
Knowledge Graph Guided Semantic Retrieval Module for IndoMentalBERT.

This module marries:
1. Pure Sub-millisecond Vector Similarity (FAISS + IndoMentalBERT Dual-Encoder)
2. Symbolic Clinical Reasoning (Indonesian Psychiatric Knowledge Graph)

Result: An explainable, clinically aligned retrieval engine that dynamically prioritizes
somatic and psychological intervention responses and outputs diagnostic metadata.
"""

import os
import json
import torch
import faiss
import numpy as np
from typing import Dict, Any, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer

from src.model import IndoMentalBERT
from src.knowledge_graph import ClinicalKnowledgeGraph

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GuidedClinicalEngine:
    def __init__(
        self,
        checkpoint_path: str = "outputs/indomentalbert/indomentalbert_best.pt",
        faiss_index_path: str = "outputs/indomentalbert/counseling_index.faiss",
        catalog_path: str = "outputs/indomentalbert/response_catalog.json",
        device: str = None
    ):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = torch.device(device)

        print(f"[*] Initializing Guided Clinical Engine on device: {self.device}")

        # 1. Load Tokenizer & Multi-Task Model
        local_model_dir = os.path.join(PROJECT_ROOT, "outputs", "indobert_base_local")
        model_name = local_model_dir if os.path.exists(local_model_dir) else "indobenchmark/indobert-base-p1"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = IndoMentalBERT(model_name=model_name).to(self.device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            # Load with strict=False to accommodate newly initialized multi-task classification heads
            self.model.load_state_dict(state_dict, strict=False)
            print("[✓] Pretrained IndoMentalBERT weights mapped.")

        self.model.eval()

        # 2. Load FAISS Index and Response Catalog
        self.faiss_index = faiss.read_index(faiss_index_path)
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        # 3. Load Clinical Knowledge Graph
        self.kg = ClinicalKnowledgeGraph()
        print(f"[✓] Knowledge Graph & FAISS Index ({self.faiss_index.ntotal} vectors) synchronized.")

    def retrieve_with_clinical_reasoning(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes dual symbolic-neural clinical retrieval:
        1. Symbolic traversal via ClinicalKnowledgeGraph
        2. Vector search via FAISS + IndoMentalBERT
        3. Explainable Clinical Synthesis
        """
        # Step A: Knowledge Graph Symbolic Triaging
        kg_analysis = self.kg.reason(query)

        # Step B: Neural Vector Encoding
        inputs = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            q_emb = self.model.encode_text(inputs["input_ids"], inputs["attention_mask"], is_question=True)
            q_vec = q_emb.cpu().numpy().astype("float32")

        # Step C: Sub-millisecond FAISS Search
        D, I = self.faiss_index.search(q_vec, k=top_k)

        candidates = []
        for rank in range(top_k):
            idx = int(I[0][rank])
            score = float(D[0][rank])
            text = self.catalog[idx]
            candidates.append({"rank": rank + 1, "score": score, "response": text})

        best_candidate = candidates[0]

        # Step D: Explainable Decision Packaging
        return {
            "query": query,
            "kg_triaging": kg_analysis,
            "retrieved_candidates": candidates,
            "selected_response": best_candidate["response"],
            "semantic_confidence": best_candidate["score"],
            "clinical_explainability": {
                "detected_symptoms": kg_analysis["symptoms_found"],
                "presumed_disorder": kg_analysis["disorder"],
                "risk_severity": kg_analysis["risk_level"],
                "cognitive_distortion_detected": kg_analysis["cognitive_distortions"],
                "evidence_based_intervention": kg_analysis["clinical_intervention"]
            }
        }
