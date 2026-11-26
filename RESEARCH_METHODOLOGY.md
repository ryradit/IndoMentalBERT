# Research Methodology: IndoMentalBERT Dual Symbolic-Neural Clinical Framework

## Abstract & Research Novelty
Most contemporary conversational systems rely on Generative Large Language Models (LLMs), which carry significant clinical hazards in psychiatry, including **hallucinations, non-deterministic reasoning, and unverified medical advice**. 

This research introduces **IndoMentalBERT**, a novel **Dual Symbolic-Neural Framework** specifically designed for Indonesian digital mental health support. The framework establishes novelty through two distinct architectural contributions:

1. **Multi-Task Hard-Parameter Sharing Dual-Encoder**: Combines sub-millisecond dense retrieval with joint clinical risk triaging (Low, Mild, Moderate, Severe, Emergency) and cognitive distortion detection.
2. **Indonesian Psychiatric Knowledge Graph Guided Retrieval**: Combines PDSKJI (Perhimpunan Dokter Spesialis Kedokteran Jiwa Indonesia) and DSM-5 clinical ontologies to map somatic complaints (e.g., GERD, palpitations, tremors) directly to evidence-based therapeutic protocols (e.g., Somatosensory Grounding, Box Breathing, Behavioral Activation).

---

## 1. System Architecture

```text
[ User Utterance (Indonesian Colloquial / Slang) ]
                       │
                       ▼
       [ Indonesian Slang Normalizer ]
                       │
                       ▼
         [ Multi-Turn Context Memory ]
                       │
       ┌───────────────┴────────────────────────┐
       ▼                                        ▼
[ Symbolic Stream ]                      [ Neural Stream ]
Indonesian Psychiatric                   IndoMentalBERT Shared Transformer
Knowledge Graph (Ontology)               (indobenchmark/indobert-base-p1)
       │                                        │
       ├─ Somatic & Affective Symptoms          ├─ Head 1: Retrieval Embedding (768d)
       ├─ Presumed Disorder Category            ├─ Head 2: Clinical Risk Classifier
       ├─ Cognitive Distortion Detection        └─ Head 3: Cognitive Distortion Head
       └─ Evidence-Based Protocol Anchor                │
                       │                                ▼
                       │                   [ FAISS Vector Index ]
                       │                   3,504 Validated Responses
                       │                   (< 1ms Inner Product Search)
                       │                                │
                       └───────────────┬────────────────┘
                                       ▼
                    [ Guided Clinical Reasoning Engine ]
                    - Sub-millisecond Candidate Selection
                    - Clinical Explainability (XAI Metadata)
                    - Modular 3-Tier Atomic Response Assembler
                                       │
                                       ▼
                   [ Safe, Non-Generative Clinical Output ]
```

---

## 2. Mathematical Formulation

### 2.1 Multi-Task Hard-Parameter Sharing
Given a shared representation $h_{CLS} = \text{Transformer}(x)$, the multi-task learning objective minimizes the joint loss:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{retrieval} + \lambda_2 \mathcal{L}_{risk} + \lambda_3 \mathcal{L}_{distortion}$$

Where:
- $\mathcal{L}_{retrieval}$ is the Contrastive Cosine Margin / InfoNCE Loss between query representation $q = \text{FFN}_Q(h_{CLS})$ and validated response representation $a = \text{FFN}_A(h_{CLS})$.
- $\mathcal{L}_{risk}$ is the categorical cross-entropy over 5 clinical triage tiers ($\text{Low}, \text{Mild}, \text{Moderate}, \text{Severe}, \text{Emergency}$).
- $\mathcal{L}_{distortion}$ is the multi-label cross-entropy over cognitive distortion archetypes (Catastrophizing, All-or-Nothing, Emotional Reasoning, etc.).

### 2.2 Sub-Millisecond Dense Retrieval via FAISS
Embeddings are L2-normalized:
$$\hat{q} = \frac{q}{\|q\|_2}, \quad \hat{a}_i = \frac{a_i}{\|a_i\|_2}$$
Thus, inner product is equivalent to cosine similarity:
$$\text{Sim}(\hat{q}, \hat{a}_i) = \hat{q} \cdot \hat{a}_i$$
Search across the pre-computed catalog $A = \{\hat{a}_1, \dots, \hat{a}_N\}$ is executed via FAISS Flat Inner Product indexing in $\mathcal{O}(N \cdot d)$ with sub-millisecond latency.

---

## 3. Empirical Evaluation & Benchmarks

Conducted on 2,000 held-out clinical validation pairs:

| Evaluation Metric | Value | Significance |
| :--- | :---: | :--- |
| **Accuracy** | **80.10%** | Overall semantic retrieval precision |
| **Precision** | **84.60%** | Reliability of clinical advice delivered |
| **Recall** | **73.60%** | Sensitivity in identifying matching clinical pairs |
| **F1-Score** | **78.72%** | Robust harmonic balance |
| **FAISS Latency** | **< 1.0 ms** | Real-time conversational throughput on standard CPU |
| **True Positives (TP)** | 736 | Verified matching pairs |
| **True Negatives (TN)** | 866 | Correctly rejected clinically invalid pairs |
| **False Positives (FP)** | 134 | Low error rate (13.4%) |
| **False Negatives (FN)** | 264 | Borderline threshold matches |

---

## 4. Clinical Explainability (XAI)
To satisfy clinical transparency guidelines, typing `/diag` in the interactive console activates researcher inspection mode, displaying:
- Exact somatic and psychological symptoms isolated
- Clinical category mapped from PDSKJI/DSM-5 guidelines
- Cognitive distortions detected in patient speech
- Evidence-based medical intervention rationale
- Exact FAISS semantic vector alignment score

---

## Citation & Attribution

If you utilize IndoMentalBERT in psychiatric NLP research, cite as:
```bibtex
@article{radityatama2026indomentalbert,
  title={IndoMentalBERT: A Deterministic, Non-Generative Dual-Encoder Framework with Sub-Millisecond FAISS Retrieval and Clinical Knowledge Graph Triaging},
  author={Radityatama, Ryan},
  journal={IndoMentalBERT Research Repository},
  year={2026},
  url={https://github.com/ryradit/IndoMentalBERT}
}
```
