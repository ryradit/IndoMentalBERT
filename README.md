# IndoMentalBERT: Indonesian Language Model for Mental Health Support

IndoMentalBERT is a domain-adapted Indonesian language model and dialogue system developed to provide accessible mental health psychoeducation, triage, and supportive conversational responses.

Built on an encoder-only architecture (`indobenchmark/indobert-base-p1`) with deterministic clinical safety rules, IndoMentalBERT prioritizes verified clinical guidance and zero hallucination.

---

## System Architecture

The pipeline consists of four modular stages:

1. **Indonesian Slang Normalizer (`src/slang_normalizer.py`)**: Converts colloquial terms, youth slang, and informal contractions into standard Indonesian before tokenization.
2. **Multi-Turn Context Buffer (`DialogueMemory`)**: Retains recent conversational turns to contextualize short user follow-ups.
3. **IndoMentalBERT Dual-Encoder (`src/model.py`)**: Encodes user queries and candidate responses with Feed-Forward projection heads, evaluating cosine alignment against clinical data.
4. **Atomic Response Builder (`src/response_builder.py`)**: Assembles structured 3-tier advice (Empathetic Validation + Educational Insight + Actionable Step).
5. **Deterministic Crisis Safeguards**: Regex-based detection for self-harm keywords that directly surfaces Indonesian emergency helplines.

---

## Evaluation & Retrieval Metrics

Comprehensive evaluation was conducted on 2,000 held-out pairs (1,000 ground-truth clinical pairs vs 1,000 semantic contrastive negatives) and tested across **FAISS Sub-millisecond Vector Indexing**:

![IndoMentalBERT Evaluation Metrics](reports/evaluation_metrics.png)

### 1. Classification & Retrieval Performance
| Metric | Value | Description |
| :--- | :---: | :--- |
| **Accuracy** | **80.10%** | Overall semantic matching accuracy across query-response pairs |
| **Precision** | **84.60%** | Reliability of selected clinical responses ($TP / (TP + FP)$) |
| **Recall** | **73.60%** | Coverage of true relevant clinical counseling pairs ($TP / (TP + FN)$) |
| **F1-Score** | **78.72%** | Harmonic balance between Precision and Recall |

### 2. Confusion Matrix
| Metric | Count | Clinical Interpretation |
| :--- | :---: | :--- |
| **True Positives (TP)** | **736** | Accurately matched patient query to valid counseling response |
| **True Negatives (TN)** | **866** | Correctly rejected irrelevant / mismatched counseling responses |
| **False Positives (FP)** | **134** | Low false alignment rate (**13.4%**) |
| **False Negatives (FN)** | **264** | Clinical response relevant but fell below standard threshold |

### 3. N-gram Precision (BLEU Scores)
Evaluated against original clinical human counselor responses:
- **BLEU-1 (Unigram - Word Alignment)**: **13.09%**
- **BLEU-2 (Bigram - Phrase Precision)**: **5.34%**
- **BLEU-3 (Trigram - Coherent Advice Patterns)**: **3.41%**
- **BLEU-4 (4-gram - Full Sentence Structure Match)**: **2.90%**

*(In non-generative mental health retrieval, lower BLEU-4 is expected and healthy, representing semantic paraphrasing and diverse clinical delivery rather than verbatim word duplication.)*

### 4. FAISS Sub-Millisecond Vector Search
- **Indexed Responses**: 3,504 unique validated clinical counseling responses from `data.csv`.
- **Search Latency**: **< 1 millisecond (< 0.001 s)** on standard CPU.
- **Index Dimensions**: 768-dimensional normalized embeddings (`outputs/indomentalbert/counseling_index.faiss`).

---

## Directory Structure

```text
IndoMentalBERT/
├── data/
│   └── intentsindonesian.json       # Categorized mental health intent data
├── data.csv                         # 5,667 counseling dialogue corpus
├── src/
│   ├── model.py                     # Siamese IndoMentalBERT architecture
│   ├── preprocess.py                # Dataset preparation and contrastive pairs
│   ├── slang_normalizer.py          # Indonesian slang normalization rules
│   ├── response_builder.py          # Modular 3-tier response assembler
│   └── train.py                     # Model training and checkpointing script
├── rasa/                            # Rasa NLU configuration and policies
├── outputs/                         # Trained model checkpoint storage
├── chat.py                          # Interactive CLI conversational agent
├── demo.py                          # Top-k response retrieval script
├── FineTuningIndoBERT_Updated.ipynb # Google Colab and Jupyter notebook
├── IMPLEMENTATION_DETAILS.md        # Technical documentation and write-up
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## Quickstart

### Setup
```bash
git clone https://github.com/ryradit/IndoMentalBERT.git
cd IndoMentalBERT
pip install -r requirements.txt
```

### Interactive Chatbot
Run the conversational interface:
```bash
python3 chat.py
```

### Model Training
Train IndoMentalBERT on the counseling dataset:
```bash
python3 src/train.py --data_path data.csv --epochs 3 --batch_size 16
```

---

## Emergency Crisis Contacts (Indonesia)

If self-harm or suicidal ideation is detected, the system immediately surfaces:
- **LISA Suicide Prevention Helpline**: 119 ext 8 (24 Jam)
- **Kemenkes Hotline Kesehatan Jiwa**: 119
- **Layanan SEJIWA**: 119 ext 8

---

## Author

**Ryan Radityatama**  
Website: [ryradit.my.id](https://ryradit.my.id/)  
GitHub: [@ryradit](https://github.com/ryradit)
