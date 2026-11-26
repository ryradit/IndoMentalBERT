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

## Evaluation

Model training on 11,334 contrastive pairs derived from 5,667 Indonesian mental health dialogues achieved:

| Architecture | Base Weights | Validation Loss | Validation Accuracy |
| :--- | :--- | :---: | :---: |
| Baseline Prototype | `BioRedditBERT-uncased` | 1.0003 | 50.00% |
| **IndoMentalBERT** | **`indobert-base-p1`** | **0.6312** | **79.84%** |

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
