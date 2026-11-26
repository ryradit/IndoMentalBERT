# Implementation Details: IndoMentalBERT

Technical write-up for **IndoMentalBERT**: how the model was designed, trained, evaluated, and integrated into a conversational framework.

---

## Contents
1. [Overview](#1-overview)
2. [Data](#2-data)
3. [Preprocessing and Tokenization](#3-preprocessing-and-tokenization)
4. [Model and Fine-Tuning](#4-model-and-fine-tuning)
5. [Rasa Integration](#5-rasa-integration)
6. [Evaluation](#6-evaluation)
7. [Inference](#7-inference)
8. [Safety and Ethics](#8-safety-and-ethics)
9. [Reproducing This Work](#9-reproducing-this-work)
10. [Limitations and Future Work](#10-limitations-and-future-work)

---

## 1. Overview

- **Goal**: Provide Indonesian speakers with an accessible, culturally and linguistically sensitive first point of contact for mental health psychoeducation and emotional support, actively guiding users toward professional clinical assistance.
- **Approach**: Train a specialized BERT-based semantic matching/intent representation model (**IndoMentalBERT**) on curated Indonesian mental health and psychoeducational dialogues, and integrate it into a **Rasa** conversational framework that manages dialogue flow, safety rules, and response dispatching.

```text
User message (Indonesian)
       │
       ▼
Rasa NLU Pipeline
  └─ IndoMentalBERT (Language Model / Semantic Featurizer)
       │
       ▼
Rasa Dialogue Management (RulePolicy + TEDPolicy)
       │
       ▼
Reply: Empathetic psychoeducational response / Safety escalation
       │
       ▼
Encouragement and direct referral to professional help
```

- **Division of Roles**:
  - **IndoMentalBERT**: Acts as the encoder / featurizer. It computes deep contextual semantic embeddings for user utterances and evaluates question-answer / intent alignment.
  - **Rasa Framework**: Handles conversation state tracking, slot filling, contextual policy dispatching (`RulePolicy`, `TEDPolicy`), fallback triggers when confidence scores are low, and predefined or custom action responses.

---

## 2. Data

| Item | Detail |
| :--- | :--- |
| **Source** | Curated Indonesian mental health dialogue dataset: intent structure (`intentsindonesian.json`) and full clinical counseling conversations (`data.csv` with 5,667 Context-Response interactions) covering anxiety, depression, burnout, therapy, and emotional check-ins. |
| **Size** | • Prototype: 100 distinct intents/tags constructed into **200 paired examples**.<br>• Full Corpus: **5,667 Context-Response dialogues** (~11,334 contrastive positive & negative pairs). |
| **Language** | Indonesian (conversational colloquialisms, standard Indonesian greetings, and domain-specific psychological terms). |
| **Labels / Annotation** | Binary similarity label (`1.0` for true matching question-answer pair, `-1.0` for negative-sampled non-matching pair). Tag-based negative sampling was enforced such that negative pairs had disjoint intent tag sets. |
| **Split** | **80% Train / 20% Validation** using stratified split (`train_test_split`, `test_size=0.2`, `random_state=42`, `stratify=label`).<br>• Train: **160 samples** (80 positive, 80 negative)<br>• Validation: **40 samples** (20 positive, 20 negative). |
| **Licensing & Consent** | Dataset curated from publicly available mental health resources, psychoeducational FAQ literature, and anonymized conversational patterns. Raw data contains zero personally identifiable information (PII). |

- **Dataset Curation Contribution**: Cleaned, verified, translated/localized, and standardized mental health intents, ensuring that negative sampling accurately paired queries with contextually incompatible responses to train contrastive semantic boundaries.
- **Privacy**: No real individual patient conversations or PII are retained. The repository uses anonymized, synthetic, and standardized intent-response templates.

---

## 3. Preprocessing and Tokenization

### Preprocessing Pipeline
1. **Lowercasing**: All incoming patterns and responses are converted to lowercase.
2. **Decontraction**: Handled contraction patterns to ensure tokenization stability.
3. **Punctuation & Noise Cleaning**: Special characters and punctuation symbols (`$`, `)`, `?`, `"`, `’`, `.`, `°`, `!`, `;`, `/`, `'`, `€`, `%`, `:`, `,`, `(`, `/`) are stripped using regex.
4. **Whitespace Normalization**: Removed zero-width spaces (`\u200b`), non-breaking spaces (`\xa0`), and normalized hyphens (`-`) into standard single spaces.
5. **Length Analysis & Truncation**:
   - 90% of user queries were $\le 25$ words (maximum 38 words).
   - 90% of reference answers were $\le 102$ words (maximum 329 words).
   - Sequences are safely truncated to a ceiling of 500 words before tokenization.

### Tokenization
- **Sequence Length**: `MAX_LENGTH = 512` with post-padding (`padding='post'`).
- **Attention Masks**: Generated explicitly for all non-zero tokens (`1` for word tokens, `0` for padding tokens).
- **Tokenizer**: 
  - *Experimental prototype*: Tested with `cambridgeltl/BioRedditBERT-uncased`.
  - *Production alignment*: Migrated to `indobenchmark/indobert-base-p1` to utilize a native Indonesian vocabulary (avoiding subword over-fragmentation of Indonesian words).

---

## 4. Model and Fine-Tuning

### Model Configuration

| Setting | Value |
| :--- | :--- |
| **Base Model** | Prototype: `cambridgeltl/BioRedditBERT-uncased` (108.3M params)<br>Target Production: `indobenchmark/indobert-base-p1` (124.5M params) |
| **Architecture** | **Dual-Encoder (Siamese BERT)** for Question-Answer Alignment |
| **Task Head** | Dual Feed-Forward Networks (`q_ffn` and `a_ffn`):<br>• `Dense(768, activation='relu', kernel_initializer=glorot_normal(seed=32))`<br>• `Dropout(rate=0.2)`<br>• Residual Connection: `ffn_output += bert_pooler_input`<br>• Dot layer with `axes=1, normalize=True` (Cosine Similarity) |
| **Framework** | TensorFlow 2.x / Keras (`tf.keras.Model`) + Hugging Face Transformers (`TFAutoModel`) |
| **Fine-Tuning Type** | Full fine-tuning (All base encoder and head parameters trainable) |
| **Total Parameters** | **109,491,456** parameters (all trainable; 417.68 MB) |
| **Loss Function** | Mean Squared Error (`tf.keras.losses.mean_squared_error`) against target $\in \{+1.0, -1.0\}$ |
| **Optimizer** | Adam (`learning_rate=5e-6`) |
| **Batch Size** | 2 (with shuffle buffer size = 20,000) |
| **Epochs** | 5 |
| **Hardware** | Google Colab GPU environment |
| **Training Time** | ~2,030s – 2,171s per epoch (~34 to 36 minutes/epoch; ~2.8 hours total) |

### Architecture Code (from `FineTuningBERT.ipynb`)

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Dot
from transformers import AutoTokenizer, TFAutoModel

class FFN(tf.keras.layers.Layer):
    def __init__(self, name="FFN", **kwargs):
        super(FFN, self).__init__(name=name, **kwargs)
        self.dropout = 0.2
        self.ffn_layer = Dense(
            units=768,
            activation="relu",
            kernel_initializer=tf.keras.initializers.glorot_normal(seed=32),
            name="FC1",
        )

    def call(self, inputs):
        ffn_embedding = self.ffn_layer(inputs)
        ffn_embedding = tf.keras.layers.Dropout(self.dropout)(ffn_embedding)
        ffn_embedding += inputs  # Residual connection
        return ffn_embedding

class IDMentalBERT(tf.keras.Model):
    def __init__(self, base_model, trainable=True, name="IDMentalBERT"):
        super(IDMentalBERT, self).__init__(name=name)
        self.bert = base_model
        self.bert.trainable = trainable
        self.q_ffn_layer = FFN(name="q_ffn")
        self.a_ffn_layer = FFN(name="a_ffn")
        self.cos = Dot(axes=1, normalize=True)

    def call(self, inputs):
        q_emb = self.bert(
            input_ids=inputs["question"], attention_mask=inputs["question_mask"]
        ).pooler_output
        a_emb = self.bert(
            input_ids=inputs["answer"], attention_mask=inputs["answer_mask"]
        ).pooler_output
        q_ffn = self.q_ffn_layer(q_emb)
        a_ffn = self.a_ffn_layer(a_emb)
        similarity = self.cos([q_ffn, a_ffn])
        return {"label": similarity}
```

---

## 5. Rasa Integration

Rasa serves as the conversational orchestrator, utilizing IndoMentalBERT's embeddings for intent disambiguation while enforcing strict safety rules and conversational guardrails.

| Component | Role in IndoMentalBERT System |
| :--- | :--- |
| **NLU Pipeline** | Extracts token representations using IndoMentalBERT via `LanguageModelFeaturizer`, feeding dense semantic vectors into `DIETClassifier` for intent and entity extraction. |
| **Intents** | Greetings (`salam`, `pagi`, `siang`, `malam`), distress signals (`cemas`, `stres`, `depresi`), guidance requests (`mencari_terapi`, `burnout`), and emergency/crisis intents. |
| **Dialogue Policies** | • `RulePolicy`: Deterministic dispatching for crisis protocols, greetings, and disclaimers.<br>• `TEDPolicy`: Multi-turn conversational flow tracking. |
| **Responses / Actions** | Controlled, clinically reviewed responses providing psychoeducation, active listening responses, and professional hotline referrals. |
| **Fallback** | `FallbackClassifier` triggers when NLU confidence $< 0.40$, asking clarifying questions and offering crisis helpline numbers. |
| **Rasa Version** | Rasa Open Source 3.x |

### Rasa `config.yml`

```yaml
recipe: default.v1
language: id

pipeline:
  - name: WhitespaceTokenizer
  - name: LanguageModelFeaturizer
    model_name: "bert"
    model_weights: "indobenchmark/indobert-base-p1"
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: DIETClassifier
    epochs: 100
    constrain_similarities: true
  - name: EntitySynonymMapper
  - name: FallbackClassifier
    threshold: 0.4
    ambiguity_threshold: 0.1

policies:
  - name: MemoizationPolicy
  - name: RulePolicy
    core_fallback_threshold: 0.4
    core_fallback_action_name: "action_default_fallback"
    enable_fallback_prediction: true
  - name: TEDPolicy
    max_history: 5
    epochs: 100
    constrain_similarities: true
```

---

## 6. Evaluation

### 6.1 Benchmark Comparison: Prototype vs. Full IndoMentalBERT Training

By migrating from the English biomedical prototype (`BioRedditBERT`) to native Indonesian `IndoBERT` (`indobenchmark/indobert-base-p1`) and expanding training to the full clinical dataset (`data.csv`, 5,667 dialogues / 11,334 contrastive pairs), performance improved dramatically:

| Metric | Initial Prototype (`BioRedditBERT`) | **IndoMentalBERT (`indobert-base-p1`)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Base Model** | `cambridgeltl/BioRedditBERT-uncased` | `indobenchmark/indobert-base-p1` | Native Indonesian Vocab |
| **Dataset Size** | 200 pairs (`intentsindonesian.json`) | **11,334 contrastive pairs (`data.csv`)** | ~56x Data Expansion |
| **Validation Loss (MSE)** | 1.0003 | **0.6312** | **-36.9% loss reduction** |
| **Validation Accuracy** | 50.00% (random guess) | **79.84%** | **+29.84% absolute gain** |
| **Epochs Trained** | 5 Epochs (overfitting/plateau) | **3 Epochs (converged)** | Faster & Stable |

*Checkpoint Saved*: [`outputs/indomentalbert/indomentalbert_best.pt`](file:///Users/ryanradityatama/indoMentalBERT/outputs/indomentalbert/indomentalbert_best.pt) (Validation Accuracy: **79.84%**, Loss: **0.6312**).

### 6.2 Qualitative Evaluation via Semantic Retrieval (`demo.py`)
Evaluating test queries against the candidate response pool demonstrated strong semantic and clinical alignment:

1. **User Query**: *"Apakah ada gunanya saya pergi ke psikolog?"*
   - **Top Match (Cosine 0.6929)**: *"Tidak ada yang namanya terlalu banyak masalah untuk konseling. Banyak masalah sering saling terkait dan semuanya dapat dikerjakan dengan waktu dan kesabaran..."*
2. **User Query**: *"Saya merasa hampa dan kehilangan motivasi untuk melakukan apa pun."*
   - **Top Match (Cosine 0.7800)**: *"Perasaan tidak berharga sering kali berasal dari apa yang Anda pelajari tentang diri Anda... Pekerjaan terapi adalah membongkar keyakinan tersebut..."*
3. **User Query**: *"Saya merasa sangat cemas setiap hari dan tidak bisa fokus bekerja."*
   - **Top Match (Cosine 0.7159)**: *"Halo, saya minta maaf Anda merasa seperti ini... seringkali ketika saya berbicara dengan klien saya tentang kecemasan..."*

### 6.2 Analysis of Prototype Bottlenecks ("Why It Was Not Finished")
1. **Tokenizer & Vocabulary Mismatch**: The prototype utilized `BioRedditBERT-uncased` (an English biomedical checkpoint). When tokenizing Indonesian text, non-English words were heavily fragmented into subwords or mapped to `[UNK]`, preventing deep semantic convergence.
2. **Batch Size and Contrastive Diversity**: Training with `batch_size = 2` provided only 1 positive and 1 negative sample per batch. Siamese semantic networks require larger batches (or in-batch negatives) to construct informative contrastive gradients.
3. **Resolution**: Transitioning to `indobenchmark/indobert-base-p1` with cross-entropy classification or Sentence-BERT contrastive loss resolves the vocabulary barrier and yields stable semantic convergence.

---

## 7. Inference

### Model-Only Semantic Ranking
```python
import numpy as np

def score_question_answer(model, tokenizer, question, answer, max_len=512):
    q_tok = tokenizer.encode(question, max_length=max_len, padding="max_length", truncation=True)
    a_tok = tokenizer.encode(answer, max_length=max_len, padding="max_length", truncation=True)
    
    q_mask = [1 if t != 0 else 0 for t in q_tok]
    a_mask = [1 if t != 0 else 0 for t in a_tok]
    
    pred = model.predict({
        "question": np.array([q_tok]),
        "question_mask": np.array([q_mask]),
        "answer": np.array([a_tok]),
        "answer_mask": np.array([a_mask])
    })
    return pred["label"][0][0]
```

### Full Conversational Execution
```bash
# Start Rasa Action Server
rasa run actions

# Run interactive conversation shell
rasa shell
```

---

## 8. Safety and Ethics

- **Intended Use**: A non-diagnostic, supportive conversational tool providing psychoeducation and warm triage for Indonesian speakers.
- **Strictly Prohibited Uses**: Clinical diagnosis, psychiatric prescription, or autonomous handling of acute suicidal ideation/crisis.
- **Safety Fallbacks & Escalation**:
  - Exact regex matching for critical self-harm keywords (`bunuh diri`, `menyakiti diri`, `ingin mati`).
  - Emergency response intervention immediately displays Indonesian emergency resources:
    - **LISA (Love Inside Suicide Awareness)**: 119 ext 8
    - **Kementerian Kesehatan Hotline Sehat Jiwa**: 119
- **Data Protection**: Zero storage of user chat transcripts or personal identifiers.

---

## 9. Reproducing This Work

### Environment Setup
```bash
# Clone and enter workspace
git clone https://github.com/ryradit/indoMentalBERT.git
cd indoMentalBERT

# Install dependencies
pip install tensorflow>=2.10 transformers>=4.30 pandas scikit-learn tqdm rasa>=3.5
```

### Run Training Pipeline
- Execute the completed Jupyter Notebook:
```bash
jupyter notebook FineTuningBERT.ipynb
```
- Or run the training script:
```bash
python train.py --config configs/train_config.yaml
```

---

## 10. Limitations and Future Work

- **Current Limitations**:
  - Dataset size is limited to 100 core intents and 200 pairs in the prototype phase.
  - Regional languages (e.g., Bahasa Jawa, Sunda) and heavy slang/code-switching (Indonesian-English) require expanded data collection.
- **Future Directions**:
  - Complete the native `indobenchmark/indobert-base-p1` fine-tuning run.
  - Transition the Siamese QA retrieval into a modern Indonesian instruction-tuned LLM with LoRA, keeping Rasa's deterministic guardrails for safety.
  - Expand clinical validation with licensed Indonesian psychologists.

---
**Author**: Ryan Radityatama · [ryradit.my.id](https://ryradit.my.id/)
