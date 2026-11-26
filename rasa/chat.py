"""
IndoMentalBERT Smart Conversational Agent (Pure Non-LLM Clinical NLP).
Features:
1. Multi-Turn Dialogue Memory: Tracks context across turns so short follow-ups are context-aware.
2. Indonesian Slang & Colloquial Normalizer: Handles "bgt", "gw", "ga bs", "overthinking", etc.
3. Modular Atomic Response Builder: 3-tier clinical advice (Validation + Insight + Actionable Step).
4. Deterministic Clinical Safety: Instant regex hotline trigger for self-harm/suicidal ideation.
"""

import os
import sys
import re
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == "rasa":
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
else:
    PROJECT_ROOT = CURRENT_DIR

for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import faiss

try:
    from demo import load_model_and_pool, query_counselor
    from slang_normalizer import normalize_slang
    from response_builder import build_atomic_response
    from knowledge_graph import ClinicalKnowledgeGraph
except ImportError:
    from demo import load_model_and_pool, query_counselor
    from src.slang_normalizer import normalize_slang
    from src.response_builder import build_atomic_response
    from src.knowledge_graph import ClinicalKnowledgeGraph

# 1. Deterministic Crisis Regex Safeguard
CRISIS_KEYWORDS = [
    r"\bbunuh diri\b",
    r"\bmati saja\b",
    r"\bmengakhiri hidup\b",
    r"\bmenyakiti diri\b",
    r"\bgantung diri\b",
    r"\btelan racun\b",
    r"\bpotong urat\b",
    r"\bingin mati\b",
    r"\btidak ingin hidup\b"
]

CRISIS_RESPONSE = """\
⚠️ PERINGATAN KESELAMATAN & DUKUNGAN DARURAT ⚠️
Kamu sangat berharga, dan kamu tidak harus menanggung beban ini sendirian.
Ada pertolongan dan harapan untuk rasa sakit yang sedang kamu rasakan.

Mohon SEGERA hubungi layanan darurat kesehatan jiwa bebas pulsa (24 Jam):
• Hotline LISA Suicide Prevention: 119 ext 8
• Hotline Kesehatan Jiwa Kemenkes: 119
• Layanan SEJIWA: 119 ext 8
• Atau kunjungi Instalasi Gawat Darurat (IGD) Rumah Sakit terdekat sekarang juga.

Bantuan profesional siap mendengarkanmu tanpa penghakiman.
"""

GREETINGS = ["hai", "halo", "selamat pagi", "selamat siang", "selamat malam", "assalamualaikum", "permisi"]
GOODBYES = ["dadah", "selamat tinggal", "sampai jumpa", "pamit", "bye"]
THANKS = ["terima kasih", "makasih", "thanks", "tengkyu", "matur nuwun"]


class DialogueMemory:
    """Multi-Turn Conversational Memory Buffer."""
    def __init__(self, max_history: int = 3):
        self.history = []  # List of (user_text, bot_text)
        self.max_history = max_history

    def add_turn(self, user_text: str, bot_text: str):
        self.history.append((user_text, bot_text))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_contextual_query(self, current_user_text: str) -> str:
        """
        If current user text is short/follow-up, combine with previous topic context
        so IndoMentalBERT knows the background story.
        """
        words = current_user_text.strip().split()
        if not self.history or len(words) > 8:
            return current_user_text

        # Extract previous user message as conversational context
        prev_user_text, _ = self.history[-1]
        contextual_query = f"Konteks masalah: {prev_user_text}. Kondisi saat ini: {current_user_text}"
        return contextual_query


def check_hardcoded_rules(text: str):
    lower_text = text.lower()

    # Rule 1: Emergency Crisis Intervention
    for pattern in CRISIS_KEYWORDS:
        if re.search(pattern, lower_text):
            return CRISIS_RESPONSE

    # Rule 2: Greetings
    if any(re.search(rf"\b{g}\b", lower_text) for g in GREETINGS):
        return "Halo! Selamat datang di ruang aman IndoMentalBERT. Bagaimana perasaanmu hari ini? Ada yang ingin kamu ceritakan?"

    # Rule 3: Thanks
    if any(re.search(rf"\b{t}\b", lower_text) for t in THANKS):
        return "Sama-sama. Merawat kesehatan mental adalah langkah penting yang patut diapresiasi. Semoga harimu menjadi lebih tenang."

    # Rule 4: Goodbye
    if any(re.search(rf"\b{b}\b", lower_text) for b in GOODBYES):
        return "Sampai jumpa. Jaga dirimu baik-baik, dan jangan ragu kembali jika kamu butuh ruang untuk bercerita."

    return None


def run_chatbot():
    print("=" * 70)
    print("🧠 INDOMENTALBERT SMART CLINICAL AGENT (FAISS + MULTI-TURN + ATOMIC)")
    print("=" * 70)
    print("Memuat model dan basis data konseling...")

    ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "indomentalbert", "indomentalbert_best.pt")
    csv_path = os.path.join(PROJECT_ROOT, "data.csv")
    faiss_path = os.path.join(PROJECT_ROOT, "outputs", "indomentalbert", "counseling_index.faiss")
    catalog_path = os.path.join(PROJECT_ROOT, "outputs", "indomentalbert", "response_catalog.json")

    try:
        model, tokenizer, candidate_pool, device = load_model_and_pool(
            model_checkpoint=ckpt_path,
            data_path=csv_path,
            sample_pool_size=100
        )
    except Exception as e:
        print(f"[!] Gagal memuat model: {e}")
        return

    # Check if FAISS index is available for sub-millisecond retrieval across full catalog
    faiss_index = None
    response_catalog = None
    if os.path.exists(faiss_path) and os.path.exists(catalog_path):
        try:
            faiss_index = faiss.read_index(faiss_path)
            with open(catalog_path, "r", encoding="utf-8") as f:
                response_catalog = json.load(f)
            print(f"[✓] FAISS Vector Engine Aktif: {faiss_index.ntotal} respon klinis terindeks (<1ms search).")
        except Exception as e:
            print(f"[!] FAISS index warning: {e}, menggunakan fallback pool.")

    # Initialize Indonesian Psychiatric Clinical Knowledge Graph
    kg = ClinicalKnowledgeGraph()
    print("[✓] Psychiatric Knowledge Graph Aktif (Ontologi Gangguan & Distorsi Kognitif).")

    memory = DialogueMemory(max_history=3)

    print("\n[✓] IndoMentalBERT siap berdiskusi!")
    print("Fitur aktif:")
    print("  • Mesin Pencari Vektor FAISS Sub-milisecond (Non-Generatif, Aman)")
    print("  • Indonesian Psychiatric Knowledge Graph (Symptom-to-Diagnosis Reasoning)")
    print("  • Normalisasi bahasa gaul / singkatan (gw, bgt, ga bs, overthinking)")
    print("  • Memori percakapan multi-turn (kontekstual)")
    print("  • Rekomendasi klinis modular 3-tier (Validasi + Insight + Solusi)")
    print("  • [Tips Peneliti] Ketik '/diag' di awal pesan untuk melihat Clinical Reasoning XAI!")
    print("Ketik pesanmu di bawah ini (atau ketik 'keluar' / 'exit' untuk selesai):\n")

    while True:
        try:
            raw_user_input = input("Kamu > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSampai jumpa.")
            break

        if not raw_user_input:
            continue

        if raw_user_input.lower() in ["keluar", "exit", "quit"]:
            print("\nIndoMentalBERT > Terima kasih telah meluangkan waktu berbagi. Jaga dirimu baik-baik!\n")
            break

        # Check Researcher Diagnostic Mode Flag
        explain_mode = False
        active_input = raw_user_input
        if raw_user_input.startswith("/diag"):
            explain_mode = True
            active_input = raw_user_input[5:].strip()

        # Step 1: Check Deterministic Safety Rules
        rule_reply = check_hardcoded_rules(active_input)
        if rule_reply:
            print(f"\nIndoMentalBERT >\n{rule_reply}\n")
            memory.add_turn(active_input, rule_reply)
            continue

        # Step 2: Slang & Colloquial Normalization
        normalized_input = normalize_slang(active_input)

        # Step 3: Multi-Turn Context Fusion
        contextual_query = memory.get_contextual_query(normalized_input)

        # Step 4: Knowledge Graph Reasoning (Ontology & Cognitive Distortions)
        kg_diag = kg.reason(contextual_query)

        # Step 5: Semantic Matching (FAISS Vector Search or Fallback)
        raw_response = None
        score = 0.0

        if faiss_index is not None and response_catalog is not None:
            # Sub-millisecond FAISS query encoding & dot product
            inputs = tokenizer([contextual_query], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                q_emb = model.encode_text(inputs["input_ids"], inputs["attention_mask"], is_question=True).cpu().numpy().astype("float32")
            D, I = faiss_index.search(q_emb, k=1)
            idx = I[0][0]
            score = float(D[0][0])
            raw_response = response_catalog[idx]
        else:
            matches = query_counselor(
                contextual_query,
                model,
                tokenizer,
                candidate_pool,
                device,
                top_k=1,
                max_length=128
            )
            if matches:
                raw_response, score = matches[0]

        if raw_response:
            # Step 6: Modular Atomic Response Assembly
            atomic_reply = build_atomic_response(active_input, raw_response, score)
            
            # If in Researcher Diagnostic Mode (/diag), output clinical reasoning metadata
            if explain_mode:
                print("\n🔬 [RESEARCHER CLINICAL REASONING XAI]")
                print(f"  • Tingkat Risiko Klinis : {kg_diag['risk_level']}")
                print(f"  • Kategori Psikiatrik  : {kg_diag['disorder']}")
                print(f"  • Gejala Somatik/Afektif: {', '.join(kg_diag['symptoms_found']) if kg_diag['symptoms_found'] else 'Keluhan non-spesifik'}")
                print(f"  • Distorsi Kognitif     : {', '.join(kg_diag['cognitive_distortions'])}")
                print(f"  • Intervensi Klinis     : {kg_diag['clinical_intervention']['technique']}")
                print(f"  • Rationale Medis       : {kg_diag['clinical_intervention']['clinical_rationale']}")
                print(f"  • Skor FAISS Alignment  : {score*100:.2f}%\n")

            print(f"\nIndoMentalBERT [Kesesuaian: {score*100:.1f}%] >")
            print(f"{atomic_reply}\n")
            memory.add_turn(active_input, atomic_reply)
        else:
            fallback = "Saya memahami ini hal yang tidak mudah. Bisa tolong ceritakan sedikit lebih banyak agar saya bisa memahamimu dengan lebih baik?"
            print(f"\nIndoMentalBERT >\n{fallback}\n")
            memory.add_turn(active_input, fallback)


if __name__ == "__main__":
    run_chatbot()
