"""
Atomic Response Builder for IndoMentalBERT.
Structures raw clinical consultation essays into 3 modular atomic tiers:
  Tier 1: Empathetic Validation (Active Listening)
  Tier 2: Psychoeducational Core (Insight & Mechanism)
  Tier 3: Actionable Coping / Professional Triage
"""

import re
import random

# Pre-formulated clinical validation openers (Tier 1)
VALIDATION_OPENERS = {
    "cemas": [
        "Wajar sekali merasa cemas atau tegang ketika menghadapi situasi yang tidak menentu.",
        "Kecemasan adalah respon alami tubuh terhadap stres, dan kamu tidak harus menanggungnya sendirian.",
        "Terima kasih sudah berani mengungkapkan rasa cemasmu; mengakui perasaan ini adalah langkah awal yang sangat berharga."
    ],
    "lelah": [
        "Wajar sekali merasa lelah setelah memikul banyak beban dan tekanan.",
        "Kelelahan mental adalah sinyal bahwa tubuh dan pikiranmu sedang membutuhkan jeda istirahat.",
        "Jangan merasa bersalah karena merasa lelah; memberi ruang bagi diri sendiri untuk berhenti sejenak adalah hal yang penting."
    ],
    "sedih": [
        "Tidak apa-apa untuk merasa sedih atau hampa hari ini. Jangan terlalu keras pada dirimu sendiri.",
        "Perasaan sedih dan berat yang kamu rasakan adalah hal yang nyata dan sangat valid.",
        "Kamu tidak sendirian dalam menghadapi masa-masa yang berat ini."
    ],
    "umum": [
        "Terima kasih sudah berbagi ceritamu di ruang aman ini.",
        "Saya memahami bahwa situasi yang sedang kamu alami bukanlah hal yang mudah.",
        "Mendengarkan apa yang kamu rasakan adalah prioritas di sini."
    ]
}

# Actionable Coping & Referral Closers (Tier 3)
ACTIONABLE_CLOSERS = {
    "cemas": [
        "Cobalah teknik pernapasan 4-7-8 (tarik 4 detik, tahan 7 detik, hembuskan 8 detik) untuk meredakan ketegangan fisikmu.",
        "Jika rasa cemas ini terus bertahan dan mengganggu tidurmu, membicarakannya dengan psikolog akan sangat membantumu."
    ],
    "lelah": [
        "Cobalah luangkan waktu 10 menit tanpa gawai, minumlah air hangat, dan beri izin pada dirimu untuk beristirahat.",
        "Menetapkan batasan yang sehat dan berani menolak beban berlebih adalah langkah penting mencegah kelelahan berlanjut."
    ],
    "sedih": [
        "Cobalah untuk melakukan satu hal kecil yang menenangkanmu hari ini, seperti berjalan santai atau menuliskan apa yang kamu rasakan.",
        "Bila perasaan hampa ini berlangsung lebih dari dua minggu, sangat disarankan untuk berkonsultasi dengan profesional kesehatan jiwa."
    ],
    "umum": [
        "Luangkan waktu sejenak untuk menarik napas dalam, dan ingatlah bahwa ada bantuan profesional yang selalu siap mendukungmu.",
        "Merawat kesehatan mental adalah proses bertahap. Jangan ragu mencari pendampingan psikolog bila terasa terlalu berat."
    ]
}


def detect_topic(text: str) -> str:
    """Detect general topic for validation category."""
    lower = text.lower()
    if any(k in lower for k in ["cemas", "panik", "deg-degan", "overthinking", "takut", "anxiety"]):
        return "cemas"
    if any(k in lower for k in ["lelah", "capek", "burnout", "letih", "drop", "kurang tidur"]):
        return "lelah"
    if any(k in lower for k in ["sedih", "hampa", "menangis", "kecewa", "depresi", "down", "patah"]):
        return "sedih"
    return "umum"


def extract_core_insight(raw_essay: str, max_sentences: int = 2) -> str:
    """
    Extract the most informative core insight from a long forum essay.
    Picks clean, self-contained sentences avoiding forum noise.
    """
    cleaned = re.sub(r"\s+", " ", raw_essay).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip().split()) >= 4]

    # Filter out forum greeting artifacts like "Halo terima kasih atas pertanyaan Anda..."
    filtered = []
    for s in sentences:
        s_lower = s.lower()
        if s_lower.startswith(("halo dan terima kasih", "terima kasih atas pertanyaan", "robin landwehr", "dr.")):
            continue
        filtered.append(s)

    if not filtered:
        filtered = sentences

    selected = filtered[:max_sentences]
    return " ".join(selected)


def build_atomic_response(user_query: str, raw_retrieved_essay: str, score: float) -> str:
    """
    Assembles a modular, 3-tier atomic clinical response:
      Tier 1: Validation
      Tier 2: Core Insight
      Tier 3: Actionable Step
    """
    topic = detect_topic(user_query)

    # If the user query is very short, add Tier 1 validation opener
    words_in_query = len(user_query.split())
    opener = random.choice(VALIDATION_OPENERS.get(topic, VALIDATION_OPENERS["umum"]))

    # Tier 2: Extract core clinical insight from the BERT-matched response
    insight = extract_core_insight(raw_retrieved_essay, max_sentences=2)

    # Tier 3: Actionable closing / referral
    closer = random.choice(ACTIONABLE_CLOSERS.get(topic, ACTIONABLE_CLOSERS["umum"]))

    # Assemble response
    if words_in_query <= 6:
        # For short queries: Opener + Actionable Closer
        return f"{opener} {closer}"
    else:
        # For detailed queries: Opener + Clinical Insight + Closer
        return f"{opener} {insight} {closer}"
