"""
Clinical User State & Adaptive Decision Engine for IndoMentalBERT.

This module determines the patient's psychological stage and dynamic cognitive state,
then applies an evidence-based Clinical Policy Matrix to adapt response delivery style:

User Conditions / Archetypes:
1. ACUTE_DISTRESS (Crying, Panic, High Affective Load):
   -> Delivery: High Empathy + Somatic Grounding (NO essays, NO complex cognitive tasks).
2. FRUSTRATED_RESISTANT (Cynical, feels therapy doesn't work, exhausted):
   -> Delivery: Collaborative Autonomy + Validation (Avoid unsolicited advice, invite agency).
3. RUMINATIVE_OVERTHINKING (Stuck in what-if loops, cognitive paralysis):
   -> Delivery: Cognitive Reframing + Concrete 2-Minute Micro-Action.
4. LONELY_EMPTY (Apathy, numbness, lack of social connection):
   -> Delivery: Warm Presence + Gentle Behavioral Activation.
5. SEEKING_PSYCHOEDUCATION (Asks direct questions about mental health):
   -> Delivery: Clear Clinical Insight + Psychoeducational Context.
"""

import re
from typing import Dict, Any, Tuple

USER_STATE_PATTERNS = {
    "ACUTE_DISTRESS": [
        r"\bmenangis\b", r"\bnangis\b", r"\bgemetar\b", r"\btakut banget\b",
        r"\btakut mati\b", r"\bsesak\b", r"\bpanik\b", r"\bhancur\b", r"\bgak kuat lagi\b",
        r"\bga kuat lagi\b", r"\btolong\b", r"\bsakit banget dada\b"
    ],
    "FRUSTRATED_RESISTANT": [
        r"\bpercuma\b", r"\bgak ada gunanya\b", r"\bga guna\b", r"\bsudah coba semua\b",
        r"\bsia sia\b", r"\bngapain\b", r"\bbohong\b", r"\bterserah\b", r"\bmuak\b",
        r"\bgak mempan\b", r"\bga ngaruh\b"
    ],
    "RUMINATIVE_OVERTHINKING": [
        r"\boverthinking\b", r"\bmikirin terus\b", r"\bgimana kalau\b", r"\bbagaimana jika\b",
        r"\btakut salah\b", r"\botak buntu\b", r"\bgak bisa tidur mikirin\b",
        r"\bmutar-mutar\b", r"\bterngiang\b", r"\bragu\b"
    ],
    "LONELY_EMPTY": [
        r"\bkesepian\b", r"\bsendirian\b", r"\bhampa\b", r"\bkosong\b",
        r"\btidak ada yang peduli\b", r"\bga ada temen\b", r"\bgak punya siapa\b",
        r"\bisolasi\b", r"\bmerasa asing\b"
    ],
    "SEEKING_PSYCHOEDUCATION": [
        r"\bkenapa ya\b", r"\bapa penyebab\b", r"\bapakah normal\b", r"\bagaimana cara\b",
        r"\bapakah ini depresi\b", r"\bgejala apa\b", r"\btips\b", r"\bsaran\b"
    ]
}

CLINICAL_ADAPTATION_POLICIES = {
    "ACUTE_DISTRESS": {
        "persona_tone": "Tenang, Lembut, Mengayomi (Soothing & Low Cognitive Load)",
        "pacing": "Singkat & Somatosensori (Hindari kalimat panjang / ceramah)",
        "framing_prefix": "Aku ada di sini bersamamu. Tarik napas pelan-pelan dulu... kamu aman saat ini.",
        "action_directive": "Fokus pada tubuh: telapak kaki menyentuh lantai, hembuskan napas panjang 3 kali."
    },
    "FRUSTRATED_RESISTANT": {
        "persona_tone": "Validatif, Tidak Menggurui (Respectful of Autonomy)",
        "pacing": "Menghormati rasa lelah tanpa memaksakan optimisme palsu",
        "framing_prefix": "Sangat wajar jika kamu merasa muak dan lelah ketika rasanya semua usaha belum membuahkan hasil.",
        "action_directive": "Tidak ada paksaan untuk melakukan apa pun sekarang. Cukup istirahat sejenak dari tuntutan."
    },
    "RUMINATIVE_OVERTHINKING": {
        "persona_tone": "Fokus, Mengarahkan ke Realitas (Grounding in Action)",
        "pacing": "Memotong spiral pikiran berulang ke hal konkret",
        "framing_prefix": "Pikiranmu sedang bekerja terlalu keras mencoba mengontrol masa depan yang belum tentu terjadi.",
        "action_directive": "Tanyakan pada diri: 'Apakah kekhawatiran ini bisa aku selesaikan dalam 5 menit ke depan?' Jika tidak, letakkan dulu."
    },
    "LONELY_EMPTY": {
        "persona_tone": "Hangat, Hadir Menemani (Companionship & Empathy)",
        "pacing": "Membangun rasa terhubung dan keberhargaan diri",
        "framing_prefix": "Rasa hampa dan sendiri adalah beban yang berat untuk dipikul. Keberadaanmu berharga, dan kamu tidak sendirian di sini.",
        "action_directive": "Cobalah rasakan sentuhan selimut hangat atau minum segelas air hangat sebagai bentuk kepedulian kecil untuk dirimu."
    },
    "SEEKING_PSYCHOEDUCATION": {
        "persona_tone": "Objektif, Edukatif, Memberdayakan (Informative Psychoeducation)",
        "pacing": "Terstruktur dan berbasis pemahaman psikologis",
        "framing_prefix": "Pertanyaan yang sangat baik. Memahami apa yang terjadi di otak dan tubuh adalah langkah awal pemulihan.",
        "action_directive": "Gunakan pemahaman ini untuk mengenali pola tubuhmu tanpa menghakimi dirimu sendiri."
    },
    "GENERAL_NEUTRAL": {
        "persona_tone": "Empatik & Pendengar Aktif",
        "pacing": "Seimbang dan reflektif",
        "framing_prefix": "Terima kasih sudah mau berbagi cerita ini.",
        "action_directive": "Luangkan waktu sejenak untuk memproses perasaan ini dengan tenang."
    }
}


class ClinicalJudgementEngine:
    """
    Evaluates patient conversational dynamics to determine psychological state
    and adaptively tailor the retrieved counseling response.
    """

    def __init__(self):
        self.state_patterns = USER_STATE_PATTERNS
        self.policies = CLINICAL_ADAPTATION_POLICIES

    def evaluate_state(self, user_text: str, history_len: int = 0) -> Dict[str, Any]:
        """
        Classifies the psychological condition / state of the user.
        """
        text_lower = user_text.lower()
        matched_state = "GENERAL_NEUTRAL"
        highest_matches = 0
        evidence_words = []

        for state, patterns in self.state_patterns.items():
            matches = []
            for pat in patterns:
                found = re.findall(pat, text_lower)
                if found:
                    matches.extend(found)
            if len(matches) > highest_matches:
                highest_matches = len(matches)
                matched_state = state
                evidence_words = matches

        # Linguistic length heuristic:
        # Extremely short utterances during distress -> Acute Distress / Shutdown
        words_count = len(text_lower.split())
        if words_count <= 4 and matched_state == "GENERAL_NEUTRAL" and any(w in text_lower for w in ["lelah", "capek", "hancur", "muak"]):
            matched_state = "ACUTE_DISTRESS"
            evidence_words = ["short_affective_utterance"]

        policy = self.policies.get(matched_state, self.policies["GENERAL_NEUTRAL"])

        return {
            "user_condition": matched_state,
            "evidence_markers": list(set(evidence_words)),
            "policy": policy
        }

    def adapt_response(
        self,
        user_input: str,
        retrieved_response: str,
        similarity_score: float,
        state_assessment: Dict[str, Any]
    ) -> str:
        """
        Dynamically adapts the retrieved response based on the patient's condition.
        """
        condition = state_assessment["user_condition"]
        policy = state_assessment["policy"]

        # Clean raw response
        cleaned_response = retrieved_response.strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_response) if s.strip()]

        # Condition 1: ACUTE_DISTRESS -> Never dump a long essay! Keep it gentle, max 2 short soothing sentences
        if condition == "ACUTE_DISTRESS":
            selected_body = sentences[0] if sentences else "Bernapaslah perlahan, kamu tidak sendirian."
            adapted_text = (
                f"{policy['framing_prefix']}\n\n"
                f"{selected_body}\n\n"
                f"🌱 Langkah saat ini: {policy['action_directive']}"
            )
            return adapted_text

        # Condition 2: FRUSTRATED_RESISTANT -> Validate cynicism, avoid preachy tone
        elif condition == "FRUSTRATED_RESISTANT":
            # Pick insight sentence, filter out preachy commands
            body = " ".join(sentences[:2]) if len(sentences) >= 2 else cleaned_response
            adapted_text = (
                f"{policy['framing_prefix']}\n\n"
                f"{body}\n\n"
                f"🕊️ Ingat: {policy['action_directive']}"
            )
            return adapted_text

        # Condition 3: RUMINATIVE_OVERTHINKING -> Break the mental loop with reframing
        elif condition == "RUMINATIVE_OVERTHINKING":
            body = " ".join(sentences[:2]) if len(sentences) >= 2 else cleaned_response
            adapted_text = (
                f"{policy['framing_prefix']}\n\n"
                f"{body}\n\n"
                f"🎯 Tindakan konkret: {policy['action_directive']}"
            )
            return adapted_text

        # Condition 4: LONELY_EMPTY -> Warm companionship
        elif condition == "LONELY_EMPTY":
            body = " ".join(sentences[:2]) if len(sentences) >= 2 else cleaned_response
            adapted_text = (
                f"{policy['framing_prefix']}\n\n"
                f"{body}\n\n"
                f"🤍 Pendampingan: {policy['action_directive']}"
            )
            return adapted_text

        # Condition 5: SEEKING_PSYCHOEDUCATION / Default -> Structured 3-Tier Advice
        else:
            body = " ".join(sentences[:3]) if len(sentences) >= 3 else cleaned_response
            adapted_text = (
                f"{policy['framing_prefix']}\n\n"
                f"{body}\n\n"
                f"💡 Rekomendasi: {policy['action_directive']}"
            )
            return adapted_text
