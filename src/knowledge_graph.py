"""
Indonesian Psychiatric Clinical Knowledge Graph (Ontology).

Maps:
Somatic / Psychological Symptoms -> Clinical Disorders -> Cognitive Distortions -> Evidence-Based Clinical Interventions
Based on PDSKJI (Perhimpunan Dokter Spesialis Kedokteran Jiwa Indonesia), DSM-5, and Kemenkes RI Clinical Guidelines.
"""

import re
from typing import Dict, List, Any, Optional

PSYCHIATRIC_ONTOLOGY = {
    "anxiety_panic": {
        "disorder_name": "Gangguan Kecemasan & Panik (Anxiety & Panic Disorder)",
        "symptoms": [
            "sesak napas", "dada sesak", "jantung berdebar", "deg-degan", "palpitasi",
            "asam lambung", "gerd", "mual", "pusing melayang", "tangan gemetar",
            "keringat dingin", "takut mati", "gelisah", "overthinking", "takut gila",
            "khawatir berlebih", "pikiran buruk", "tidak tenang"
        ],
        "cognitive_distortions": [
            "Catastrophizing (Membesar-besarkan kemungkinan terburuk)",
            "Emotional Reasoning (Menganggap perasaan cemas sebagai bukti bahaya nyata)"
        ],
        "default_risk_level": "Moderate",
        "clinical_intervention": {
            "technique": "Grounding 5-4-3-2-1 & Box Breathing (Teknik Pernapasan Somatik)",
            "clinical_rationale": "Menurunkan stimulasi sistem saraf simpatik (fight-or-flight) dan mengaktifkan saraf vagus untuk relaksasi otot dada dan lambung.",
            "protocol": "Tarik napas 4 detik, tahan 4 detik, hembuskan perlahan 4 detik. Amati 5 benda di sekitar yang bisa kamu lihat."
        }
    },
    "depression_major": {
        "disorder_name": "Depresi / Gangguan Afektif (Major Depressive Symptoms)",
        "symptoms": [
            "hampa", "kosong", "hilang minat", "anhedonia", "menangis terus",
            "lelah tak beralasan", "capek batin", "tidak berharga", "merasa gagal",
            "beban keluarga", "rasa bersalah", "putus asa", "malas mandi",
            "sulit bangun tidur", "isolasi diri", "tidak ada masa depan"
        ],
        "cognitive_distortions": [
            "All-or-Nothing Thinking (Pola pikir hitam-putih)",
            "Personalization (Menyalahkan diri sendiri secara ekstrem)",
            "Mental Filter (Hanya fokus pada kegagalan)"
        ],
        "default_risk_level": "Severe",
        "clinical_intervention": {
            "technique": "Behavioral Activation & Micro-Steps (Aktivasi Perilaku Bertahap)",
            "clinical_rationale": "Memecah siklus pasivitas depresi dengan tindakan mikroskopik yang tidak memerlukan motivasi tinggi.",
            "protocol": "Jangan tuntut dirimu melakukan hal besar hari ini. Cukup minum satu gelas air, atau duduk di tepi kasur selama 2 menit."
        }
    },
    "burnout_chronic_stress": {
        "disorder_name": "Kelelahan Mental Kronis & Burnout Akademik/Kerja",
        "symptoms": [
            "burnout", "muak kerja", "kuliah berat", "tugas menumpuk", "tidak fokus",
            "otak buntu", "brain fog", "insomnia karena mikir kerjaan", "sinis",
            "kehilangan motivasi", "lelah fisik mental", "ingin kabur", "capek banget"
        ],
        "cognitive_distortions": [
            "Should Statements (Tuntutan 'harus sempurna' tanpa jeda)",
            "Discounting the Positive (Merasa pencapaian diri belum cukup)"
        ],
        "default_risk_level": "Mild",
        "clinical_intervention": {
            "technique": "Pacing & Boundary Setting (Penetapan Batasan Energi)",
            "clinical_rationale": "Mencegah kolaps sistemik kognitif akibat paparan stresor kerja/tugas berkepanjangan.",
            "protocol": "Berikan izin pada dirimu untuk 'clock out' secara mental. Terapkan prinsip Pomodoro jeda 5 menit tiap 25 menit kerja."
        }
    },
    "ocd_intrusive_thoughts": {
        "disorder_name": "Pikiran Obsesif & Distres Intrusif (Intrusive Thoughts)",
        "symptoms": [
            "pikiran aneh terus muncul", "takut tertular", "mengulang cuci tangan",
            "cek pintu berulang", "pikiran kotor tiba-tiba", "takut mencelakai orang",
            "pikiran mengganggu", "ragu berlebihan"
        ],
        "cognitive_distortions": [
            "Thought-Action Fusion (Mengira memikirkan hal buruk sama dengan melakukannya)",
            "Overestimation of Threat (Melebih-lebihkan peluang bencana)"
        ],
        "default_risk_level": "Moderate",
        "clinical_intervention": {
            "technique": "Defusion & Non-Engagement (Pemisahan Pikiran)",
            "clinical_rationale": "Menghentikan kompulsi netralisasi dan melatih toleransi terhadap ketidakpastian.",
            "protocol": "Akui keberadaan pikiran tersebut: 'Pikiran ini hanyalah gelombang listrik di otakku, bukan niat atau karakter asliku'."
        }
    },
    "trauma_grief": {
        "disorder_name": "Reaksi Duka & Trauma Psikologis (Grief & Trauma Response)",
        "symptoms": [
            "ditinggal pergi", "patah hati parah", "kehilangan orang tua",
            "kehilangan pasangan", "trauma masa lalu", "teringat kejadian buruk",
            "flashback", "mimpi buruk kejadian itu", "hancur lebur", "kematian"
        ],
        "cognitive_distortions": [
            "Permanentizing (Menganggap rasa sakit ini tidak akan pernah selesai)",
            "Hindsight Bias (Menyesali masa lalu seolah-olah bisa dicegah)"
        ],
        "default_risk_level": "Severe",
        "clinical_intervention": {
            "technique": "Compassionate Holding & Grief Normalization (Validasi Kedukaan)",
            "clinical_rationale": "Menyediakan wadah emosional aman bagi proses pemulihan tanpa pemaksaan 'toxic positivity'.",
            "protocol": "Berduka adalah harga dari cinta yang pernah ada. Rasakan air matamu tanpa perlu merasa lemah."
        }
    }
}

RISK_LEVELS = ["Low", "Mild", "Moderate", "Severe", "Emergency"]

CRISIS_PATTERNS = [
    r"\bbunuh diri\b", r"\bmati saja\b", r"\bmengakhiri hidup\b",
    r"\bmenyakiti diri\b", r"\bgantung diri\b", r"\btelan racun\b",
    r"\bpotong urat\b", r"\bingin mati\b", r"\btidak ingin hidup\b"
]


class ClinicalKnowledgeGraph:
    """
    Knowledge Graph reasoning engine for Indonesian mental health symptoms,
    cognitive distortions, and guideline-backed intervention anchoring.
    """

    def __init__(self):
        self.ontology = PSYCHIATRIC_ONTOLOGY

    def reason(self, text: str) -> Dict[str, Any]:
        """
        Extract symptoms, predict clinical category, identify cognitive distortions,
        and supply targeted intervention guidance.
        """
        text_lower = text.lower()

        # 1. Emergency Risk Override
        for cp in CRISIS_PATTERNS:
            if re.search(cp, text_lower):
                return {
                    "matched": True,
                    "risk_level": "Emergency",
                    "disorder": "Krisis Darurat (Emergency Mental Crisis)",
                    "symptoms_found": ["suicidal_ideation_or_self_harm"],
                    "cognitive_distortions": ["Hopelessness / Tunnel Vision"],
                    "clinical_intervention": {
                        "technique": "Protokol Darurat & Hotline Segera",
                        "clinical_rationale": "Penyelamatan nyawa primer tanpa penundaan.",
                        "protocol": "Segera hubungi Hotline LISA 119 ext 8 atau IGD terdekat."
                    }
                }

        # 2. Symptom traversal across Psychiatric Ontology
        matched_category = None
        best_count = 0
        detected_symptoms = []

        for cat_key, cat_data in self.ontology.items():
            found_in_cat = []
            for symptom in cat_data["symptoms"]:
                if symptom in text_lower:
                    found_in_cat.append(symptom)
            
            if len(found_in_cat) > best_count:
                best_count = len(found_in_cat)
                matched_category = cat_key
                detected_symptoms = found_in_cat

        if matched_category and best_count > 0:
            cat_data = self.ontology[matched_category]
            # Risk scaling by symptom density
            risk = cat_data["default_risk_level"]
            if best_count >= 3 and risk == "Moderate":
                risk = "Severe"

            return {
                "matched": True,
                "category_key": matched_category,
                "risk_level": risk,
                "disorder": cat_data["disorder_name"],
                "symptoms_found": detected_symptoms,
                "cognitive_distortions": cat_data["cognitive_distortions"],
                "clinical_intervention": cat_data["clinical_intervention"]
            }

        # 3. Default benign/mild triaging
        return {
            "matched": False,
            "category_key": "general_support",
            "risk_level": "Mild",
            "disorder": "Keluhan Umum & Suportif (General Supportive)",
            "symptoms_found": [],
            "cognitive_distortions": ["Unspecified"],
            "clinical_intervention": {
                "technique": "Active Listening & Supportive Reflection",
                "clinical_rationale": "Memberikan ruang aman berventilasi.",
                "protocol": "Eksplorasi emosi secara perlahan dan teratur."
            }
        }
