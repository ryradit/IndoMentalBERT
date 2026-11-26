"""
Indonesian Colloquial & Slang Normalizer for Clinical NLP.
Normalizes informal abbreviations, slang (bahasa gaul), and mental health loanwords
into standard Indonesian tokens to improve BERT semantic encoding.
"""

import re

# Dictionary of informal words, abbreviations, and loanwords to formal/clinical Indonesian
RAW_SLANG_MAP = {
    # Multi-word collocations first (to prevent partial replacement)
    r"\bgak bisa\b|\bga bisa\b|\bga bs\b|\bgabisa\b": "tidak bisa",
    r"\bgak mau\b|\bga mau\b|\bgamau\b": "tidak mau",
    r"\bgak tahu\b|\bga tahu\b|\bgatau\b|\bgtw\b": "tidak tahu",
    r"\bgak papa\b|\bgak apa apa\b|\bgapapa\b|\bgp2\b|\bgpp\b": "tidak apa-apa",
    r"\bga kuat\b|\bgak kuat\b": "tidak sanggup",
    r"\bpanic attack\b": "serangan panik",
    r"\bmood swing\b": "perubahan suasana hati",
    r"\bmental breakdown\b": "krisis emosional",
    r"\bstres bgt\b": "sangat stres",
    r"\bsering bgt\b": "sering sekali",
    
    # Pronouns
    r"\bgw\b|\bgue\b|\bgua\b": "saya",
    r"\blu\b|\bloe\b|\blo\b": "kamu",
    r"\bdoi\b": "dia",
    
    # Negations & Common words
    r"\bga\b|\bgak\b|\bngga\b|\bnggak\b|\bnda\b|\bg\b": "tidak",
    r"\bbkn\b": "bukan",
    r"\budh\b|\buda\b|\bsdh\b": "sudah",
    r"\bblm\b|\bblom\b": "belum",
    r"\bjg\b": "juga",
    r"\byg\b": "yang",
    r"\bdgn\b": "dengan",
    r"\butk\b": "untuk",
    r"\bsm\b": "sama",
    r"\bdr\b": "dari",
    r"\bkrn\b|\bkarna\b": "karena",
    r"\btp\b|\btpi\b": "tetapi",
    r"\bklo\b|\bkalo\b": "kalau",
    r"\bjdi\b": "jadi",
    r"\baja\b|\bwae\b": "saja",
    
    # Temporal & Intensifiers
    r"\bbgt\b|\bbgt2\b|\bbanget\b": "sekali",
    r"\bskrg\b|\bskrg2\b": "sekarang",
    r"\bkmrn\b|\bkemaren\b": "kemarin",
    r"\btrs\b|\btrus\b": "terus",
    r"\blg\b|\blgi\b": "sedang",
    
    # Actions & Mental State Collocations
    r"\bngerasa\b|\bngrasa\b": "merasa",
    r"\bmikir\b|\bmikirin\b": "memikirkan",
    r"\btdr\b": "tidur",
    r"\bmkn\b": "makan",
    r"\bkrj\b": "kerja",
    r"\bcape\b|\bcapek\b|\bpegel\b": "lelah",
    r"\boverthinking\b": "berpikir berlebihan",
    r"\bburnout\b": "kelelahan kerja",
    r"\banxiety\b": "kecemasan",
    r"\bdown\b": "terpuruk",
    r"\binsecure\b": "tidak percaya diri",
    r"\buseless\b": "tidak berguna",
    r"\bworthless\b": "tidak berharga",
    r"\bhopeless\b": "putus asa",
    r"\btrigger\b|\bketrigger\b": "terpicu",
    r"\bhealing\b": "pemulihan diri",
    r"\bnyerah\b": "menyerah"
}


def normalize_slang(text: str) -> str:
    """Normalize informal Indonesian text to standard grammatical forms."""
    if not isinstance(text, str):
        return ""
    normalized = text.lower()
    
    # Multi-character repetitive reduction (e.g., "capekkkk" -> "capek")
    normalized = re.sub(r"(.)\1{2,}", r"\1", normalized)
    
    for pattern, replacement in RAW_SLANG_MAP.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    
    # Clean multiple spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
