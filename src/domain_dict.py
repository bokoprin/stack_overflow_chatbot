import os
import re


DEFAULT_MAP = {
    "rtos": ["real-time os", "real time os", "real time operating system"],
    "raspberry pi": ["raspi", "ラズパイ"],
    "arduino": ["マイコン", "microcontroller"],
    "c++": ["cplusplus", "cpp"],
    "c言語": ["c language", "c-lang"],
    "linux": ["gnu/linux", "linux kernel"],
    "pytorch": ["torch", "py torch"],
    "tensorflow": ["tf"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "machine learning": ["ml", "機械学習"],
    "deep learning": ["dl", "深層学習"],
}


def normalize_terms(text, mapping=None):
    if not text:
        return text
    if os.getenv("ENABLE_DOMAIN_NORMALIZE", "true").lower() not in {"1", "true", "yes", "on"}:
        return text
    mapping = mapping or DEFAULT_MAP
    normalized = text
    for canonical, variants in mapping.items():
        for variant in variants:
            normalized = _replace(normalized, variant, canonical)
    return normalized


def _replace(text, variant, canonical):
    if not text:
        return text
    pattern = re.escape(variant)
    if re.search(r"[A-Za-z0-9_]", variant):
        return re.sub(rf"\\b{pattern}\\b", canonical, text, flags=re.IGNORECASE)
    return text.replace(variant, canonical)
