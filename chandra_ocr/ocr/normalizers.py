
from __future__ import annotations
import re

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def normalize_upper(s: str) -> str:
    return normalize_whitespace(s).upper()

def normalize_msp_code(s: str) -> str:
    s = normalize_upper(s)
    s = s.replace('O', '0')
    return s
