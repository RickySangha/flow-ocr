from __future__ import annotations
import re
from typing import Callable


def regex_validator(pattern: str) -> Callable[[str], bool]:
    rx = re.compile(pattern)

    def _f(s: str) -> bool:
        return bool(rx.fullmatch(s.strip()))

    return _f


def always_true(_: str) -> bool:
    return True


PHN_PATTERN = r"^9[0-9]{9}$"
validate_phn = regex_validator(PHN_PATTERN)

MSP_CODE_PATTERN = r"[0-9]{5}$"
validate_msp_code = regex_validator(MSP_CODE_PATTERN)
