from __future__ import annotations
from typing import Iterable, Tuple
from difflib import SequenceMatcher


def levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = [list(range(lb + 1))] + [[i] + [0] * lb for i in range(1, la + 1)]
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[la][lb]


# def best_fuzzy_match(query: str, candidates: Iterable[str]) -> Tuple[str, float]:
#     best = ("", 0.0)
#     for c in candidates:
#         ratio = SequenceMatcher(None, query, c).ratio()
#         if ratio > best[1]:
#             best = (c, ratio)
#     return best
