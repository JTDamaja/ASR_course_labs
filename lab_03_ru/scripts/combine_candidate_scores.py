from __future__ import annotations

import math
from typing import Any, Sequence


def rank_candidates(
    candidates: Sequence[dict[str, Any]], *, weight: float = 1.0
) -> list[dict[str, Any]]:
    """Ранжировать кандидатов по акустическому и взвешенному LM лог-баллам."""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("вес LM должен быть конечным неотрицательным числом")
    ranked = []
    for candidate in candidates:
        try:
            acoustic = float(candidate["acoustic_log_score"])
            language = float(candidate["lm_log_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("для каждого кандидата нужны числовые acoustic_log_score и lm_log_score") from exc
        if not math.isfinite(acoustic) or not math.isfinite(language):
            raise ValueError("баллы кандидатов должны быть конечными")
        row = dict(candidate)
        row["combined_log_score"] = # напишите ваш код здесь
        ranked.append(row)
    return sorted(ranked, key=lambda row: row["combined_log_score"], reverse=True)
