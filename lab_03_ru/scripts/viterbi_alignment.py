from __future__ import annotations

import math
from typing import Sequence


def viterbi_align(
    costs: Sequence[Sequence[float]],
    *,
    start_state: int = 0,
    final_state: int | None = None,
) -> tuple[list[int], float]:
    """Найти монотонный путь с минимальной стоимостью (остаться/перейти).

    При равенстве стоимостей предпочтительно остаться в текущем состоянии.
    На первом кадре достижимо только start_state, на последнем нужен final_state.
    """
    if not costs or not costs[0]:
        raise ValueError("матрица стоимостей должна содержать хотя бы один кадр и состояние")
    state_count = len(costs[0])
    if any(len(row) != state_count for row in costs):
        raise ValueError("матрица стоимостей должна быть прямоугольной")
    if any(not math.isfinite(float(value)) for row in costs for value in row):
        raise ValueError("значения матрицы стоимостей должны быть конечными")
    if final_state is None:
        final_state = state_count - 1
    if not (0 <= start_state < state_count and 0 <= final_state < state_count):
        raise ValueError("start_state и final_state должны быть допустимыми индексами состояний")
    if final_state < start_state:
        raise ValueError("final_state не должно предшествовать start_state")
    if len(costs) < final_state - start_state + 1:
        raise ValueError("недостаточно кадров для перехода по всем состояниям")

    frame_count = len(costs)
    dp = [[math.inf] * state_count for _ in range(frame_count)]
    back = [[None] * state_count for _ in range(frame_count)]
    dp[0][start_state] = float(costs[0][start_state])

    for t in range(1, frame_count):
        for state in range(start_state, final_state + 1):
            # При равенстве min() выбирает первый вариант: остаться в состоянии.
            choices = # напишите ваш код здесь
            previous_state, previous_cost = # напишите ваш код здесь
            if math.isfinite(previous_cost):
                dp[t][state] = # напишите ваш код здесь
                back[t][state] = # напишите ваш код здесь

    total_cost = dp[-1][final_state]
    if not math.isfinite(total_cost):
        raise ValueError("нет допустимого пути до final_state на последнем кадре")
    path = [final_state]
    for t in range(frame_count - 1, 0, -1):
        previous_state = back[t][path[-1]]
        if previous_state is None:
            raise RuntimeError("нарушена цепочка обратных указателей Витерби")
        path.append(previous_state)
    path.reverse()
    return path, total_cost
