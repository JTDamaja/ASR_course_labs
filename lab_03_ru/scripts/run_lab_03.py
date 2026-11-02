#!/usr/bin/env python3
"""Запустить детерминированную демонстрацию выравнивания и биграммной ЯМ."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import sys

LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from scripts.bigram_lm import BigramLM
from scripts.combine_candidate_scores import rank_candidates
from scripts.viterbi_alignment import viterbi_align


def render_trellis_svg(
    costs: list[list[float]],
    state_labels: list[str],
    frame_labels: list[str],
    path: list[int],
    output_path: Path,
) -> None:
    """Записать SVG-карту стоимостей с наложенным лучшим путём."""
    width, height = 900, 430
    left, top, cell_w, cell_h = 190, 100, 105, 48
    rows, cols = len(state_labels), len(frame_labels)
    maximum = max(max(row) for row in costs)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#07111f"/>',
        '<text x="450" y="38" fill="#edf6ff" font-family="Arial,sans-serif" font-size="23" font-weight="bold" text-anchor="middle">Учебное выравнивание Витерби: «yes»</text>',
        '<text x="450" y="65" fill="#bad0e4" font-family="Arial,sans-serif" font-size="14" text-anchor="middle">Синтетические стоимости эмиссии: меньше — лучше</text>',
    ]
    for col, label in enumerate(frame_labels):
        x = left + col * cell_w + cell_w / 2
        parts.append(f'<text x="{x}" y="88" fill="#67e8f9" font-family="Arial,sans-serif" font-size="14" text-anchor="middle">{label}</text>')
    for row, label in enumerate(state_labels):
        y = top + row * cell_h
        parts.append(f'<text x="{left-14}" y="{y+29}" fill="#edf6ff" font-family="Arial,sans-serif" font-size="14" text-anchor="end">{label}</text>')
        for col in range(cols):
            value = costs[col][row]
            ratio = min(1.0, value / maximum) if maximum else 0.0
            # Более тёмный синий означает меньшую стоимость и лучшее соответствие.
            red = int(24 + 35 * ratio)
            green = int(78 - 38 * ratio)
            blue = int(105 - 35 * ratio)
            fill = f"#{red:02x}{green:02x}{blue:02x}"
            parts.append(f'<rect x="{left+col*cell_w}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" rx="4" fill="{fill}" stroke="#315d7b"/>')
            parts.append(f'<text x="{left+col*cell_w+8}" y="{y+41}" fill="#edf6ff" font-family="monospace" font-size="11" text-anchor="start">{value:.1f}</text>')
    points = [f"{left + t*cell_w + (cell_w-3)/2},{top + state*cell_h + (cell_h-3)/2}" for t, state in enumerate(path)]
    parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#bef264" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
    for point in points:
        x, y = point.split(",")
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#bef264" stroke="#07111f" stroke-width="2"/>')
    parts.append(f'<text x="450" y="{top+rows*cell_h+28}" fill="#bef264" font-family="Arial,sans-serif" font-size="14" text-anchor="middle">Лучший допустимый путь · остаться или перейти на одно состояние</text>')
    parts.append('</svg>')
    output_path.write_text("".join(parts), encoding="utf-8")


def run_lab(data_dir: Path, output_dir: Path, *, weights: tuple[float, ...] = (0.0, 0.5, 1.0)) -> dict[str, Any]:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    alignment = json.loads((data_dir / "toy_alignment_costs.json").read_text(encoding="utf-8"))
    path, total_cost = viterbi_align(alignment["costs"], start_state=0, final_state=len(alignment["state_labels"])-1)

    corpus = (data_dir / "toy_lm_corpus.txt").read_text(encoding="utf-8").splitlines()
    lm = BigramLM.train(corpus, k=0.5)
    candidates = json.loads((data_dir / "toy_candidate_scores.json").read_text(encoding="utf-8"))
    for candidate in candidates:
        candidate["lm_log_score"] = lm.sentence_logprob(candidate["transcript"])

    output_dir.mkdir(parents=True, exist_ok=True)
    render_trellis_svg(alignment["costs"], alignment["state_labels"], alignment["frame_labels"], path, output_dir / "viterbi_trellis.svg")

    with (output_dir / "alignment_path.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "state_index", "state_label", "emission_cost", "path_cost_so_far"])
        cumulative = 0.0
        for t, state in enumerate(path):
            emission = float(alignment["costs"][t][state])
            cumulative += emission
            writer.writerow([alignment["frame_labels"][t], state, alignment["state_labels"][state], f"{emission:.3f}", f"{cumulative:.3f}"])

    with (output_dir / "candidate_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lm_weight", "rank", "candidate_id", "transcript", "acoustic_log_score", "lm_log_score", "combined_log_score"])
        for weight in weights:
            for rank, row in enumerate(rank_candidates(candidates, weight=weight), start=1):
                writer.writerow([f"{weight:.2f}", rank, row["candidate_id"], row["transcript"], f"{row['acoustic_log_score']:.6f}", f"{row['lm_log_score']:.6f}", f"{row['combined_log_score']:.6f}"])

    summary = {
        "alignment_path": path,
        "alignment_total_cost": total_cost,
        "alternative_path_cost": alignment["known_alternative"]["total_cost"],
        "candidate_rankings": {
            str(weight): [row["candidate_id"] for row in rank_candidates(candidates, weight=weight)]
            for weight in weights
        },
        "note": "Стоимость и акустические баллы синтетические: это учебный пример, а не вывод модели.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    try:
        summary = run_lab(args.data_dir, args.output_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Стоимость лучшего выравнивания: {summary['alignment_total_cost']:.3f}")
    print(f"Результаты записаны в: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
