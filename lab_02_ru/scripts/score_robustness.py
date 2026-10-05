#!/usr/bin/env python3
"""Использует единый локальный нормализатор текста для оценки гипотез Lab 02."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

RESULT_FIELDS = ["item_id", "condition", "parameters", "seed", "reference_raw", "reference_normalized", "hypothesis_raw", "hypothesis_normalized", "wer", "cer", "audio_seconds", "decode_seconds", "rtf"]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9']+", " ", text.lower())).strip()


def _distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def score_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scored = []
    for input_row in rows:
        row = dict(input_row)
        reference = normalize_text(str(row["reference_raw"]))
        hypothesis = normalize_text(str(row.get("hypothesis_raw", "")))
        ref_words, hyp_words = reference.split(), hypothesis.split()
        row["reference_normalized"] = reference
        row["hypothesis_normalized"] = hypothesis
        row["wer"] = round(_distance(ref_words, hyp_words) / max(len(ref_words), 1), 6)
        row["cer"] = round(_distance(list(reference), list(hypothesis)) / max(len(reference), 1), 6)
        row["audio_seconds"] = round(float(row["audio_seconds"]), 6)
        row["decode_seconds"] = round(float(row["decode_seconds"]), 6)
        row["rtf"] = round(row["decode_seconds"] / max(row["audio_seconds"], 1e-12), 6)
        scored.append({field: row.get(field, "") for field in RESULT_FIELDS})
    return scored


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["condition"])].append(row)
    aggregates = []
    for condition, group in sorted(groups.items()):
        reference_words = sum(len(str(row["reference_normalized"]).split()) for row in group)
        total_decode_seconds = sum(float(row["decode_seconds"]) for row in group)
        total_audio_seconds = sum(float(row["audio_seconds"]) for row in group)
        aggregates.append({"condition": condition, "utterance_count": len(group), "reference_word_count": reference_words, "mean_wer": round(sum(float(row["wer"]) for row in group) / len(group), 6), "mean_cer": round(sum(float(row["cer"]) for row in group) / len(group), 6), "mean_rtf": round(total_decode_seconds / max(total_audio_seconds, 1e-12), 6)})
    return aggregates


def write_results_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="ASR JSONL rows")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    source = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    scored = score_rows(source)
    write_results_csv(scored, args.out_dir / "per_item_results.csv")
    aggregate_path = args.out_dir / "aggregate_results.csv"
    aggregate = aggregate_rows(scored)
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "utterance_count", "reference_word_count", "mean_wer", "mean_cer", "mean_rtf"])
        writer.writeheader(); writer.writerows(aggregate)
    print(f"Scored {len(scored)} rows with the shared normalizer")


if __name__ == "__main__":
    main()
