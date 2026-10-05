#!/usr/bin/env python3
"""Проверяет фиксированный включённый в комплект набор Lab 02 на основе LibriSpeech.
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

REQUIRED_MANIFEST_FIELDS = {"item_id", "audio_path", "reference_raw", "reference_normalized", "duration_seconds", "source_utterance_id", "seed"}


def validate_manifest(manifest_path: Path, lab_dir: Path | None = None) -> list[dict[str, object]]:
    lab_dir = (lab_dir or manifest_path.parents[1]).resolve()
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 4 <= len(rows) <= 6:
        raise ValueError("The Lab 02 manifest must contain 4–6 utterances")
    for row in rows:
        missing = REQUIRED_MANIFEST_FIELDS - set(row)
        if missing:
            raise ValueError(f"Manifest row {row.get('item_id', '?')} missing {sorted(missing)}")
        audio_path = (lab_dir / str(row["audio_path"])).resolve()
        try:
            audio_path.relative_to(lab_dir)
        except ValueError as exc:
            raise ValueError(f"Audio path must stay within Lab 02: {row['audio_path']}") from exc
        if not audio_path.is_file():
            raise ValueError(f"Missing manifest audio: {audio_path}")
        with wave.open(str(audio_path), "rb") as wav:
            if wav.getframerate() != 16_000 or wav.getnchannels() != 1:
                raise ValueError(f"Expected 16 kHz mono WAV: {audio_path}")
            measured = wav.getnframes() / wav.getframerate()
        if abs(float(row["duration_seconds"]) - measured) > 0.02:
            raise ValueError(f"Duration mismatch for {row['item_id']}: manifest versus WAV")
        if not str(row["reference_raw"]).strip() or not str(row["reference_normalized"]).strip():
            raise ValueError(f"Empty transcript for {row['item_id']}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "manifest.jsonl")
    args = parser.parse_args()
    rows = validate_manifest(args.manifest)
    print(f"Validated {len(rows)} fixed local utterances in {args.manifest}")


if __name__ == "__main__":
    main()
