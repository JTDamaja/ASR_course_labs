#!/usr/bin/env python3
"""Запускает кэшированный faster-whisper локально для вариантов сигналов Lab 02."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf


def transcribe(audio_path: Path, model_dir: str | None = None, allow_model_download: bool = False) -> tuple[str, float]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper from requirements.txt. This lab never sends audio to a service.") from exc
    try:
        model = WhisperModel(model_dir or "tiny.en", device="cpu", compute_type="int8", local_files_only=not allow_model_download)
    except Exception as exc:
        raise RuntimeError("Could not load cached tiny.en. Run the documented one-time model setup, then pass --model-dir to the cache/model directory for offline runs.") from exc
    start = time.perf_counter()
    segments, _ = model.transcribe(str(audio_path), language="en", beam_size=1, vad_filter=False)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--allow-model-download", action="store_true", help="Explicit one-time setup only; normal runs use the local cache.")
    args = parser.parse_args()
    references = {row["item_id"]: row for row in (json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line)}
    result_rows = []
    for condition in (json.loads(line) for line in args.conditions.read_text(encoding="utf-8").splitlines() if line):
        hypothesis, seconds = transcribe(Path(condition["audio_path"]), args.model_dir, args.allow_model_download)
        info = sf.info(condition["audio_path"])
        result_rows.append({**condition, "reference_raw": references[condition["item_id"]]["reference_raw"], "hypothesis_raw": hypothesis, "audio_seconds": info.frames / info.samplerate, "decode_seconds": seconds})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in result_rows), encoding="utf-8")
    print(f"Decoded {len(result_rows)} local waveform conditions to {args.out}")


if __name__ == "__main__":
    main()
