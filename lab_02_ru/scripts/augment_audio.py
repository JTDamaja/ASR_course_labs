#!/usr/bin/env python3
"""Создаёт детерминированные аугментации сигналов для Lab 02."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

DEFAULT_SEED = 17
SAMPLE_RATE = 16_000


def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
    return (audio if maximum == 0 else audio / maximum * peak).astype(np.float32)


def add_noise_at_snr(audio: np.ndarray, snr_db: float, seed: int = DEFAULT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, len(audio)).astype(np.float32)
    speech_power = float(np.mean(np.square(audio)))
    noise_power = float(np.mean(np.square(noise)))
    scale = np.sqrt((speech_power / (10 ** (snr_db / 10))) / max(noise_power, 1e-12))
    return peak_normalize(audio + noise * scale)


def make_synthetic_rir(sample_rate: int, variant: str = "room_a") -> np.ndarray:
    if variant not in {"room_a", "room_b"}:
        raise ValueError(f"Unknown RIR variant: {variant}")
    duration, decay, reflections = (0.38, 10.0, [(0.031, 0.48), (0.079, 0.30), (0.143, 0.16)]) if variant == "room_a" else (0.55, 7.0, [(0.049, 0.56), (0.121, 0.34), (0.213, 0.20)])
    time = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    rir = np.exp(-decay * time).astype(np.float32)
    rir[0] = 1.0
    for delay, gain in reflections:
        rir[int(delay * sample_rate)] += gain
    return (rir / np.sqrt(np.sum(rir**2))).astype(np.float32)


def reverberate(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    return peak_normalize(signal.fftconvolve(audio, rir, mode="full")[: len(audio)])


def apply_speed_perturbation(audio: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0:
        raise ValueError("Speed factor must be positive")
    return peak_normalize(signal.resample_poly(audio, 100, int(round(100 * factor))))


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate}: {path}")
    return audio.mean(axis=1), sample_rate


def write_waveform_conditions(items: list[dict[str, object]], out_dir: Path, seed: int = DEFAULT_SEED, include_alternate_rir: bool = False) -> list[dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = [("clean", {}), ("noise_20db", {"snr_db": 20.0}), ("noise_10db", {"snr_db": 10.0}), ("noise_0db", {"snr_db": 0.0}), ("reverb_room_a", {"rir": "room_a"}), ("speed_0.9x", {"speed_factor": 0.9}), ("speed_1.1x", {"speed_factor": 1.1})]
    if include_alternate_rir:
        conditions.append(("reverb_room_b", {"rir": "room_b"}))
    output: list[dict[str, object]] = []
    for item_index, item in enumerate(items):
        audio, sample_rate = _read_mono(Path(str(item["audio_path"])))
        for condition, parameters in conditions:
            if condition == "clean":
                rendered = audio
            elif condition.startswith("noise"):
                rendered = add_noise_at_snr(audio, float(parameters["snr_db"]), seed + item_index)
            elif condition.startswith("reverb"):
                rendered = reverberate(audio, make_synthetic_rir(sample_rate, str(parameters["rir"])))
            else:
                rendered = apply_speed_perturbation(audio, float(parameters["speed_factor"]))
            path = out_dir / f"{item['item_id']}__{condition}.wav"
            sf.write(path, rendered, sample_rate, subtype="PCM_16")
            output.append({"item_id": item["item_id"], "condition": condition, "parameters": json.dumps(parameters, sort_keys=True), "seed": seed, "audio_path": str(path.resolve())})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alternate-rir", action="store_true")
    args = parser.parse_args()
    items = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    lab_dir = args.manifest.parents[1]
    for item in items:
        item["audio_path"] = str((lab_dir / str(item["audio_path"])).resolve())
    rows = write_waveform_conditions(items, args.out_dir, args.seed, args.alternate_rir)
    (args.out_dir / "conditions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} deterministic waveform conditions to {args.out_dir}")


if __name__ == "__main__":
    main()
