#!/usr/bin/env python3
"""Генерирует небольшие автономные аудиофайлы для Лабораторной 01 курса ASR.

Клипы намеренно синтетические и не имеют лицензионных ограничений. Они не
должны звучать как естественная речь; их задача — показать speech-like тайминг,
voiced/unvoiced энергию, паузы, шум, реверберацию, клиппинг и изменения скорости
для упражнений по визуальному анализу.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SAMPLE_RATE = 16_000
RANDOM_SEED = 7


def _normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs == 0.0:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def _raised_cosine_envelope(length: int, fade_samples: int) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float32)
    fade_samples = max(1, min(fade_samples, length // 2))
    fade = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_samples))
    envelope[:fade_samples] = fade
    envelope[-fade_samples:] = fade[::-1]
    return envelope


def synthesize_speech_like(sample_rate: int = SAMPLE_RATE, seed: int = RANDOM_SEED) -> np.ndarray:
    """Создаёт короткий speech-like сигнал с voiced-сегментами и паузами."""

    rng = np.random.default_rng(seed)
    total_duration = 5.2
    total_samples = int(total_duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    # segment_start_s, duration_s, base_f0_hz, vowel_formant_scale
    voiced_segments = [
        (0.18, 0.62, 125.0, 1.00),
        (1.02, 0.70, 155.0, 1.08),
        (2.00, 0.52, 118.0, 0.92),
        (2.82, 0.78, 172.0, 1.15),
        (4.02, 0.68, 138.0, 1.04),
    ]

    for start_s, duration_s, base_f0, scale in voiced_segments:
        start = int(start_s * sample_rate)
        length = int(duration_s * sample_rate)
        t = np.arange(length, dtype=np.float32) / sample_rate
        vibrato = 1.0 + 0.025 * np.sin(2 * np.pi * 4.5 * t)
        phase = 2 * np.pi * base_f0 * vibrato * t
        voiced = (
            0.55 * np.sin(phase)
            + 0.24 * np.sin(2 * phase + 0.2)
            + 0.13 * np.sin(3 * phase + 0.4)
            + 0.07 * np.sin(2 * np.pi * 720 * scale * t)
            + 0.04 * np.sin(2 * np.pi * 1_250 * scale * t)
        )
        voiced *= _raised_cosine_envelope(length, int(0.035 * sample_rate))
        audio[start : start + length] += voiced.astype(np.float32)

        # Добавляем короткий fricative-like всплеск около границы каждого сегмента.
        burst_len = min(int(0.09 * sample_rate), total_samples - start)
        burst = rng.normal(0.0, 0.12, burst_len).astype(np.float32)
        b, a = signal.butter(2, [1_800, 5_500], btype="bandpass", fs=sample_rate)
        burst = signal.lfilter(b, a, burst).astype(np.float32)
        audio[start : start + burst_len] += burst * _raised_cosine_envelope(burst_len, 120)

    # Слабый breath/background floor, чтобы детекция тишины была реалистичной.
    audio += rng.normal(0.0, 0.003, total_samples).astype(np.float32)
    return _normalize(audio, 0.78)


def add_noise(audio: np.ndarray, snr_db: float = 8.0, seed: int = RANDOM_SEED + 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, len(audio)).astype(np.float32)
    speech_power = float(np.mean(audio**2))
    noise_power = float(np.mean(noise**2))
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    noisy = audio + noise * np.sqrt(target_noise_power / max(noise_power, 1e-12))
    return _normalize(noisy, 0.92)


def add_reverb(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    rir_duration = 0.42
    n = int(rir_duration * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    rir = np.exp(-t * 9.0)
    rir[0] = 1.0
    # Add two early reflections to make smearing visible.
    for delay_s, gain in [(0.045, 0.45), (0.095, 0.25)]:
        idx = int(delay_s * sample_rate)
        if idx < n:
            rir[idx] += gain
    rir = rir / np.sqrt(np.sum(rir**2))
    reverberant = signal.fftconvolve(audio, rir, mode="full")[: len(audio)]
    return _normalize(reverberant, 0.88)


def add_clipping(audio: np.ndarray, threshold: float = 0.34) -> np.ndarray:
    amplified = audio * 2.4
    return np.clip(amplified, -threshold, threshold).astype(np.float32)


def speed_perturb(audio: np.ndarray, speed: float = 1.15) -> np.ndarray:
    # speed > 1.0 делает аудио короче. Рациональный ресемплинг сохраняет зависимости лёгкими.
    up = 100
    down = int(round(up * speed))
    perturbed = signal.resample_poly(audio, up, down).astype(np.float32)
    return _normalize(perturbed, 0.78)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sample_rate, subtype="PCM_16")
    return path


def generate_lab_audio(out_dir: Path | str, sample_rate: int = SAMPLE_RATE) -> list[Path]:
    out_dir = Path(out_dir)
    clean = synthesize_speech_like(sample_rate=sample_rate)
    examples = {
        "clean_speech_like.wav": clean,
        "noisy_speech_like.wav": add_noise(clean),
        "reverberant_speech_like.wav": add_reverb(clean, sample_rate=sample_rate),
        "clipped_speech_like.wav": add_clipping(clean),
        "speed_perturbed_speech_like.wav": speed_perturb(clean),
    }
    return [write_wav(out_dir / name, audio, sample_rate) for name, audio in examples.items()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерирует WAV-файлы для Лабораторной 01.")
    parser.add_argument("--out-dir", type=Path, default=Path("updated_course/lab_01_ru/data/wav"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = generate_lab_audio(args.out_dir)
    for wav_path in written:
        audio, sample_rate = sf.read(wav_path)
        print(f"записано {wav_path} | sr={sample_rate} | длительность={len(audio)/sample_rate:.2f}с | пик={np.max(np.abs(audio)):.3f}")


if __name__ == "__main__":
    main()
