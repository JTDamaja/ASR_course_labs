#!/usr/bin/env python3
"""Строит сравнения сигналов/log-Mel и SpecAugment только для признаков в Lab 02."""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def log_mel(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=400, hop_length=160, win_length=400, n_mels=80, fmin=20, fmax=7600)
    return librosa.power_to_db(mel, ref=np.max)


def specaugment(matrix: np.ndarray, time_width: int = 24, frequency_width: int = 10) -> tuple[np.ndarray, dict[str, int]]:
    result = matrix.copy()
    time_start = max(0, matrix.shape[1] // 2 - time_width // 2)
    frequency_start = max(0, matrix.shape[0] // 2 - frequency_width // 2)
    result[:, time_start:time_start + time_width] = matrix.min()
    result[frequency_start:frequency_start + frequency_width, :] = matrix.min()
    return result, {"time_start": time_start, "time_width": time_width, "frequency_start": frequency_start, "frequency_width": frequency_width}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    clean, sr = librosa.load(args.clean, sr=16_000, mono=True)
    augmented, _ = librosa.load(args.augmented, sr=16_000, mono=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, (label, audio) in enumerate((("clean", clean), ("augmented", augmented))):
        librosa.display.waveshow(audio, sr=sr, ax=axes[row, 0]); axes[row, 0].set_title(f"{label} waveform")
        image = librosa.display.specshow(log_mel(audio, sr), sr=sr, hop_length=160, x_axis="time", y_axis="mel", ax=axes[row, 1], cmap="viridis")
        axes[row, 1].set_title(f"{label} log-Mel"); fig.colorbar(image, ax=axes[row, 1], format="%+2.0f dB")
    fig.tight_layout(); fig.savefig(args.out_dir / "clean_vs_augmented.png", dpi=160); plt.close(fig)
    clean_mel = log_mel(clean, sr); masked, metadata = specaugment(clean_mel)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for axis, matrix, title in zip(axes, (clean_mel, masked), ("Clean log-Mel", "Feature-only SpecAugment (not decoded)")):
        image = librosa.display.specshow(matrix, sr=sr, hop_length=160, x_axis="time", y_axis="mel", ax=axis, cmap="viridis"); axis.set_title(title); fig.colorbar(image, ax=axis, format="%+2.0f dB")
    fig.tight_layout(); fig.savefig(args.out_dir / "specaugment_feature_only.png", dpi=160); plt.close(fig)
    print(f"Wrote figures to {args.out_dir}; SpecAugment metadata: {metadata}")


if __name__ == "__main__":
    main()
