#!/usr/bin/env python3
"""Анализирует WAV-файлы лабораторной 01 и строит графики для проверки аудио.

Скрипт намеренно написан читаемо: студенты могут копировать отдельные функции
в notebook и экспериментировать с параметрами.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 400          # 25 мс при 16 кГц
HOP_LENGTH = 160     # 10 мс при 16 кГц
WIN_LENGTH = 400
N_MELS = 80
N_MFCC = 13
FMIN = 20
FMAX = 7_600


def summarize_audio(filename: str, audio: np.ndarray, sample_rate: int) -> dict[str, float | int | str]:
    """Возвращает базовую статистику качества аудио для одной волновой формы."""

    duration_s = len(audio) / sample_rate if sample_rate else 0.0
    abs_audio = np.abs(audio)
    peak = float(abs_audio.max()) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    # Для WAV с плавающей точкой, клиппированных ниже 1.0, считаем отсчёты,
    # прижатые к наблюдаемому пику. У чистых speech-like примеров очень мало
    # точных пиковых отсчётов; у клиппированных примеров видны пиковые плато.
    clipping_threshold = min(0.999, 0.98 * peak) if peak > 0 else 0.999
    clipping_fraction = float(np.mean(abs_audio >= clipping_threshold)) if audio.size else 0.0

    # Простая лабораторная аппроксимация тишины: отсчёты ниже 2% пиковой амплитуды.
    silence_threshold = max(0.02 * peak, 1e-4)
    silence_fraction = float(np.mean(abs_audio < silence_threshold)) if audio.size else 0.0

    return {
        "filename": filename,
        "sample_rate_hz": sample_rate,
        "duration_s": round(duration_s, 3),
        "num_samples": int(len(audio)),
        "peak_amplitude": round(peak, 6),
        "rms": round(rms, 6),
        "silence_fraction": round(silence_fraction, 6),
        "clipping_fraction": round(clipping_fraction, 6),
    }


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32), int(sr)


def save_waveform(audio: np.ndarray, sr: int, title: str, out_path: Path, zoom: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    if zoom:
        start = min(int(0.5 * sr), max(0, len(audio) - int(0.1 * sr)))
        end = min(start + int(0.1 * sr), len(audio))
        librosa.display.waveshow(audio[start:end], sr=sr, ax=ax)
        ax.set_title(f"{title} — увеличенная волновая форма (100 мс)")
    else:
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set_title(f"{title} — волновая форма")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Амплитуда")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_stft_spectrogram(audio: np.ndarray, sr: int, title: str, out_path: Path) -> None:
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    ax.set_title(f"{title} — лог-спектрограмма STFT")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Частота (Гц)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_log_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)


def save_log_mel(audio: np.ndarray, sr: int, title: str, out_path: Path) -> None:
    log_mel = compute_log_mel(audio, sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(log_mel, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=ax, cmap="viridis")
    ax.set_title(f"{title} — log-Mel-спектрограмма ({N_MELS} полос)")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Mel-частота")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_mfcc(audio: np.ndarray, sr: int, title: str, out_path: Path) -> None:
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LENGTH, x_axis="time", ax=ax, cmap="coolwarm")
    ax.set_title(f"{title} — MFCC ({N_MFCC} коэффициентов)")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Коэффициент MFCC")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_contact_sheet(items: list[tuple[str, np.ndarray, int]], out_path: Path) -> None:
    if not items:
        return
    rows = len(items)
    fig, axes = plt.subplots(rows, 2, figsize=(12, max(3, rows * 2.4)))
    if rows == 1:
        axes = np.array([axes])
    for row, (name, audio, sr) in enumerate(items):
        librosa.display.waveshow(audio, sr=sr, ax=axes[row, 0])
        axes[row, 0].set_title(f"{name} — волновая форма")
        axes[row, 0].set_xlabel("Время (с)")
        axes[row, 0].set_ylabel("Амплитуда")
        log_mel = compute_log_mel(audio, sr)
        librosa.display.specshow(log_mel, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=axes[row, 1], cmap="viridis")
        axes[row, 1].set_title(f"{name} — log-Mel")
        axes[row, 1].set_xlabel("Время (с)")
        axes[row, 1].set_ylabel("Mel")
    fig.suptitle("Лабораторная 01: сравнение признаков", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary_csv(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "sample_rate_hz", "duration_s", "num_samples", "peak_amplitude", "rms", "silence_fraction", "clipping_fraction"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_audio_dir(audio_dir: Path | str, out_dir: Path | str) -> list[dict[str, float | int | str]]:
    audio_dir = Path(audio_dir)
    out_dir = Path(out_dir)
    wav_paths = sorted(audio_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"В {audio_dir} не найдены .wav файлы. Сначала запустите make_lab_audio.py.")

    rows: list[dict[str, float | int | str]] = []
    loaded: list[tuple[str, np.ndarray, int]] = []
    for wav_path in wav_paths:
        audio, sr = load_audio(wav_path)
        stem = wav_path.stem
        rows.append(summarize_audio(wav_path.name, audio, sr))
        loaded.append((wav_path.name, audio, sr))
        save_waveform(audio, sr, wav_path.name, out_dir / f"{stem}_waveform.png")
        save_waveform(audio, sr, wav_path.name, out_dir / f"{stem}_waveform_zoom.png", zoom=True)
        save_stft_spectrogram(audio, sr, wav_path.name, out_dir / f"{stem}_stft_spectrogram.png")
        save_log_mel(audio, sr, wav_path.name, out_dir / f"{stem}_log_mel.png")
        save_mfcc(audio, sr, wav_path.name, out_dir / f"{stem}_mfcc.png")

    write_summary_csv(rows, out_dir / "audio_summary.csv")
    save_contact_sheet(loaded, out_dir / "feature_comparison_contact_sheet.png")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Строит графики волновой формы, спектрограммы, log-Mel и MFCC для аудио лабораторной 01.")
    parser.add_argument("--audio-dir", type=Path, default=Path("updated_course/lab_01_ru/data/wav"))
    parser.add_argument("--out-dir", type=Path, default=Path("updated_course/lab_01_ru/outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_audio_dir(args.audio_dir, args.out_dir)
    print(f"Проанализировано WAV-файлов: {len(rows)} из {args.audio_dir}")
    print(f"Графики и audio_summary.csv записаны в {args.out_dir}")


if __name__ == "__main__":
    main()
