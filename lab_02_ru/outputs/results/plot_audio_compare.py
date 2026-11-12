import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

# ---------- 1. Настройки ----------
ITEM_ID = "1272-128104-0000"
COND_AUG = "reverb_room_a"  # "noise_10db", "reverb_room_a", "speed_0.9x" и т.д.

BASE_DIR = Path("/mnt/d/ASR_ITMO/ASR_course_labs/lab_02_ru")
AUG_DIR = BASE_DIR / "outputs" / "augmented"

# ---------- 2. Функция графика ----------
def plot_comparison(y_clean, y_aug, sr, title_clean, title_aug, out_file):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # Waveform
    librosa.display.waveshow(y_clean, sr=sr, ax=axes[0, 0], color='tab:blue')
    axes[0, 0].set_title(f"{title_clean} waveform")
    axes[0, 0].set_xlabel("Time")
    
    librosa.display.waveshow(y_aug, sr=sr, ax=axes[1, 0], color='tab:blue')
    axes[1, 0].set_title(f"{title_aug} waveform")
    axes[1, 0].set_xlabel("Time")

    # Log-Mel
    S_clean = librosa.feature.melspectrogram(y=y_clean, sr=sr, n_fft=400, hop_length=160, n_mels=80)
    S_aug = librosa.feature.melspectrogram(y=y_aug, sr=sr, n_fft=400, hop_length=160, n_mels=80)
    S_clean_db = librosa.power_to_db(S_clean, ref=np.max)
    S_aug_db = librosa.power_to_db(S_aug, ref=np.max)

    img1 = librosa.display.specshow(S_clean_db, sr=sr, hop_length=160, x_axis='time', y_axis='mel', ax=axes[0, 1], cmap='viridis')
    axes[0, 1].set_title(f"{title_clean} log-Mel")
    fig.colorbar(img1, ax=axes[0, 1], format='%+2.0f dB')

    img2 = librosa.display.specshow(S_aug_db, sr=sr, hop_length=160, x_axis='time', y_axis='mel', ax=axes[1, 1], cmap='viridis')
    axes[1, 1].set_title(f"{title_aug} log-Mel")
    fig.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"✅ Сохранено: {out_file}")

# ---------- 3. Запуск ----------
if __name__ == "__main__":
    # ВАЖНО: обрати внимание на двойное подчеркивание "__" в имени файла!
    path_clean = AUG_DIR / f"{ITEM_ID}__clean.wav"
    path_aug = AUG_DIR / f"{ITEM_ID}__{COND_AUG}.wav"

    if not path_clean.exists():
        raise FileNotFoundError(f"Не найден файл: {path_clean}")
    if not path_aug.exists():
        raise FileNotFoundError(f"Не найден файл: {path_aug}")

    print(f"Загружаем: {path_clean.name} и {path_aug.name}")
    
    y_clean, sr = librosa.load(path_clean, sr=16000)
    y_aug, _ = librosa.load(path_aug, sr=16000)

    out_name = f"compare_{ITEM_ID}_{COND_AUG}.png"
    plot_comparison(y_clean, y_aug, sr, "clean", COND_AUG, out_name)