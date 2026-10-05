#!/usr/bin/env python3
"""Запускает шаги Lab 02."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = LAB_DIR / "scripts"


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True, cwd=LAB_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=LAB_DIR / "outputs")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--model-dir", default=None, help="Local cached tiny.en directory; avoids normal-run network access.")
    parser.add_argument("--allow-model-download", action="store_true", help="Explicit one-time setup only; normal runs remain cache-only.")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve(); augmented = out_dir / "augmented"
    manifest = LAB_DIR / "data" / "manifest.jsonl"
    run(str(SCRIPTS / "make_lab02_subset.py"), "--manifest", str(manifest))
    run(str(SCRIPTS / "augment_audio.py"), "--manifest", str(manifest), "--out-dir", str(augmented), "--seed", str(args.seed))
    run(str(SCRIPTS / "plot_augmentation_features.py"), "--clean", str(LAB_DIR / "data/wav/1272-128104-0000.wav"), "--augmented", str(augmented / "1272-128104-0000__noise_10db.wav"), "--out-dir", str(out_dir / "figures"))
    if args.skip_asr:
        print("Skipped ASR by request. Figures and deterministic waveform conditions are ready.")
        return
    raw = out_dir / "raw_asr.jsonl"
    asr_args = [str(SCRIPTS / "run_local_asr.py"), "--conditions", str(augmented / "conditions.jsonl"), "--manifest", str(manifest), "--out", str(raw)]
    if args.model_dir:
        asr_args += ["--model-dir", args.model_dir]
    if args.allow_model_download:
        asr_args += ["--allow-model-download"]
    run(*asr_args)
    run(str(SCRIPTS / "score_robustness.py"), "--input", str(raw), "--out-dir", str(out_dir / "results"))
    print(f"Lab 02 completed. Results: {out_dir / 'results'}")


if __name__ == "__main__":
    main()
