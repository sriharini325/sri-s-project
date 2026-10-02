"""
generate_sample_data.py
------------------------
Creates a small synthetic dataset so you can run the ENTIRE pipeline
(preprocessing -> features -> training -> evaluation -> app) immediately,
without waiting on a real dataset download or unzip.

This is a bootstrap/smoke-test dataset, not a research dataset:
- "bonafide" clips: smooth harmonic tones with natural pitch drift and
  soft noise, loosely mimicking the harmonic structure of a human voice.
- "spoof" clips: the same base tone but with robotic artifacts layered in
  (phase jumps, bit-quantization, unnatural periodicity) that mimic the
  kind of artifacts neural vocoders/TTS systems leave behind.

Only needs numpy + scipy (both very commonly pre-installed), so it works
even before you've finished installing everything in requirements.txt.

Usage:
    python generate_sample_data.py --count 100
"""

import argparse
import os
import numpy as np
from scipy.io import wavfile


def make_bonafide_clip(duration, sr, rng):
    """A smooth, harmonic, slightly-imperfect tone -- stands in for natural human speech."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Natural pitch isn't perfectly steady -- add slow random drift
    base_freq = rng.uniform(110, 220)  # roughly human vocal pitch range
    drift = np.cumsum(rng.normal(0, 0.5, size=t.shape)) / sr
    freq = base_freq + drift * 5

    # A few harmonics with naturally decaying amplitude (like a real voice)
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        amp = 1.0 / harmonic
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * harmonic * freq * t + phase)

    # Soft, natural-sounding background noise
    signal += rng.normal(0, 0.02, size=t.shape)

    # Gentle amplitude envelope (speech isn't constant volume)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 2 * np.pi))
    signal *= envelope

    return signal / (np.max(np.abs(signal)) + 1e-8)


def make_spoof_clip(duration, sr, rng):
    """The same idea, but with synthetic-sounding artifacts layered on top."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    base_freq = rng.uniform(110, 220)
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        amp = 1.0 / harmonic
        signal += amp * np.sin(2 * np.pi * harmonic * base_freq * t)  # perfectly steady pitch

    # Bit-quantization artifact (common in vocoder output)
    levels = 32
    signal = np.round(signal * levels) / levels

    # Occasional phase discontinuities (glitchy jumps TTS systems sometimes leave)
    n_glitches = rng.integers(3, 8)
    for _ in range(n_glitches):
        idx = rng.integers(0, len(signal) - 200)
        signal[idx:idx + 200] *= -1

    # Very uniform, robotic amplitude envelope (too regular to be natural)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    signal *= envelope

    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_dataset(out_real="data/real", out_fake="data/fake", count=100,
                      duration=4.0, sr=16000, seed=42):
    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_fake, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(count):
        real_signal = make_bonafide_clip(duration, sr, rng)
        fake_signal = make_spoof_clip(duration, sr, rng)

        # scipy expects int16 PCM for standard wav playback
        wavfile.write(os.path.join(out_real, f"real_{i:04d}.wav"), sr,
                       (real_signal * 32767).astype(np.int16))
        wavfile.write(os.path.join(out_fake, f"fake_{i:04d}.wav"), sr,
                       (fake_signal * 32767).astype(np.int16))

    print(f"Generated {count} bonafide clips -> {out_real}")
    print(f"Generated {count} spoof clips    -> {out_fake}")
    print("\nThis is synthetic bootstrap data so you can test the full pipeline")
    print("immediately. Replace it with real ASVspoof/FoR data (see sort_asvspoof.py)")
    print("before reporting any results in your project write-up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="Clips to generate per class")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    generate_dataset(count=args.count, duration=args.duration, sr=args.sr)
