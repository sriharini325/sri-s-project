"""
sort_asvspoof.py
----------------
ASVspoof 2019 LA doesn't sort files into "real" and "fake" folders --
labels live in a separate protocol .txt file. This script reads that
protocol file, matches each entry to its .flac audio file, converts it
to .wav, and copies it into data/real/ or data/fake/ so the rest of
the project (dataset.py, train.py) can use it unchanged.

Usage:
    python sort_asvspoof.py \
        --protocol /path/to/ASVspoof2019.LA.cm.train.trn.txt \
        --flac_dir /path/to/ASVspoof2019_LA_train/flac \
        --max_per_class 3000

--max_per_class is optional. ASVspoof's train split has far more spoof
files (~22,800) than bonafide (~2,580) -- this caps each class so your
first training run is balanced and manageable in size. Omit it (or set
to 0) to copy everything.
"""

import argparse
import os
import soundfile as sf
import librosa


def sort_asvspoof(protocol_path, flac_dir, out_real="data/real", out_fake="data/fake",
                   max_per_class=0, target_sr=16000):
    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_fake, exist_ok=True)

    real_count = 0
    fake_count = 0
    skipped = 0

    with open(protocol_path, "r") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} entries in protocol file.")

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # malformed line, skip

        file_name = parts[1]       # e.g. LA_T_1138215
        label = parts[-1]          # "bonafide" or "spoof"

        # Stop early per class if a cap was set
        if label == "bonafide" and max_per_class and real_count >= max_per_class:
            continue
        if label == "spoof" and max_per_class and fake_count >= max_per_class:
            continue

        flac_path = os.path.join(flac_dir, file_name + ".flac")
        if not os.path.exists(flac_path):
            skipped += 1
            continue

        # Load and resample, then write out as .wav
        signal, sr = librosa.load(flac_path, sr=target_sr)
        dest_dir = out_real if label == "bonafide" else out_fake
        dest_path = os.path.join(dest_dir, file_name + ".wav")
        sf.write(dest_path, signal, target_sr)

        if label == "bonafide":
            real_count += 1
        else:
            fake_count += 1

        # Stop entirely once both classes hit the cap
        if max_per_class and real_count >= max_per_class and fake_count >= max_per_class:
            break

    print(f"Done. Copied {real_count} bonafide -> {out_real}")
    print(f"Copied {fake_count} spoof -> {out_fake}")
    if skipped:
        print(f"Skipped {skipped} entries (flac file not found).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, help="Path to the protocol .txt file")
    parser.add_argument("--flac_dir", required=True, help="Path to the folder containing .flac files")
    parser.add_argument("--out_real", default="data/real")
    parser.add_argument("--out_fake", default="data/fake")
    parser.add_argument("--max_per_class", type=int, default=0,
                         help="Cap files per class (0 = no cap, copy everything)")
    args = parser.parse_args()

    sort_asvspoof(args.protocol, args.flac_dir, args.out_real, args.out_fake, args.max_per_class)
