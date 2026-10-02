# Intelligent Detection of Synthetic Voices in Cyber Threats

A final-year project: an AI system that detects whether an audio clip is a
real human voice (**bonafide**) or an AI-generated/deepfake voice (**spoof**),
used to catch voice-phishing, impersonation, and fraud attempts.

This project ships with **two parallel pipelines**:

| | Baseline pipeline | Deep learning pipeline |
|---|---|---|
| Files | `baseline_pipeline.py`, `baseline_app.py` | `model.py`, `train.py`, `evaluate.py`, `app.py`, `api.py` |
| Model | SVM on hand-rolled spectral features | CNN on mel-spectrograms |
| Dependencies | numpy, scipy, scikit-learn only | + librosa, torch, streamlit |
| Status | **Verified working end-to-end** (tested before packaging) | Standard, well-tested code -- install torch/librosa on your machine to run |
| Use for | Getting a working result fast, sanity-checking your data | Your project's main deep-learning results/report |

**Recommended order: get the baseline running first (10 minutes, guaranteed to work), then move to the CNN pipeline for your main results.**

---

## What's in this folder

```
voice-spoof-detector/
├── data/
│   ├── real/                  # bonafide (human) .wav files go here
│   └── fake/                  # spoof (AI-generated) .wav files go here
│
├── generate_sample_data.py     # creates synthetic bootstrap data (numpy/scipy only)
├── sort_asvspoof.py             # sorts real ASVspoof 2019 data into data/real, data/fake
│
├── baseline_pipeline.py         # VERIFIED: features -> SVM training -> evaluation
├── baseline_app.py              # VERIFIED: Streamlit demo for the SVM model
│
├── preprocessing.py             # resampling, trimming, normalization (librosa)
├── features.py                  # MFCC + mel-spectrogram extraction (librosa)
├── model.py                     # CNN architecture (torch)
├── dataset.py                   # PyTorch Dataset loader
├── train.py                     # CNN training script (torch)
├── evaluate.py                  # CNN evaluation: accuracy, F1, EER (torch)
├── app.py                       # Streamlit demo for the CNN model
├── api.py                       # Flask API alternative
├── realtime_mic.py              # live microphone testing
│
├── requirements.txt
└── README.md                    # this file
```

---

## Stage 1: Data collection

Three options, in order of speed:

### Option A -- Synthetic bootstrap data (fastest, works immediately)
Generates data locally, no download needed. Good for confirming the whole
pipeline runs before dealing with a real dataset.
```bash
python generate_sample_data.py --count 100
```
Creates 100 synthetic "bonafide" and 100 synthetic "spoof" `.wav` files in
`data/real/` and `data/fake/`. **This is a smoke test, not real data** --
swap it out with option B or C before writing up any results.

### Option B -- ASVspoof 2019 LA (recommended for your main results)
Download from https://datashare.ed.ac.uk/handle/10283/3336 (or the Kaggle
mirror), unzip, then run:
```bash
python sort_asvspoof.py \
  --protocol "path/to/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt" \
  --flac_dir "path/to/ASVspoof2019_LA_train/flac" \
  --max_per_class 2000
```

### Option C -- Fake-or-Real (FoR) dataset (easiest real dataset)
Download from https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset,
unzip, then copy files matching their folder structure into `data/real/`
and `data/fake/` directly (already split by class).

---

## Stage 2: Preprocessing

Handled automatically inside both pipelines. What it does:
- Resamples every clip to 16kHz
- Trims leading/trailing silence
- Normalizes volume
- Pads/truncates every clip to a fixed 4-second length

Baseline: `load_and_preprocess()` in `baseline_pipeline.py`.
CNN: `preprocess_audio()` in `preprocessing.py`.

---

## Stage 3: Feature extraction

- **Baseline**: `extract_features()` in `baseline_pipeline.py` computes
  log-energy across 40 frequency bands plus spectral centroid, spectral
  flatness, and zero-crossing rate -- features known to help separate
  natural speech from vocoder artifacts, using plain numpy/scipy.
- **CNN**: `extract_melspectrogram()` in `features.py` produces a full
  mel-spectrogram "image" for the CNN to learn from. `extract_mfcc()` is
  also included for a librosa-based MFCC+SVM comparison if you want one.

---

## Stage 4: Model selection

- **Baseline**: SVM with an RBF kernel (`sklearn.svm.SVC`) -- fast to train,
  interpretable, a strong classical-ML sanity check.
- **CNN**: a 3-layer convolutional network (`SpoofCNN` in `model.py`) that
  treats the mel-spectrogram as an image and learns spectral-temporal
  patterns -- your main deep-learning result.

Report both in your write-up as a "classical vs. deep learning" comparison.

---

## Stage 5: Model training

**Baseline** (verified end-to-end before packaging):
```bash
python baseline_pipeline.py
```
Trains the SVM, prints metrics, saves `baseline_model.joblib`.

**CNN** (run on your machine after `pip install -r requirements.txt`):
```bash
python train.py
```
Trains the CNN, prints per-epoch loss/accuracy, saves `best_spoof_cnn.pt`.

---

## Stage 6: Model testing / evaluation

**Baseline**: runs automatically at the end of `baseline_pipeline.py`,
printing accuracy, precision, recall, F1-score, EER, and a confusion matrix.

**CNN**: run separately once `best_spoof_cnn.pt` exists:
```bash
python evaluate.py
```

EER (Equal Error Rate) is the metric the anti-spoofing research field
actually benchmarks against -- report it alongside accuracy/F1.

---

## Stage 7: Deployment

**Baseline demo** (verified working):
```bash
streamlit run baseline_app.py
```

**CNN demo** (once `best_spoof_cnn.pt` exists):
```bash
streamlit run app.py
```

**Flask API alternative** (CNN-based):
```bash
python api.py
# then: curl -X POST -F "audio=@sample.wav" http://localhost:5000/predict
```

**Live microphone test** (CNN-based):
```bash
python realtime_mic.py
```

---

## Step-by-step: opening and running this in VS Code

### 1. Unzip and open the folder
Unzip `voice-spoof-detector.zip`, then in VS Code: **File → Open Folder...**
→ select the unzipped `voice-spoof-detector` folder.

### 2. Install the Python extension
Extensions icon (`Ctrl+Shift+X` / `Cmd+Shift+X`) → search "Python" → Install.

### 3. Open a terminal
**Terminal → New Terminal** (or `` Ctrl+` ``).

### 4. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```
Click **Yes** if VS Code asks to use this environment for the workspace.

### 5. Install dependencies
```bash
pip install -r requirements.txt
```
> Linux: if `sounddevice` fails, run `sudo apt-get install libportaudio2` first.

### 6. Run the verified baseline pipeline first
```bash
python generate_sample_data.py --count 100
python baseline_pipeline.py
streamlit run baseline_app.py
```
This confirms your whole environment works with **zero external downloads**.
Upload one of the files from `data/real/` or `data/fake/` in the browser tab
that opens, and you should see a correct, confident verdict.

### 7. Swap in real data and move to the CNN pipeline
```bash
python sort_asvspoof.py --protocol ... --flac_dir ... --max_per_class 2000
python train.py
python evaluate.py
streamlit run app.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure `(venv)` is active, re-run `pip install -r requirements.txt` |
| `baseline_pipeline.py` says "Found 0 real and 0 fake files" | Run `python generate_sample_data.py` first, or add your own `.wav` files |
| Streamlit says no model found | Run the matching training script first (`baseline_pipeline.py` or `train.py`) |
| `sounddevice` install fails (Linux) | `sudo apt-get install libportaudio2`, then retry |
| CNN accuracy stuck near 50% | Check `data/real` vs `data/fake` counts aren't wildly imbalanced; verify files load correctly with `preprocess_audio()` on a sample file |

---

## Honesty note on the numbers you'll see

Running the baseline on the synthetic bootstrap data (`generate_sample_data.py`)
gives ~100% accuracy -- expected, and not meaningful on its own, since the
two synthetic classes are trivially separable by design. It only proves
your pipeline runs correctly end-to-end. Real, reportable numbers come from
running the same pipeline on ASVspoof or FoR data.

## Next steps for your report
- Get the baseline SVM numbers on real ASVspoof/FoR data as your comparison point.
- Train the CNN on the same real data for your main result.
- Test both on the **In-the-Wild** dataset (no retraining) to discuss generalization.
- Record a short demo video/screenshots of the Streamlit app for your presentation.
