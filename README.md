# Induction Machine Fault Detection using FFT + Random Forest & CNN

## Problem Statement

Induction motors are the most widely used machines in industrial systems. Faults — if undetected — lead to costly downtime and equipment damage. Early fault detection through current signal analysis is a proven, non-invasive diagnostic technique.

This project builds a dual-model system that analyzes the **Motor Current Signature Analysis (MCSA)** of an induction machine and classifies it as **healthy** or **faulty** based on harmonic distortion patterns in the frequency spectrum.

---

## The Core Insight: Why Harmonics?

A healthy induction motor running on 50Hz power produces a current signal dominated by the fundamental frequency (50Hz). Faults — such as broken rotor bars, bearing defects, or winding issues — introduce **harmonic distortion**: abnormal amplitudes at odd harmonic frequencies (150Hz, 250Hz, 350Hz — the 3rd, 5th, and 7th harmonics).

This project detects faults by:
1. Applying **Fast Fourier Transform (FFT)** to convert raw time-domain signals into the frequency domain
2. Extracting harmonic amplitudes at 50Hz, 150Hz, 250Hz, and 350Hz
3. Classifying the machine state using two independent ML models

---

## System Architecture

```
Raw Motor Current Signal (CSV)
            │
            ▼
    FFT Signal Processing
            │
            ├──────────────────────────┐
            ▼                          ▼
  Feature Extraction            Raw Signal Array
  (4 harmonic amplitudes)       (10,000 samples)
            │                          │
            ▼                          ▼
   Random Forest Classifier       1D CNN Model
   (tabular features)          (learns spatial patterns)
            │                          │
            └──────────┬───────────────┘
                       ▼
              HEALTHY / FAULTY + Confidence %
```

---

## Models

### Model 1 — Random Forest on FFT Features
- Input: 4 harmonic amplitude values extracted via FFT
- 500 estimators
- Trained on 1000 synthetic samples (500 healthy, 500 faulty)
- Fast inference — runs in under 1 second

### Model 2 — 1D Convolutional Neural Network on Raw Signal
- Input: Full 10,000-sample waveform reshaped as (10000, 1)
- Architecture: Conv1D(32) → MaxPool → Conv1D(64) → MaxPool → Dense(64) → Sigmoid
- Trained for 5 epochs on the same dataset
- Learns spatial temporal patterns directly from raw signal

### Why Two Models?
Random Forest with FFT features provides fast, interpretable results — the feature importance plot shows exactly which harmonic frequency contributes most to the fault decision. CNN operates on raw signals without manual feature engineering, validating results from a different perspective. Agreement between models increases diagnostic confidence.

---

## Data Generation

Real motor fault datasets are difficult to obtain. This project uses **physics-informed synthetic data** based on established MCSA theory:

```python
# Healthy motor: harmonic amplitudes 0.05–0.25 (low distortion)
# Faulty motor:  harmonic amplitudes 0.26–0.55 (elevated distortion)

signal = (
    1.0 * sin(2π × 50t)     # Fundamental
  + a3  * sin(2π × 150t)    # 3rd harmonic
  + a5  * sin(2π × 250t)    # 5th harmonic
  + a7  * sin(2π × 350t)    # 7th harmonic
  + Gaussian noise (σ=0.02)
)
```

The amplitude thresholds are grounded in real-world motor fault literature where harmonic amplitudes above ~25% of the fundamental indicate fault conditions.

---

## Desktop Application

Built with **Tkinter** — a GUI that allows non-technical users to interact with the diagnostic system:

- Load a real or synthetic motor current CSV file
- View time-domain signal and FFT frequency spectrum side by side
- One-click analysis with confidence percentage output
- Color-coded result (green = healthy, red = faulty)
- Live harmonic amplitude readout at 50Hz, 150Hz, 250Hz, 350Hz

---

## Results

| Model | Accuracy |
|-------|----------|
| Random Forest (FFT features) | ~99% on synthetic test set |
| CNN (raw signal) | ~97–99% on synthetic test set |

Feature importance analysis shows the 3rd harmonic (150Hz) carries the highest discriminative weight, consistent with motor fault literature.

---

## Key Learnings

- FFT-based feature engineering is a powerful bridge between signal processing and machine learning
- Random Forest provides interpretability through feature importance — critical for engineering applications where explainability matters
- 1D CNN can learn fault-relevant patterns directly from raw waveforms without domain-specific feature engineering
- Synthetic physics-informed data is a valid starting point when real datasets are unavailable

---

## Tech Stack

- Python, NumPy, SciPy
- scikit-learn (Random Forest)
- TensorFlow / Keras (1D CNN)
- Matplotlib (visualization)
- Tkinter (desktop GUI)

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow

# Train models and view plots
python motor_ai.py

# Launch desktop application
python ui.py
```

Load any CSV file with a single column of motor current samples (10,000 samples at 10kHz sampling rate).