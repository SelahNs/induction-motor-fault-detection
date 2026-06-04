import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def generate_sample(condition):
    fs = 10000
    T = 1.0
    t = np.linspace(0, T, int(fs * T))

    if condition == 'healthy':
        a3 = np.random.uniform(0.05, 0.25)
        a5 = np.random.uniform(0.05, 0.25)
        a7 = np.random.uniform(0.05, 0.25)
    else:
        a3 = np.random.uniform(0.26, 0.55)
        a5 = np.random.uniform(0.26, 0.55)
        a7 = np.random.uniform(0.26, 0.55)

    signal = (
        1.0 * np.sin(2 * np.pi * 50 * t)
      + a3  * np.sin(2 * np.pi * 150 * t)
      + a5  * np.sin(2 * np.pi * 250 * t)
      + a7  * np.sin(2 * np.pi * 350 * t)
      + np.random.normal(0, 0.02, len(t))
    )

    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / fs)
    amplitudes = np.abs(fft_result) / len(signal) * 2

    def get_amp(freq):
        idx = np.argmin(np.abs(freqs - freq))
        return amplitudes[idx]

    return {
        'amp_50hz'  : get_amp(50),
        'amp_150hz' : get_amp(150),
        'amp_250hz' : get_amp(250),
        'amp_350hz' : get_amp(350),
        'label'     : condition
    }


def train_model():
    rows = []
    for _ in range(500):
        rows.append(generate_sample('healthy'))
    for _ in range(500):
        rows.append(generate_sample('faulty'))

    df = pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)
    X = df[['amp_50hz', 'amp_150hz', 'amp_250hz', 'amp_350hz']]
    y = df['label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def get_fft(signal, fs=10000):
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / fs)
    amplitudes = np.abs(fft_result) / len(signal) * 2
    half = len(freqs) // 2
    return freqs[:half], amplitudes[:half]


def get_features(signal, fs=10000):
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / fs)
    amplitudes = np.abs(fft_result) / len(signal) * 2

    def get_amp(freq):
        idx = np.argmin(np.abs(freqs - freq))
        return amplitudes[idx]

    return [get_amp(50), get_amp(150), get_amp(250), get_amp(350)]


print("Training model please wait...")
rf = train_model()
print("Done. Ready.")

root = tk.Tk()
root.title("Motor Fault Detector — Group 13")
root.geometry("950x680")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Induction Machine Harmonic Fault Detector",
         font=("Arial", 16, "bold"), bg="#1F4E79", fg="white",
         pady=12).pack(fill=tk.X)

tk.Label(root, text="Group 13 — ECEG-3151 | Addis Ababa University",
         font=("Arial", 10), bg="#2E75B6", fg="white",
         pady=6).pack(fill=tk.X)

control_frame = tk.Frame(root, bg="#f0f0f0", pady=15)
control_frame.pack()

file_var = tk.StringVar(value="No file loaded")
tk.Label(control_frame, textvariable=file_var,
         font=("Arial", 10), bg="#f0f0f0", fg="#555555",
         width=35).grid(row=0, column=1, padx=10)

result_var = tk.StringVar(value="Load a CSV file and click Analyze")
result_label = tk.Label(control_frame, textvariable=result_var,
                        font=("Arial", 13, "bold"),
                        bg="#f0f0f0", fg="#333333", width=35)
result_label.grid(row=0, column=3, padx=20)

plot_frame = tk.Frame(root, bg="#f0f0f0")
plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.patch.set_facecolor('#f0f0f0')
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

amp_frame = tk.Frame(root, bg="#f0f0f0")
amp_frame.pack(pady=5)

amp_labels = {}
for i, freq in enumerate(['50Hz', '150Hz', '250Hz', '350Hz']):
    tk.Label(amp_frame, text=f"{freq}:",
             font=("Arial", 10, "bold"), bg="#f0f0f0").grid(row=0, column=i*2, padx=8)
    var = tk.StringVar(value="—")
    tk.Label(amp_frame, textvariable=var,
             font=("Arial", 10), bg="#f0f0f0", width=8).grid(row=0, column=i*2+1, padx=4)
    amp_labels[freq] = var

loaded_signal = [None]


def load_csv():
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not path:
        return
    try:
        df = pd.read_csv(path)
        signal = df.iloc[:, 0].values
        loaded_signal[0] = signal
        file_var.set(path.split("/")[-1])
        result_var.set("File loaded. Click Analyze.")
        result_label.config(fg="#333333")

        t = np.linspace(0, 1, len(signal))
        axes[0].cla()
        axes[0].plot(t[:500], signal[:500], color='steelblue')
        axes[0].set_title('Loaded Motor Current Signal')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Current (A)')

        freqs, amps = get_fft(signal)
        axes[1].cla()
        axes[1].plot(freqs, amps, color='steelblue')
        axes[1].set_xlim(0, 600)
        axes[1].set_title('FFT Frequency Spectrum')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Amplitude')

        fig.tight_layout()
        canvas.draw()

    except Exception as e:
        messagebox.showerror("Error", f"Could not read file:\n{e}")


def analyze():
    if loaded_signal[0] is None:
        messagebox.showwarning("No Signal", "Please load a CSV file first.")
        return

    signal = loaded_signal[0]
    features = get_features(signal)
    prediction = rf.predict([features])[0]
    proba = rf.predict_proba([features])[0]
    confidence = max(proba) * 100
    print(f"Prediction: {prediction}, Confidence: {confidence:.1f}%")

    amp_labels['50Hz'].set(f"{features[0]:.3f}")
    amp_labels['150Hz'].set(f"{features[1]:.3f}")
    amp_labels['250Hz'].set(f"{features[2]:.3f}")
    amp_labels['350Hz'].set(f"{features[3]:.3f}")

    color = 'red' if prediction == 'faulty' else 'green'

    axes[0].cla()
    t = np.linspace(0, 1, len(signal))
    axes[0].plot(t[:500], signal[:500], color=color)
    axes[0].set_title('Motor Current Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Current (A)')

    freqs, amps = get_fft(signal)
    axes[1].cla()
    axes[1].plot(freqs, amps, color=color)
    axes[1].set_xlim(0, 600)
    axes[1].set_title('FFT Frequency Spectrum')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Amplitude')

    fig.tight_layout()
    canvas.draw()
    
    if prediction == 'faulty':
        result_var.set(f"FAULTY MOTOR  ({confidence:.1f}% confidence)")
        result_label.config(fg="red")
    else:
        result_var.set(f"HEALTHY MOTOR  ({confidence:.1f}% confidence)")
        result_label.config(fg="green")
    root.update()


tk.Button(control_frame, text="Load CSV",
          font=("Arial", 11, "bold"),
          bg="#2E75B6", fg="white",
          padx=15, pady=5,
          command=load_csv).grid(row=0, column=0, padx=10)

tk.Button(control_frame, text="Analyze",
          font=("Arial", 11, "bold"),
          bg="#1F4E79", fg="white",
          padx=15, pady=5,
          command=analyze).grid(row=0, column=2, padx=10)

root.mainloop()