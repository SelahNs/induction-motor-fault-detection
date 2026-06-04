import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models


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


def generate_signal(condition):
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

    return signal, condition


# generate dataset
rows = []
for _ in range(500):
    rows.append(generate_sample('healthy'))
for _ in range(500):
    rows.append(generate_sample('faulty'))

df = pd.DataFrame(rows)
df = df.sample(frac=1).reset_index(drop=True)

X = df[['amp_50hz', 'amp_150hz', 'amp_250hz', 'amp_350hz']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# random forest
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

for f, i in zip(['amp_50hz', 'amp_150hz', 'amp_250hz', 'amp_350hz'], rf.feature_importances_):
    print(f"{f}: {i:.3f}")

# cnn dataset
x_cnn, y_cnn = [], []
for _ in range(500):
    s, c = generate_signal('healthy')
    x_cnn.append(s)
    y_cnn.append(c)
for _ in range(500):
    s, c = generate_signal('faulty')
    x_cnn.append(s)
    y_cnn.append(c)

x_cnn = np.array(x_cnn).reshape(1000, 10000, 1)
y_cnn = (np.array(y_cnn) == 'faulty').astype(int)

X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    x_cnn, y_cnn, test_size=0.2, random_state=42
)

# cnn model
cnn = models.Sequential([
    layers.Conv1D(32, kernel_size=50, activation='relu', input_shape=(10000, 1)),
    layers.MaxPooling1D(pool_size=4),
    layers.Conv1D(64, kernel_size=25, activation='relu'),
    layers.MaxPooling1D(pool_size=4),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = cnn.fit(
    X_train_cnn, y_train_cnn,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_cnn, y_test_cnn)
)

print("CNN Evaluation:")
cnn.evaluate(X_test_cnn, y_test_cnn)

# plots
healthy_signal, _ = generate_signal('healthy')
faulty_signal, _  = generate_signal('faulty')
t = np.linspace(0, 1, 10000)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(t[:500], healthy_signal[:500], color='green')
axes[0].set_title('Healthy Motor Current')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Current (A)')
axes[1].plot(t[:500], faulty_signal[:500], color='red')
axes[1].set_title('Faulty Motor Current')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Current (A)')
plt.suptitle('Healthy vs Faulty Motor Current Signal')
plt.tight_layout()
plt.savefig('plot1_signals.png')
plt.show()

def get_fft(signal, fs=10000):
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / fs)
    amplitudes = np.abs(fft_result) / len(signal) * 2
    half = len(freqs) // 2
    return freqs[:half], amplitudes[:half]

h_freqs, h_amps = get_fft(healthy_signal)
f_freqs, f_amps = get_fft(faulty_signal)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(h_freqs, h_amps, color='green')
axes[0].set_xlim(0, 600)
axes[0].set_title('Healthy Motor FFT Spectrum')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Amplitude')
axes[1].plot(f_freqs, f_amps, color='red')
axes[1].set_xlim(0, 600)
axes[1].set_title('Faulty Motor FFT Spectrum')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Amplitude')
plt.suptitle('FFT Spectrum — Healthy vs Faulty')
plt.tight_layout()
plt.savefig('plot2_fft.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(['50Hz', '150Hz', '250Hz', '350Hz'], rf.feature_importances_,
       color=['gray', 'red', 'orange', 'yellow'])
ax.set_title('Random Forest Feature Importance')
ax.set_xlabel('Harmonic Frequency')
ax.set_ylabel('Importance Score')
plt.tight_layout()
plt.savefig('plot3_importance.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 6), history.history['accuracy'], color='blue', marker='o', label='Training Accuracy')
ax.plot(range(1, 6), history.history['val_accuracy'], color='green', marker='o', label='Validation Accuracy')
ax.set_title('CNN Training Accuracy per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.8, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig('plot4_cnn_accuracy.png')
plt.show()