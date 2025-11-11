import pandas as pd
import numpy as np

def extract_features(data, fs=250):
    required_columns = {"Channel1", "Channel2", "Condition"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    window_size = fs
    features = []

    for i in range(0, len(data) - window_size + 1, window_size):
        chunk = data.iloc[i:i + window_size]
        sig = chunk[["Channel1", "Channel2"]].values
        centered = sig - sig.mean(axis=0)
        power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
        freqs = np.fft.rfftfreq(window_size, d=1 / fs)

        delta = _band_power(power, freqs, 0.5, 4)
        theta = _band_power(power, freqs, 4, 8)
        alpha = _band_power(power, freqs, 8, 13)
        beta = _band_power(power, freqs, 13, 30)

        label = chunk["Condition"].mode()[0]
        features.append([delta, theta, alpha, beta, label])

    if not features:
        raise ValueError(f"Need at least {window_size} samples to extract one feature window")

    return pd.DataFrame(features, columns=["Delta", "Theta", "Alpha", "Beta", "Condition"])


def _band_power(power, freqs, low, high):
    band = (freqs >= low) & (freqs < high)
    return power[band].mean()
