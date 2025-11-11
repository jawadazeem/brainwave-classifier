import pandas as pd
import numpy as np

def generate_data(duration_seconds=30, fs=250, output_path="eeg_timeseries.csv", random_state=42):
    n_samples = duration_seconds * fs
    conditions = ["Nature", "WhiteNoise", "City"]
    rng = np.random.default_rng(random_state)
    data = []

    for sample_idx in range(n_samples):
        elapsed = sample_idx / fs
        cond = conditions[int(elapsed) % len(conditions)]

        if cond == "Nature":
            ch1 = 10 + 2.0 * np.sin(2 * np.pi * 10 * elapsed) + rng.normal(0, 0.5)
            ch2 = 8 + 1.8 * np.sin(2 * np.pi * 10 * elapsed) + rng.normal(0, 0.5)
        elif cond == "City":
            ch1 = 10 + 2.0 * np.sin(2 * np.pi * 20 * elapsed) + rng.normal(0, 0.8)
            ch2 = 8 + 1.8 * np.sin(2 * np.pi * 20 * elapsed) + rng.normal(0, 0.8)
        else:
            ch1 = 10 + rng.normal(0, 1.5)
            ch2 = 8 + rng.normal(0, 1.5)

        data.append([elapsed, ch1, ch2, cond])

    df = pd.DataFrame(data, columns=["Time", "Channel1", "Channel2", "Condition"])
    df.to_csv(output_path, index=False)
    print(f"Data generated and saved to {output_path}")
    return df
