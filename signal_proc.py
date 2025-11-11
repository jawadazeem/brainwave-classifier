import numpy as np
import pandas as pd

def extract_features(df, fs=50):
    features = []
    for i in range(0, len(df) - fs, fs):
        chunk = df.iloc[i:i+fs]
        sig = chunk[["Channel1", "Channel2"]].values
        fft = np.abs(np.fft.rfft(sig, axis=0))**2
        
        # Power per band
        row = [fft[8:12].mean(), fft[13:30].mean(), fft[4:8].mean(), chunk["Condition"].mode()[0]]
        features.append(row)
    return pd.DataFrame(features, columns=["Alpha", "Beta", "Theta", "Condition"])