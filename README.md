# Brainwave Classifier

This project generates synthetic EEG-like time series data, extracts frequency-band features with FFT (Fast Fourier Transform), and trains a Random Forest model to classify listening conditions.

## Overview
1. Generate synthetic two-channel EEG-like samples for `Nature`, `WhiteNoise`, and `City` conditions.
2. Save the generated samples to `eeg_timeseries.csv`.
3. Split the time series into 1-second windows.
4. Compute Delta, Theta, Alpha, and Beta power features for each window.
5. Train and evaluate a reproducible Random Forest classifier.

## Requirements

Install packages:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```text
pandas
numpy
matplotlib
scikit-learn
```


## Run the Script

```bash
python3 main.py
```

You’ll get:

- A regenerated `eeg_timeseries.csv` file
- Console output with the model accuracy

Example output:

```text
Initializing pipeline...
Data generated and saved to eeg_timeseries.csv
Pipeline execution complete. Model Accuracy: 1.00
```

## Project Files

- `main.py` runs the complete pipeline.
- `eeg_data_generator.py` creates synthetic EEG-like data and returns it as a DataFrame.
- `signal_proc.py` extracts frequency-band features without running training code on import.
- `model_engine.py` trains and evaluates the classifier.
- `eeg_timeseries.csv` is generated data and may change when the pipeline runs.

## Data Notice

The CSV uses synthetic (fake) EEG-like values for demonstration.
Do not upload real EEG data from participants or experiments publicly.

## Learning Value

- Data preprocessing and feature extraction
- Frequency analysis with FFT
- Machine learning model training and evaluation
- Reproducible train/test evaluation

## Author
Jawad Azeem
