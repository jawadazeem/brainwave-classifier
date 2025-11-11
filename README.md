# EEG Auditory Stimulation Analysis

This project investigates the correlation between auditory environments (Nature, White Noise, City) and prefrontal cortex neural activity using spectral analysis. 

## Methodology
Data is processed using Fast Fourier Transform (FFT) to extract spectral power from specific frequency bands (Alpha, Beta, Theta). A Random Forest classifier is employed to predict the auditory environment based on neural signatures.

## Data Structure
- `Time`: Sampling index.
- `Channel[1-3]`: EEG signal in µV.
- `Condition`: Exposure state (Nature, WhiteNoise, City).

## Setup & Running
1. Ensure dependencies are installed: `pip install pandas numpy scikit-learn`.
2. Generate synthetic data (if needed): `python generate_data.py`.
3. Train and evaluate the model: `python train_model.py`.

## References
Azeem, J., & Malik, S. (2025). The Effect of Auditory Stimulation on Prosocial Decision-Making and EEG Brain Activity Using the Keep or Share Method.