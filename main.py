from eeg_data_generator import generate_data 
from signal_proc import extract_features
from model_engine import train_and_evaluate

def run_pipeline():
    print("Initializing pipeline...")
    raw_data = generate_data()
    features = extract_features(raw_data)
    accuracy = train_and_evaluate(features)
    
    print(f"Pipeline execution complete. Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    run_pipeline()