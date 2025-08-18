import os
import json
import joblib
import numpy as np
from scipy.fftpack import fft

# -----------------------
# Feature extractor (must match training exactly)
# -----------------------
def extract_eeg_features(sample):
    features = []
    for ch in range(8):
        ch_data = sample[:, ch, :]
        ch_flat = ch_data.flatten()

        # Basic stats
        features.append(np.mean(ch_flat))
        features.append(np.std(ch_flat))
        features.append(np.max(ch_flat))
        features.append(np.min(ch_flat))

        # Frequency domain
        fft_vals = np.abs(fft(ch_flat))
        features.append(np.mean(fft_vals))
        features.append(np.std(fft_vals))

    return np.array(features)

# -----------------------
# Load JSON EEG sample
# -----------------------
def load_json_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
        sample = []
        # Sort by timestamp keys like "0ms", "10ms"
        for time_key in sorted(data.keys(), key=lambda x: int(x.replace("ms", ""))):
            sample.append(np.array(data[time_key]))
        return np.stack(sample, axis=0)

# -----------------------
# Predict for a single file
# -----------------------
def predict_file(file_path, model_path):
    # Load the trained model
    model = joblib.load(model_path)
    print(f"Loaded model: {model_path}\n")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    sample_raw = load_json_data(file_path)
    sample_feat = extract_eeg_features(sample_raw).reshape(1, -1)

    pred = model.predict(sample_feat)[0]
    print(f"{os.path.basename(file_path)} → Predicted: {pred}")

# -----------------------
# Run prediction
# -----------------------
if __name__ == "__main__":
    # Choose the single file to test
    file_to_test = "./dataset/sameed/right/sameed_right75.json"  # change as needed

    # Choose the correct model file
    model_file = "RF_movement_n200.joblib"  # change as needed

    predict_file(file_to_test, model_file)
