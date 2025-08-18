import numpy as np
import json
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Feature Extraction ---
def extract_eeg_features(sample):
    features = []
    for ch in range(8):
        ch_data = sample[:, ch, :]
        ch_flat = ch_data.flatten()
        # Time-domain
        features.append(np.mean(ch_flat))
        features.append(np.std(ch_flat))
        features.append(np.max(ch_flat))
        features.append(np.min(ch_flat))
        # Frequency-domain
        fft_vals = np.abs(fft(ch_flat))
        features.append(np.mean(fft_vals))
        features.append(np.std(fft_vals))
    return np.array(features)

# --- Load a single EEG JSON sample ---
def load_json_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        sample = []
        for time_key in sorted(data.keys(), key=lambda x: int(x.replace("ms", ""))):
            sample.append(np.array(data[time_key]))
        return np.stack(sample, axis=0)

# --- Load all samples from a folder ---
def load_action_samples(dataset_path, action, subject, indices):
    samples = []
    for i in indices:
        path = os.path.join(dataset_path, subject, action, f"{subject}_{action}{i}.json")
        if os.path.exists(path):
            samples.append(load_json_data(path))
        else:
            print(f"⚠ Skipping missing file: {path}")
            continue
    return np.array(samples)


# --- Group definitions (adjust as you like) ---
groups = {
    "movement": ["left", "right", "stop", "come"],
    "others": ["danger", "eat", "drink", "help", "restroom"]
}

# --- Base dataset path ---
DATASET_PATH = "./dataset"

# --- Train each group separately ---
for group_name, actions in groups.items():
    print(f"\n=== Training model for group: {group_name} ===")
    
    raw_data = []
    labels = []
    
    for action in actions:
        if action == "eat":
            data = load_action_samples(DATASET_PATH, action, "sameed2", range(101, 151))
        else:
            data = load_action_samples(DATASET_PATH, action, "sameed", range(1, 76))
        
        raw_data.extend(data)
        labels.extend([action] * len(data))
    
    # Feature extraction
    X_features = np.array([extract_eeg_features(sample) for sample in raw_data])
    y_labels = np.array(labels)
    print(f"Group '{group_name}' feature shape: {X_features.shape}")
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    # Cross-validation for best n_estimators
    n_values = [50, 100, 150, 200]
    cv_scores = []
    print("\n🔍 Cross-validation results:")
    for n in n_values:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        scores = cross_val_score(rf, X_train, y_train, cv=5)
        mean_score = np.mean(scores)
        cv_scores.append(mean_score)
        print(f"  n_estimators={n}: mean accuracy = {mean_score:.4f}")
    
    best_n = n_values[np.argmax(cv_scores)]
    print(f"Best n_estimators for '{group_name}': {best_n}")
    
    # Final model
    rf_final = RandomForestClassifier(n_estimators=best_n, random_state=42)
    rf_final.fit(X_train, y_train)
    
    # Validation performance
    y_pred = rf_final.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred))
    
    # Save model
    model_filename = f"RF_{group_name}_n{best_n}.joblib"
    joblib.dump(rf_final, model_filename)
    print(f"Saved model: {model_filename}")