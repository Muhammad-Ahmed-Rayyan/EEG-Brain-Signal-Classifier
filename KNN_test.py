import numpy as np
import json
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# --- Feature Extraction Function ---
def extract_eeg_features(sample):
    """
    Extracts 6 features per channel from a single EEG sample of shape (16, 8, 16)
    Total output shape: (48,)
    """
    features = []
    for ch in range(8):
        ch_data = sample[:, ch, :]  # shape: (16, 16)
        ch_flat = ch_data.flatten()  # shape: (256,)

        # Time-domain features
        features.append(np.mean(ch_flat))
        features.append(np.std(ch_flat))
        features.append(np.max(ch_flat))
        features.append(np.min(ch_flat))

        # Frequency-domain features
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
            sample.append(np.array(data[time_key]))  # shape: (8, 16)
        return np.stack(sample, axis=0)  # shape: (16, 8, 16)


# --- Load all data from folders ---


sameed_left1 = [
    load_json_data(f"./dataset/sameed/left/sameed_left{i}.json") for i in range(1, 76)
]
sameed_right1 = [
    load_json_data(f"./dataset/sameed/right/sameed_right{i}.json") for i in range(1, 76)
]
sameed_help1 = [
    load_json_data(f"./dataset/sameed/help/sameed_help{i}.json") for i in range(1, 76)
]
sameed_eat1 = [
    load_json_data(f"./dataset/sameed/eat/sameed_eat{i}.json") for i in range(1, 76)
]

stop_data = [
    load_json_data(f"./dataset/sameed/stop/sameed_stop{i}.json") for i in range(1, 76)
]
come_data = [
    load_json_data(f"./dataset/sameed/come/sameed_come{i}.json") for i in range(1, 76)
]
danger_data = [
    load_json_data(f"./dataset/sameed/danger/sameed_danger{i}.json")
    for i in range(1, 76)
]

drink_data = [
    load_json_data(f"./dataset/sameed/drink/sameed_drink{i}.json") for i in range(1, 76)
]
restroom_data = [
    load_json_data(f"./dataset/sameed/restroom/sameed_restroom{i}.json")
    for i in range(1, 76)
]

# --- Load sameed2 data ---
sameed_left2 = [
    load_json_data(f"./dataset/sameed2/left/sameed_left{i}.json")
    for i in range(101, 151)
]

sameed_right2 = [
    load_json_data(f"./dataset/sameed2/right/sameed_right{i}.json")
    for i in range(101, 151)
]

sameed_eat2 = [
    load_json_data(f"./dataset/sameed2/eat/sameed_eat{i}.json") for i in range(101, 151)
]
sameed_help2 = [
    load_json_data(f"./dataset/sameed2/help/sameed_help{i}.json")
    for i in range(101, 151)
]


left_data = sameed_left1 + sameed_left2
right_data = sameed_right1 + sameed_right2
eat_data = sameed_eat1 + sameed_eat2
help_data = sameed_help1 + sameed_help2

# --- Combine all data ---
raw_data = (
    left_data
    + right_data
    + help_data
    + stop_data
    + come_data
    + danger_data
    + eat_data
    + drink_data
    + restroom_data
)
labels = (
    [0] * len(left_data)
    + [1] * len(right_data)
    + [2] * len(help_data)
    + [3] * len(stop_data)
    + [4] * len(come_data)
    + [5] * len(danger_data)
    + [6] * len(eat_data)
    + [7] * len(drink_data)
    + [8] * len(restroom_data)
)

# --- Feature extraction ---
X_features = np.array([extract_eeg_features(sample) for sample in raw_data])
y_data = np.array(labels)

print(f"\n Feature shape: {X_features.shape}")  # (420, 48)

# --- Train/test split ---
X_train, X_val, y_train, y_val = train_test_split(
    X_features, y_data, test_size=0.2, random_state=42, stratify=y_data
)

print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

# --- Cross-validation to find best k ---
k_values = list(range(1, 10, 2))  # Try odd k: 1, 3, 5, 7, 9
cv_scores = []

print("\n Cross-validation results:")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    mean_score = np.mean(scores)
    cv_scores.append(mean_score)
    print(f"  k={k}: mean accuracy = {mean_score:.4f}")

# --- Train final model with best k ---
best_k = k_values[np.argmax(cv_scores)]
print(f"\n Best k: {best_k} with accuracy = {max(cv_scores):.4f}")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)

# --- Evaluate on validation set ---
y_pred = knn_final.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"\n<� Validation Accuracy with k={best_k}: {accuracy:.4f}")
print("\n=� Classification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        target_names=[
            "Left",
            "Right",
            "Help",
            "Stop",
            "Come",
            "Danger",
            "Eat",
            "Drink",
            "Restroom",
        ],
    )
)

# --- Save the final model ---
model_filename = f"KNN_k{best_k}_sameed_combined.joblib"
joblib.dump(knn_final, model_filename)
print(f"\ns Model saved to: {model_filename}")
