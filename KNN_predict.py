import json
import joblib
import numpy as np
from scipy.fftpack import fft


# --------------------------------------------------
# 1)  Feature extraction function (must match training)
# --------------------------------------------------
def extract_eeg_features(sample):
    """
    Parameters
    ----------
    sample : np.ndarray
        EEG sample of shape (16, 8, 16)

    Returns
    -------
    np.ndarray
        Feature vector of shape (48,)
    """
    feats = []
    for ch in range(8):
        ch_data = sample[:, ch, :]  # (16,16)
        ch_flat = ch_data.flatten()  # (256,)

        # time domain
        feats.append(np.mean(ch_flat))
        feats.append(np.std(ch_flat))
        feats.append(np.max(ch_flat))
        feats.append(np.min(ch_flat))

        # frequency domain
        fft_vals = np.abs(fft(ch_flat))
        feats.append(np.mean(fft_vals))
        feats.append(np.std(fft_vals))

    return np.array(feats)  # (48,)


# --------------------------------------------------
# 2)  Helper to read a JSON trial
# --------------------------------------------------
def load_json_sample(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # order keys numerically: "0ms", "1ms", &
    frames = [
        np.array(data[k]) for k in sorted(data, key=lambda x: int(x.replace("ms", "")))
    ]
    sample = np.stack(frames, axis=0)  # (16, 8, 16)
    return sample


# --------------------------------------------------
# 3)  Load trained model
# --------------------------------------------------
knn = joblib.load("knn_model_features_k1_3_users.joblib")

# --------------------------------------------------
# 4)  Predict on a new trial
# --------------------------------------------------
sample_raw = load_json_sample("./dataset/sameed/right/sameed_right79.json")
sample_feat = extract_eeg_features(sample_raw)  # shape (48,)

prediction = knn.predict([sample_feat])[0]

class_names = [
    "Left",
    "Right",
    "Help",
    "Stop",
    "Come",
    "Danger",
    "Eat",
    "Drink",
    "Restroom",
    "Hurt",
]
print("Predicted word:", class_names[prediction])